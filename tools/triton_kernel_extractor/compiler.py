"""Step 1: Multi-GPU parallel compilation of graph samples."""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

from .cache_pruner import prune_compilation_cache
from .config import PipelineConfig
from .sample_enumerator import compute_unique_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-sample compilation
# ---------------------------------------------------------------------------


def _is_already_compiled(
    log_file: Path,
    cache_dir: Path,
    unique_dir: str,
) -> bool:
    """Check whether a sample has already been compiled in a prior run.

    Mirrors the bash resume logic::

        { [ -f "$log_file" ] && grep -q '[Speedup][kernel]:' "$log_file"; } \
          || [ -d "$cache_dir/kept/$unique_dir" ] \
          || [ -d "$cache_dir/discarded/$unique_dir" ]
    """
    if log_file.is_file():
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace")
            if "[Speedup][kernel]:" in content:
                return True
        except OSError:
            pass

    if (cache_dir / "kept" / unique_dir).is_dir():
        return True
    if (cache_dir / "discarded" / unique_dir).is_dir():
        return True

    return False


def _compile_one_sample(
    sample_path: str,
    graph_dir: str,
    cache_dir: Path,
    gpu_id: int,
    progress_label: str,
    compiler_config: str | None = None,
) -> str:
    """Compile a single graph sample on a specific GPU.

    Returns one of ``"compiled"``, ``"skipped"``, or ``"failed"``.

    After each compile attempt, prune the sample-local TorchInductor cache
    immediately so that large intermediate artifacts do not accumulate across
    many samples.
    """
    unique_dir = compute_unique_dir(sample_path, graph_dir)

    sample_cache_dir = cache_dir / unique_dir
    log_file = sample_cache_dir / "test_compiler_log.log"

    if _is_already_compiled(log_file, cache_dir, unique_dir):
        logger.info("%s SKIP: %s", progress_label, sample_path)
        return "skipped"

    # Remove incomplete cache from a prior interrupted attempt.
    if sample_cache_dir.exists():
        shutil.rmtree(sample_cache_dir)
    sample_cache_dir.mkdir(parents=True)

    logger.info("%s Compiling: %s", progress_label, sample_path)

    # Build a clean environment for the compiler subprocess.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TORCH_COMPILE_DEBUG"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(sample_cache_dir)
    # Ensure graph_net_bench is importable by the subprocess (tools/ -> repo root).
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graph_net_bench.torch.test_compiler",
            "--model-path",
            sample_path,
            "--kernel-time",
            "--warmup",
            "5",
            "--trials",
            "100",
            *(["--config", compiler_config] if compiler_config else []),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Write combined stdout+stderr to the log file (matches bash "> log 2>&1").
    log_file.write_text(result.stdout or "", encoding="utf-8")

    # Copy the original graph source into the cache.
    og_subdir = sample_cache_dir / "original_graph"
    og_subdir.mkdir(exist_ok=True)
    model_src = Path(sample_path)
    if model_src.is_dir():
        for item in model_src.iterdir():
            dest = og_subdir / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dest))

    prune_compilation_cache(sample_cache_dir)

    if result.returncode != 0:
        return "failed"
    return "compiled"


# ---------------------------------------------------------------------------
# Per-GPU sequential chunk worker
# ---------------------------------------------------------------------------


def _compile_chunk(
    samples: list[str],
    graph_dir: str,
    cache_dir: Path,
    gpu_id: int,
    compiler_config: str | None = None,
) -> dict[str, int]:
    """Process a chunk of samples sequentially on one GPU.

    This function is the top-level callable submitted to the process pool.
    Each invocation runs in its own process, isolating CUDA state.
    """
    total = len(samples)
    stats = {"compiled": 0, "skipped": 0, "failed": 0}

    for idx, sample_path in enumerate(samples, 1):
        label = f"[GPU{gpu_id} {idx}/{total}]"
        status = _compile_one_sample(
            sample_path=sample_path,
            graph_dir=graph_dir,
            cache_dir=cache_dir,
            gpu_id=gpu_id,
            progress_label=label,
            compiler_config=compiler_config,
        )
        stats[status] += 1

    logger.info(
        "[GPU%d] Done: %d compiled, %d skipped, %d failed (total: %d)",
        gpu_id,
        stats["compiled"],
        stats["skipped"],
        stats["failed"],
        total,
    )
    return stats


# ---------------------------------------------------------------------------
# Multi-GPU orchestrator
# ---------------------------------------------------------------------------


def compile_all_samples(
    samples: list[str],
    config: PipelineConfig,
) -> dict[str, int]:
    """Split samples across GPUs round-robin and compile in parallel.

    Each GPU gets its own worker process that processes its chunk
    sequentially, matching the original bash behaviour of one
    ``compile_worker`` per GPU.

    Returns aggregated ``{"compiled": N, "skipped": N, "failed": N}``.
    """
    gpu_ids = config.gpu_ids
    num_gpus = len(gpu_ids)

    # Build base64-encoded config for test_compiler --config, if needed.
    compiler_config: str | None = None
    if config.max_autotune_no_cudagraphs:
        config_dict = {"mode": "max-autotune-no-cudagraphs"}
        compiler_config = base64.b64encode(json.dumps(config_dict).encode()).decode()

    # Round-robin assignment (mirrors bash: gpu_id = GPU_IDS[local_idx % NUM_GPUS]).
    chunks: dict[int, list[str]] = {gid: [] for gid in gpu_ids}
    for idx, sample in enumerate(samples):
        gid = gpu_ids[idx % num_gpus]
        chunks[gid].append(sample)

    aggregated: dict[str, int] = {"compiled": 0, "skipped": 0, "failed": 0}

    # Use one process per GPU.  max_workers == num_gpus ensures no GPU
    # contention.
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        future_to_gpu: dict[Future[dict[str, int]], int] = {}

        for gid in gpu_ids:
            chunk = chunks[gid]
            if not chunk:
                continue
            future = executor.submit(
                _compile_chunk,
                samples=chunk,
                graph_dir=str(config.graph_dir),
                cache_dir=config.output_dir,
                gpu_id=gid,
                compiler_config=compiler_config,
            )
            future_to_gpu[future] = gid
            logger.info("  Launched worker GPU %d (%d samples)", gid, len(chunk))

        logger.info("  Waiting for %d workers...", len(future_to_gpu))

        has_errors = False
        for future in as_completed(future_to_gpu):
            gid = future_to_gpu[future]
            try:
                stats = future.result()
                for key in aggregated:
                    aggregated[key] += stats[key]
            except Exception:
                has_errors = True
                logger.exception("Worker GPU %d raised an exception", gid)

        if has_errors:
            logger.warning("WARNING: Some workers had errors. Check logs for details.")

    return aggregated
