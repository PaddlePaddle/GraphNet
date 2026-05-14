"""CLI entry point for the triton kernel extraction pipeline.

Subcommands
-----------

**extract** (default when no subcommand is given)::

    python3 -m tools.triton_kernel_extractor [extract] \\
        --graph-dir /data/graphs/typical_subgraphs \\
        --output-dir /data/output/typical_inductor_dump \\
        [--allow-list paths.txt] \\
        [--gpu-ids 0 2 5 7]

**analyze**::

    python3 -m tools.triton_kernel_extractor analyze \\
        <cache_dir> [--output-dir DIR]

When ``--allow-list`` is provided, sample paths are read from the file and
resolved relative to ``--graph-dir``.  Otherwise ``--graph-dir`` is scanned
recursively for ``model.py`` files.

When ``--gpu-ids`` is omitted the script auto-detects all available GPUs
by parsing the output of ``nvidia-smi -L``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU detection (shared by the extract subcommand)
# ---------------------------------------------------------------------------


def _detect_gpu_ids() -> list[int]:
    """Auto-detect available GPU IDs.

    Priority order (matching the original bash script):
      1. ``CUDA_VISIBLE_DEVICES`` environment variable
      2. ``nvidia-smi -L`` output

    Returns a list of integer GPU indices.  Raises ``RuntimeError``
    when no GPUs are found.
    """
    # Priority 1: honour CUDA_VISIBLE_DEVICES if set.
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_env:
        try:
            return [int(x) for x in cuda_env.split(",") if x.strip()]
        except ValueError:
            pass  # Fall through to nvidia-smi.

    # Priority 2: auto-detect from nvidia-smi.
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        ids = [int(m) for m in re.findall(r"GPU (\d+):", result.stdout)]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        ids = []

    if not ids:
        raise RuntimeError(
            "No GPUs detected. Pass --gpu-ids explicitly or check nvidia-smi."
        )
    return ids


# ---------------------------------------------------------------------------
# Subcommand: extract
# ---------------------------------------------------------------------------


def _add_extract_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="Run the compilation and extraction pipeline.",
        description=(
            "Compile graph samples and extract "
            "(subgraph, triton_kernel, ptx) triples."
        ),
    )
    parser.add_argument(
        "--allow-list",
        type=Path,
        default=None,
        help=(
            "Text file with sample paths (one per line), relative to --graph-dir. "
            "When omitted, --graph-dir is scanned recursively for model.py."
        ),
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help=(
            "Root directory of input graph data. "
            "Scanned recursively for model.py by default. "
            "When --allow-list is given, used as base for resolving relative paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for pipeline output (compilation cache, extracted kernels, analysis).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="*",
        default=None,
        help=(
            "GPU IDs to use for parallel compilation. "
            "Auto-detected via nvidia-smi when omitted."
        ),
    )
    parser.add_argument(
        "--enable-cache-analysis",
        action="store_true",
        default=False,
        help="Run cache analysis (statistics, plots) after extraction.",
    )
    parser.add_argument(
        "--max-autotune-no-cudagraphs",
        action="store_true",
        default=False,
        help=(
            "Enable Inductor max-autotune-no-cudagraphs mode during compilation "
            "(passes mode='max-autotune-no-cudagraphs' to torch.compile "
            "via graph_net_bench.torch.test_compiler)."
        ),
    )
    parser.set_defaults(func=_run_extract)


def _run_extract(args: argparse.Namespace) -> None:
    from .pipeline import run_pipeline

    gpu_ids = args.gpu_ids if args.gpu_ids else _detect_gpu_ids()

    config = PipelineConfig(
        gpu_ids=gpu_ids,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        allow_list=args.allow_list,
        max_autotune_no_cudagraphs=args.max_autotune_no_cudagraphs,
    )

    logger.info(
        "Using %d GPU(s): %s",
        len(config.gpu_ids),
        " ".join(str(g) for g in config.gpu_ids),
    )

    # Unset CUDA_VISIBLE_DEVICES in the parent process so that worker
    # subprocesses start with a clean slate and receive only the per-GPU
    # value assigned by compiler.py.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    run_pipeline(config, enable_cache_analysis=args.enable_cache_analysis)


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------


def _add_analyze_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "analyze",
        help="Analyze an inductor cache directory (logs, statistics, plots).",
        description=(
            "Concatenate compiler logs, compute speedup statistics, and "
            "generate distribution plots for an inductor cache directory."
        ),
    )
    parser.add_argument(
        "cache_dir",
        type=Path,
        help="Inductor cache directory to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=("Directory for analysis output. " "Defaults to <cache_dir>/analysis."),
    )
    parser.set_defaults(func=_run_analyze)


def _run_analyze(args: argparse.Namespace) -> None:
    from .cache_analyzer import analyze_cache

    cache_dir: Path = args.cache_dir
    output_dir: Path = args.output_dir or (cache_dir / "analysis")
    analyze_cache(cache_dir, output_dir)


# ---------------------------------------------------------------------------
# Backward-compatible argument detection
# ---------------------------------------------------------------------------


def _needs_implicit_extract(argv: list[str]) -> bool:
    """Return True if *argv* does not start with a known subcommand.

    When the first argument is a flag like ``--graph-dir`` rather than a
    subcommand name, we prepend ``extract`` for convenience.
    """
    if not argv:
        return False
    known_subcommands = {"extract", "analyze"}
    first = argv[0]
    if first in known_subcommands:
        return False
    if first in ("-h", "--help"):
        return False
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    if argv is None:
        argv = sys.argv[1:]

    # Backward compatibility: insert "extract" when no subcommand is given.
    if _needs_implicit_extract(argv):
        argv = ["extract"] + argv

    parser = argparse.ArgumentParser(
        prog="python3 -m tools.triton_kernel_extractor",
        description=(
            "Triton kernel extraction toolkit: compile, filter, extract, "
            "and analyze TorchInductor compilation caches."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_extract_parser(subparsers)
    _add_analyze_parser(subparsers)

    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
