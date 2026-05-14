#!/usr/bin/env python3
"""
Parallel extraction script: dynamic scheduling, all models placed in a shared task queue,
each GPU worker picks up tasks when idle.

Usage examples:
    # Load model list from file (one model_id per line)
    python parallel_extract.py --model-list models.txt

    # Fetch 400 models from HuggingFace Hub
    python parallel_extract.py --count 400

    # Specify workspace and HF token
    python parallel_extract.py --model-list models.txt \
        --workspace /data/graphnet_workspace \
        --hf-token YOUR_TOKEN

    # Specify GPUs to use (default: auto-detect all available GPUs)
    python parallel_extract.py --model-list models.txt --gpus 0,1,2,3
"""

import argparse
import json
import multiprocessing
import os
import queue
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ensure graph_net is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_GRAPHNET_ROOT = _SCRIPT_DIR.parent.parent  # GraphNet/
if str(_GRAPHNET_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRAPHNET_ROOT))

from graph_net.agent import GraphNetAgent  # noqa: E402

try:
    from huggingface_hub import list_models as _hf_list_models

    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False


def load_models_from_file(path: str) -> List[str]:
    models = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                models.append(line)
    return models


def get_models_from_hf(task: Optional[str] = None, limit: int = 100) -> List[str]:
    return [
        m.modelId
        for m in _hf_list_models(task=task, limit=limit, sort="downloads", direction=-1)
    ]


def _get_default_gpus() -> List[int]:
    """Detect available GPU indices from environment or nvidia-smi."""
    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        try:
            return [int(g.strip()) for g in cvd.split(",") if g.strip()]
        except ValueError:
            pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpus = [
                int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()
            ]
            if gpus:
                return gpus
    except Exception:
        pass
    return [0]


DEFAULT_GPUS = _get_default_gpus()
DEFAULT_WORKSPACE = os.environ.get(
    "GRAPH_NET_EXTRACT_WORKSPACE",
    os.path.expanduser("~/graphnet_workspace"),
)


# ---------------------------------------------------------------------------
# Worker — runs in a dedicated subprocess, bound to one GPU
# ---------------------------------------------------------------------------


def _worker(
    gpu_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    workspace: str,
    hf_token: Optional[str],
    total: int,
) -> None:
    """
    Worker function, runs in a dedicated subprocess bound to a single GPU.
    Dynamically pulls tasks from task_queue and exits when the queue is empty.

    Args:
        gpu_id:       CUDA device index (e.g. 2)
        task_queue:   Shared task queue; each item is a model_id string
        result_queue: Queue for reporting results back to the main process
        workspace:    Root workspace directory path
        hf_token:     HuggingFace token (optional)
        total:        Total task count (used for logging only)
    """
    # Bind GPU: subprocess only sees this card, internal code can use cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Pass workspace to the environment variable used by SubprocessGraphExtractor
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace

    print(f"[GPU {gpu_id}] Worker started", flush=True)

    try:
        agent = GraphNetAgent(workspace=workspace, hf_token=hf_token, llm_retry=False)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to initialize agent: {e}", flush=True)
        # Drain queue and mark remaining tasks as failed to avoid blocking the main process
        while True:
            try:
                mid = task_queue.get_nowait()
                result_queue.put(
                    {
                        "gpu": gpu_id,
                        "model_id": mid,
                        "success": False,
                        "error": str(e),
                        "elapsed": 0.0,
                    }
                )
            except queue.Empty:
                break
        return

    while True:
        try:
            model_id = task_queue.get_nowait()
        except queue.Empty:
            break

        print(f"[GPU {gpu_id}] Extracting: {model_id}", flush=True)
        t0 = time.time()
        try:
            success = agent.extract_sample(model_id)
            elapsed = time.time() - t0
            status = "OK" if success else "FAIL"
            print(f"[GPU {gpu_id}] {status} {model_id} ({elapsed:.1f}s)", flush=True)
            result_queue.put(
                {
                    "gpu": gpu_id,
                    "model_id": model_id,
                    "success": success,
                    "elapsed": round(elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[GPU {gpu_id}] ERROR {model_id}: {e} ({elapsed:.1f}s)", flush=True)
            result_queue.put(
                {
                    "gpu": gpu_id,
                    "model_id": model_id,
                    "success": False,
                    "error": str(e),
                    "elapsed": round(elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    print(f"[GPU {gpu_id}] Worker finished (queue empty)", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_results(results: Dict, output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] Results saved to: {output_path}")


def _print_summary(results: Dict) -> None:
    details = results.get("details", [])
    total = len(details)
    success = sum(1 for d in details if d.get("success"))
    failed = total - success
    rate = (success / total * 100) if total else 0.0
    print("\n" + "=" * 60)
    print("[SUMMARY] Parallel Extraction Summary")
    print("=" * 60)
    print(f"  Total  : {total}")
    print(f"  Success: {success}")
    print(f"  Failed : {failed}")
    print(f"  Rate   : {rate:.2f}%")
    # Per-GPU breakdown
    gpu_stats: Dict[int, Dict] = {}
    for d in details:
        g = d.get("gpu", -1)
        if g not in gpu_stats:
            gpu_stats[g] = {"total": 0, "success": 0}
        gpu_stats[g]["total"] += 1
        if d.get("success"):
            gpu_stats[g]["success"] += 1
    print("\n  Per-GPU:")
    for g in sorted(gpu_stats):
        gs = gpu_stats[g]
        gr = (gs["success"] / gs["total"] * 100) if gs["total"] else 0.0
        print(f"    GPU {g}: {gs['success']}/{gs['total']} ({gr:.1f}%)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel computation graph extraction from HuggingFace; one agent process per GPU"
    )
    parser.add_argument(
        "--model-list",
        type=str,
        default=None,
        help="Path to model list file (one model_id per line, lines starting with # are comments)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of models to fetch from HuggingFace Hub (used when --model-list is not set, default 100)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="HuggingFace task filter (e.g. text-classification)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Root workspace directory (default: {DEFAULT_WORKSPACE} or GRAPH_NET_EXTRACT_WORKSPACE env var)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API Token (required for private models)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=",".join(str(g) for g in DEFAULT_GPUS),
        help=f"Comma-separated GPU indices to use (default: {','.join(str(g) for g in DEFAULT_GPUS)})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated filename with timestamp)",
    )

    args = parser.parse_args()

    # --- Resolve workspace ---
    workspace = (
        args.workspace or os.getenv("GRAPH_NET_EXTRACT_WORKSPACE") or DEFAULT_WORKSPACE
    )
    print(f"[INFO] Workspace: {workspace}")

    # --- Parse GPU list ---
    try:
        gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    except ValueError:
        print(f"[ERROR] Invalid --gpus value: {args.gpus}")
        return 1
    if not gpus:
        print("[ERROR] No GPUs specified")
        return 1
    print(f"[INFO] GPUs: {gpus}")

    # --- Load model list ---
    if args.model_list:
        model_ids = load_models_from_file(args.model_list)
    elif HUGGINGFACE_HUB_AVAILABLE:
        print(
            f"[INFO] Fetching {args.count} models from HuggingFace Hub (task={args.task})..."
        )
        model_ids = get_models_from_hf(task=args.task, limit=args.count)
    else:
        print("[ERROR] No model list provided and huggingface_hub not available")
        return 1

    if not model_ids:
        print("[ERROR] Empty model list, nothing to do")
        return 1

    print(f"[INFO] Total models: {len(model_ids)}, workers: {len(gpus)}")

    # --- Populate shared task queue ---
    task_queue: multiprocessing.Queue = multiprocessing.Queue()
    for mid in model_ids:
        task_queue.put(mid)

    # --- Launch workers ---
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    processes = []

    start_time = datetime.now()
    print(
        f"\n[START] {start_time.strftime('%Y-%m-%d %H:%M:%S')} — launching {len(gpus)} workers\n"
    )

    for gpu_id in gpus:
        p = multiprocessing.Process(
            target=_worker,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                workspace,
                args.hf_token,
                len(model_ids),
            ),
            name=f"worker-gpu{gpu_id}",
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Collect results ---
    details = []
    total_expected = len(model_ids)

    while len(details) < total_expected:
        try:
            entry = result_queue.get(timeout=5)
            details.append(entry)
            done = len(details)
            success_so_far = sum(1 for d in details if d.get("success"))
            print(
                f"[PROGRESS] {done}/{total_expected} done, "
                f"success rate so far: {success_so_far/done*100:.1f}%",
                flush=True,
            )
        except Exception:
            # Check if all workers are done
            alive = [p for p in processes if p.is_alive()]
            if not alive:
                break

    # Wait for all workers to exit cleanly
    for p in processes:
        p.join(timeout=10)

    end_time = datetime.now()
    elapsed_total = (end_time - start_time).total_seconds()

    # --- Build result summary ---
    results = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": round(elapsed_total, 1),
        "gpus": gpus,
        "workspace": workspace,
        "total": len(details),
        "success": sum(1 for d in details if d.get("success")),
        "failed": sum(1 for d in details if not d.get("success")),
        "success_rate": 0.0,
        "details": details,
    }
    if results["total"] > 0:
        results["success_rate"] = round(results["success"] / results["total"] * 100, 2)

    # --- Save results ---
    output_file = (
        args.output or f"parallel_extract_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    _save_results(results, output_file)

    # --- Print summary ---
    _print_summary(results)
    print(f"\n[DONE] Total elapsed: {elapsed_total:.0f}s")

    return 0 if results["success_rate"] > 0 else 1


if __name__ == "__main__":
    # Linux defaults to fork; explicitly use spawn to avoid CUDA fork issues
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
