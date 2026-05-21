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

from graph_net.agent import ExtractionStatus, GraphNetAgent  # noqa: E402

import torch  # noqa: E402

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


def get_device_type() -> str:
    """Return 'cuda' if torch CUDA is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def get_gpu_ids(args) -> List[int]:
    """Resolve GPU indices from args or fall back to nvidia-smi.

    Returns:
        List of GPU indices if specified or detected.
        Falls back to _get_default_gpus() if args.gpus is None, empty, or invalid.
    """
    if args.gpus is not None:
        gpus_str = args.gpus.strip()
        if gpus_str:
            try:
                return [int(g.strip()) for g in gpus_str.split(",") if g.strip()]
            except ValueError:
                print(f"[WARN] Invalid --gpus value: {args.gpus}, using default GPUs")

    return _get_default_gpus()


DEFAULT_WORKSPACE = os.environ.get(
    "GRAPH_NET_EXTRACT_WORKSPACE", "/tmp/graphnet_workspace"
)


# ---------------------------------------------------------------------------
# Worker — runs in a dedicated subprocess, bound to one GPU
# ---------------------------------------------------------------------------


def worker_fn(
    worker_id: int,
    gpu_id: Optional[int],
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    workspace: str,
    hf_token: Optional[str],
    total: int,
    extract_timeout: int,
    verify_timeout: int,
    llm_retry: bool,
    max_model_size_b: float = 20.0,
) -> None:
    """
    Worker function, runs in a dedicated subprocess bound to a single GPU or CPU.
    Dynamically pulls tasks from task_queue and exits when the queue is empty.

    Args:
        worker_id:       Worker process index (e.g. 0)
        gpu_id:          CUDA device index (e.g. 2), None for CPU mode
        task_queue:      Shared task queue; each item is a model_id string
        result_queue:    Queue for reporting results back to the main process
        workspace:       Root workspace directory path
        hf_token:        HuggingFace token (optional)
        total:           Total task count (used for logging only)
        extract_timeout: Timeout in seconds for graph extraction subprocess
        verify_timeout:  Timeout in seconds for forward verification subprocess
        llm_retry:       Whether to enable LLM retry for failed extractions
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        prefix = f"[Worker-{worker_id} GPU:{gpu_id}]"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        prefix = f"[Worker-{worker_id} CPU]"
    # Pass workspace to the environment variable used by SubprocessGraphExtractor
    if "GRAPH_NET_EXTRACT_WORKSPACE" not in os.environ:
        os.environ[
            "GRAPH_NET_EXTRACT_WORKSPACE"
        ] = f"{workspace}/samples/transformers-auto-model"

    print(
        f"{prefix} Started (extract_timeout={extract_timeout}s, "
        f"verify_timeout={verify_timeout}s)",
        flush=True,
    )

    # Orphan watcher: if main process is killed with SIGKILL, worker becomes
    # orphaned (ppid == 1).  Detect this and kill all child process groups to
    # prevent GPU memory leaks from run_model.py subprocesses.
    import threading

    def _orphan_watcher():
        while True:
            time.sleep(5)
            if os.getppid() == 1:
                print(
                    f"{prefix} Parent died (orphaned), cleaning up child processes...",
                    flush=True,
                )
                # Multiple rounds to catch any late-starting children
                for _ in range(5):
                    from graph_net.agent.graph_extractor.subprocess_graph_extractor import (
                        kill_all_active_children,
                    )

                    kill_all_active_children()
                    time.sleep(1)
                os._exit(1)

    threading.Thread(target=_orphan_watcher, daemon=True).start()

    try:
        agent = GraphNetAgent(
            workspace=workspace,
            hf_token=hf_token,
            llm_retry=llm_retry,
            extract_timeout=extract_timeout,
            verify_timeout=verify_timeout,
            max_model_size_b=max_model_size_b,
        )
    except Exception as e:
        print(f"{prefix} Failed to initialize agent: {e}", flush=True)
        # Drain queue and mark remaining tasks as failed to avoid blocking the main process
        while True:
            try:
                mid = task_queue.get_nowait()
                result_queue.put(
                    {
                        "gpu": gpu_id if gpu_id is not None else worker_id,
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
        model_id = task_queue.get()
        if model_id is None:
            break

        print(f"{prefix} Extracting: {model_id}", flush=True)
        t0 = time.time()

        result_dict = {
            "gpu": gpu_id if gpu_id is not None else worker_id,
            "model_id": model_id,
        }
        try:
            status = agent.extract_sample(model_id)
            elapsed = time.time() - t0
            ok = status == ExtractionStatus.OK
            timeout_success = getattr(agent, "last_timeout_success", False)
            label = "OK" if ok else status.name.replace("_", " ")
            if ok and timeout_success:
                label = "OK(timeout)"
            print(f"{prefix} {label} {model_id} ({elapsed:.1f}s)", flush=True)
            result_dict["success"] = ok
            result_dict["status"] = status.value
            result_dict["timeout_success"] = timeout_success
            # Expose error category so the main process can decide policy
            rec = agent.error_classifier.get_record(model_id)
            if rec is not None:
                result_dict["error_category"] = rec.category.value
                result_dict["error_message"] = rec.message
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{prefix} ERROR {model_id}: {e} ({elapsed:.1f}s)", flush=True)
            result_dict["success"] = False
            result_dict["status"] = ExtractionStatus.ERROR.value
            result_dict["error"] = str(e)
            result_dict["timeout_success"] = False
            raw_cat = getattr(e, "error_category", None)
            if raw_cat is not None:
                result_dict["error_category"] = str(raw_cat)

        result_dict["elapsed"] = round(elapsed, 2)
        result_dict["timestamp"] = datetime.now().isoformat()
        result_queue.put(result_dict)
    print(f"{prefix} Worker finished (queue empty)", flush=True)


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
    timeout_success = sum(1 for d in details if d.get("timeout_success"))
    extract_success = sum(
        1
        for d in details
        if d.get("status")
        in (ExtractionStatus.OK.value, ExtractionStatus.VERIFY_FAILED.value)
    )
    failed = total - success
    rate = (success / total * 100) if total else 0.0
    timeout_rate = (timeout_success / total * 100) if total else 0.0
    extract_rate = (extract_success / total * 100) if total else 0.0
    print("\n" + "=" * 60)
    print("[SUMMARY] Parallel Extraction Summary")
    print("=" * 60)
    print(f"  Total   : {total}")
    print(f"  Success : {success} (verify ok)")
    print(f"  Timeout : {timeout_success} (verify skipped by timeout)")
    print(f"  Extract : {extract_success} (graph extracted)")
    print(f"  Failed  : {failed}")
    print(f"  Rate    : {rate:.2f}% (overall, timeout_success={timeout_rate:.2f}%)")
    print(f"  Extract : {extract_rate:.2f}% (extraction only)")
    # Per-GPU breakdown
    gpu_stats: Dict[int, Dict] = {}
    for d in details:
        g = d.get("gpu", -1)
        if g not in gpu_stats:
            gpu_stats[g] = {"total": 0, "success": 0, "extract": 0, "timeout": 0}
        gpu_stats[g]["total"] += 1
        if d.get("success"):
            gpu_stats[g]["success"] += 1
        if d.get("timeout_success"):
            gpu_stats[g]["timeout"] += 1
        if d.get("status") in (
            ExtractionStatus.OK.value,
            ExtractionStatus.VERIFY_FAILED.value,
        ):
            gpu_stats[g]["extract"] += 1

    label = "GPU" if results.get("gpus") else "Worker"
    print(f"\n  Per-{label}:")
    for g in sorted(gpu_stats):
        gs = gpu_stats[g]
        gr = (gs["success"] / gs["total"] * 100) if gs["total"] else 0.0
        er = (gs["extract"] / gs["total"] * 100) if gs["total"] else 0.0
        tr = (gs["timeout"] / gs["total"] * 100) if gs["total"] else 0.0
        print(
            f"    {label} {g}: success={gs['success']}/{gs['total']} ({gr:.1f}%), "
            f"extract={gs['extract']}/{gs['total']} ({er:.1f}%), "
            f"timeout={gs['timeout']}/{gs['total']} ({tr:.1f}%)"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel computation graph extraction from HuggingFace; one agent process per GPU or CPU"
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
        default=None,
        help="Comma-separated GPU indices to use (default: auto-detect all available GPUs)",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="Number of worker processes in CPU-only mode (default: half of CPU cores)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated filename with timestamp)",
    )
    parser.add_argument(
        "--extract-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for graph extraction (default: 1000 on GPU, 2000 on CPU)",
    )
    parser.add_argument(
        "--verify-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for forward verification (default: 300 on GPU, 600 on CPU)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Enable LLM retry for failed extractions",
    )
    parser.add_argument(
        "--max-model-size-b",
        type=str,
        default="auto",
        help="Maximum model size in billions of parameters to attempt. "
        "'auto' calculates from total RAM / workers (default); "
        "or specify manually, e.g. '10' for 10B.",
    )
    return parser.parse_args()


def _load_model_ids(args: argparse.Namespace) -> List[str]:
    if args.model_list:
        return load_models_from_file(args.model_list)
    if HUGGINGFACE_HUB_AVAILABLE:
        print(
            f"[INFO] Fetching {args.count} models from HuggingFace Hub (task={args.task})..."
        )
        return get_models_from_hf(task=args.task, limit=args.count)
    return []


def _resolve_config(args: argparse.Namespace):
    workspace = (
        args.workspace or os.getenv("GRAPH_NET_EXTRACT_WORKSPACE") or DEFAULT_WORKSPACE
    )
    print(f"[INFO] Workspace: {workspace}")

    # Decide GPU vs CPU mode: if --cpu-workers is set, force CPU-only mode.
    # If no CUDA available, also fall back to CPU mode.
    if (args.cpu_workers and args.cpu_workers > 0) or get_device_type() == "cpu":
        # CPU-only mode. Default to half of CPU cores to avoid overloading the
        # system, since each worker is a heavy process (model loading + graph
        # extraction).
        gpus = []
        num_workers = (
            args.cpu_workers if args.cpu_workers else max(1, (os.cpu_count() or 2) // 2)
        )
        print(f"[INFO] CPU-only mode: {num_workers} workers")
        extract_timeout = (
            args.extract_timeout if args.extract_timeout is not None else 2000
        )
        verify_timeout = args.verify_timeout if args.verify_timeout is not None else 600
    else:
        gpus = get_gpu_ids(args)
        num_workers = len(gpus)
        print(f"[INFO] GPU mode: {num_workers} workers on GPUs {gpus}")
        extract_timeout = (
            args.extract_timeout if args.extract_timeout is not None else 1000
        )
        verify_timeout = args.verify_timeout if args.verify_timeout is not None else 300

    # Resolve max_model_size_b
    if args.max_model_size_b == "auto":
        try:
            import psutil

            total_ram_gb = psutil.virtual_memory().total / 1024**3
        except ImportError:
            total_ram_gb = 256.0  # conservative fallback
        max_model_size_b = total_ram_gb * 0.7 / num_workers / 4
        print(
            f"[INFO] max_model_size_b=auto → {max_model_size_b:.1f}B "
            f"(RAM={total_ram_gb:.0f}GB, workers={num_workers})"
        )
    else:
        max_model_size_b = float(args.max_model_size_b)
        print(f"[INFO] max_model_size_b={max_model_size_b:.1f}B (manually set)")

    return (
        workspace,
        gpus,
        num_workers,
        extract_timeout,
        verify_timeout,
        max_model_size_b,
    )


def main() -> int:
    args = _parse_args()

    (
        workspace,
        gpus,
        num_workers,
        extract_timeout,
        verify_timeout,
        max_model_size_b,
    ) = _resolve_config(args)
    llm_retry = args.use_llm

    model_ids = _load_model_ids(args)
    if not model_ids:
        print("[ERROR] Empty model list, nothing to do")
        return 1

    if gpus:
        print(f"[INFO] Total models: {len(model_ids)}, GPU workers: {num_workers}")
    else:
        print(f"[INFO] Total models: {len(model_ids)}, CPU workers: {num_workers}")

    # --- Populate shared task queue ---
    task_queue: multiprocessing.Queue = multiprocessing.Queue()
    for mid in model_ids:
        task_queue.put(mid)
    # Sentinel: each worker gets one None to signal shutdown
    for _ in range(num_workers):
        task_queue.put(None)

    # --- Launch workers ---
    result_queue: multiprocessing.Queue = multiprocessing.Queue()

    start_time = datetime.now()
    worker_type = "GPU" if gpus else "CPU"
    print(
        f"\n[START] {start_time.strftime('%Y-%m-%d %H:%M:%S')} — launching {num_workers} {worker_type} workers\n"
    )

    processes = []
    for worker_id in range(num_workers):
        gpu_id = gpus[worker_id] if gpus else None
        p = multiprocessing.Process(
            target=worker_fn,
            args=(
                worker_id,
                gpu_id,
                task_queue,
                result_queue,
                workspace,
                args.hf_token,
                len(model_ids),
                extract_timeout,
                verify_timeout,
                llm_retry,
                max_model_size_b,
            ),
            name=f"worker-{worker_id}"
            + (f"-gpu{gpu_id}" if gpu_id is not None else "-cpu"),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Collect results ---
    details = []
    while len(details) < len(model_ids):
        try:
            entry = result_queue.get(timeout=5)
            details.append(entry)
            done = len(details)
            ok_so_far = sum(1 for d in details if d.get("success"))
            timeout_so_far = sum(1 for d in details if d.get("timeout_success"))
            extract_ok_so_far = sum(
                1
                for d in details
                if d.get("status")
                in (ExtractionStatus.OK.value, ExtractionStatus.VERIFY_FAILED.value)
            )
            print(
                f"[PROGRESS] {done}/{len(model_ids)} done, "
                f"success={ok_so_far/done*100:.1f}%(timeout_success={timeout_so_far/done*100:.1f}%), "
                f"extract={extract_ok_so_far/done*100:.1f}%",
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
    success_count = sum(1 for d in details if d.get("success"))
    timeout_success_count = sum(1 for d in details if d.get("timeout_success"))
    extract_success_count = sum(
        1
        for d in details
        if d.get("status")
        in (ExtractionStatus.OK.value, ExtractionStatus.VERIFY_FAILED.value)
    )
    results = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": round((end_time - start_time).total_seconds(), 1),
        "gpus": gpus,
        "workspace": workspace,
        "total": len(details),
        "success": success_count,
        "timeout_success": timeout_success_count,
        "extract_success": extract_success_count,
        "failed": len(details) - success_count,
        "success_rate": 0.0,
        "timeout_success_rate": 0.0,
        "extract_success_rate": 0.0,
        "details": details,
    }
    if results["total"] > 0:
        results["success_rate"] = round(results["success"] / results["total"] * 100, 2)
        results["timeout_success_rate"] = round(
            results["timeout_success"] / results["total"] * 100, 2
        )
        results["extract_success_rate"] = round(
            results["extract_success"] / results["total"] * 100, 2
        )

    # --- Save results ---
    from graph_net.agent.utils.workspace_manager import (
        WorkspaceManager as _WorkspaceManager,
    )

    _ws = _WorkspaceManager(workspace)
    output_file = args.output or str(
        _ws.logs_and_lists_dir
        / f"parallel_extract_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    _save_results(results, output_file)

    # --- Print summary ---
    _print_summary(results)
    print(f"\n[DONE] Total elapsed: {results['elapsed_seconds']:.0f}s")

    return 0 if results["success_rate"] > 0 else 1


if __name__ == "__main__":
    # Linux defaults to fork; explicitly use spawn to avoid CUDA fork issues
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
