#!/usr/bin/env python3
"""
并行抽取脚本：动态调度，所有模型放入共享任务队列，每张 GPU 的 worker 空闲时主动取任务。

用法示例：
    # 从文件读取模型列表（每行一个 model_id）
    python parallel_extract.py --model-list models.txt

    # 从 HuggingFace Hub 抓取 400 个模型
    python parallel_extract.py --count 400

    # 指定 workspace 和 HF token
    python parallel_extract.py --model-list models.txt \
        --workspace /home/luotao02/workspace \
        --hf-token YOUR_TOKEN

    # 指定使用的 GPU（默认 2,3,4,5）
    python parallel_extract.py --model-list models.txt --gpus 2,3,4,5
"""

import argparse
import json
import multiprocessing
import os
import queue
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# 确保能 import graph_net
_SCRIPT_DIR = Path(__file__).resolve().parent
_GRAPHNET_ROOT = _SCRIPT_DIR.parent.parent  # GraphNet/
if str(_GRAPHNET_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRAPHNET_ROOT))

from graph_net.agent import GraphNetAgent  # noqa: E402
from graph_net.agent.tests.test_batch_success_rate import (  # noqa: E402
    HUGGINGFACE_HUB_AVAILABLE,
    get_models_from_hf,
    load_models_from_file,
)

DEFAULT_GPUS = [2, 3, 4, 5]
DEFAULT_WORKSPACE = "/home/luotao02/workspace"


# ---------------------------------------------------------------------------
# Worker — runs in a dedicated subprocess, bound to one GPU
# ---------------------------------------------------------------------------


def _setup_nvidia_ld_library_path() -> None:
    """
    将 pip nvidia 包的 lib 目录注入 LD_LIBRARY_PATH 最前面，
    确保子进程加载正确版本的 NCCL/CUPTI/nvJitLink 等库，
    避免系统旧版库导致 undefined symbol 错误。
    """
    import glob

    base = "/usr/local/lib/python3.12/site-packages/nvidia"
    nvidia_libs = ":".join(glob.glob(f"{base}/*/lib"))
    if not nvidia_libs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if nvidia_libs not in current:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{nvidia_libs}:{current}" if current else nvidia_libs
        )


def _worker(
    gpu_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    workspace: str,
    hf_token: Optional[str],
    total: int,
    llm_retry: bool = False,
    max_model_size_b: float = 20.0,
    timeout: int = 1000,
) -> None:
    """
    Worker 函数，在独立子进程中运行，绑定到指定 GPU。
    动态从 task_queue 取任务，队列为空时退出。

    Args:
        gpu_id:       CUDA 设备编号（如 2）
        task_queue:   共享任务队列，每项为 model_id 字符串
        result_queue: 用于向主进程汇报结果的队列
        workspace:    工作目录根路径
        hf_token:     HuggingFace token（可选）
        total:        总任务数（仅用于日志显示）
    """
    # 绑定 GPU：子进程只看到这一张卡，内部用 cuda:0 即可
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 传递 workspace 给 SubprocessGraphExtractor 使用的环境变量
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace

    # 确保 pip nvidia 库优先加载（修复 NCCL/nvJitLink 符号缺失问题）
    _setup_nvidia_ld_library_path()

    print(f"[GPU {gpu_id}] Worker started", flush=True)

    try:
        agent = GraphNetAgent(workspace=workspace, hf_token=hf_token, llm_retry=llm_retry,
                              max_model_size_b=max_model_size_b, timeout=timeout)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to initialize agent: {e}", flush=True)
        # 把队列里剩余任务都标记为失败并排空，避免主进程死等
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


def _cpu_worker(
    worker_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    workspace: str,
    hf_token: Optional[str],
    total: int,
    llm_retry: bool = False,
    max_model_size_b: float = 20.0,
    timeout: int = 1000,
) -> None:
    """
    CPU-only worker，专门处理 oom_risk=high 的模型（不绑定 GPU）。
    通过隐藏所有 GPU 让模型自动跑在 CPU 上。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 对子进程隐藏所有 GPU
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace

    print(f"[CPU {worker_id}] Worker started", flush=True)

    try:
        agent = GraphNetAgent(workspace=workspace, hf_token=hf_token, llm_retry=llm_retry,
                              max_model_size_b=max_model_size_b, timeout=timeout)
    except Exception as e:
        print(f"[CPU {worker_id}] Failed to initialize agent: {e}", flush=True)
        while True:
            try:
                mid = task_queue.get_nowait()
                result_queue.put(
                    {
                        "gpu": f"cpu-{worker_id}",
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

        print(f"[CPU {worker_id}] Extracting: {model_id}", flush=True)
        t0 = time.time()
        try:
            success = agent.extract_sample(model_id)
            elapsed = time.time() - t0
            status = "OK" if success else "FAIL"
            print(f"[CPU {worker_id}] {status} {model_id} ({elapsed:.1f}s)", flush=True)
            result_queue.put(
                {
                    "gpu": f"cpu-{worker_id}",
                    "model_id": model_id,
                    "success": success,
                    "elapsed": round(elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[CPU {worker_id}] ERROR {model_id}: {e} ({elapsed:.1f}s)", flush=True)
            result_queue.put(
                {
                    "gpu": f"cpu-{worker_id}",
                    "model_id": model_id,
                    "success": False,
                    "error": str(e),
                    "elapsed": round(elapsed, 2),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    print(f"[CPU {worker_id}] Worker finished (queue empty)", flush=True)





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
    print("\n  Per-Worker:")
    for g in sorted(gpu_stats, key=lambda x: (str(x))):
        gs = gpu_stats[g]
        gr = (gs["success"] / gs["total"] * 100) if gs["total"] else 0.0
        label = f"CPU {g.split('-')[1]}" if isinstance(g, str) and g.startswith("cpu-") else f"GPU {g}"
        print(f"    {label}: {gs['success']}/{gs['total']} ({gr:.1f}%)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="并行从 HuggingFace 抽取计算图，每张 GPU 跑一个独立 agent 进程"
    )
    parser.add_argument(
        "--model-list",
        type=str,
        default=None,
        help="模型列表文件路径（每行一个 model_id，# 开头为注释）",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="从 HuggingFace Hub 抓取的模型数量（model-list 未指定时生效，默认 100）",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="HuggingFace 任务类型过滤（如 text-classification）",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"工作目录根路径（默认 {DEFAULT_WORKSPACE} 或 GRAPH_NET_EXTRACT_WORKSPACE 环境变量）",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API Token（私有模型需要）",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=",".join(str(g) for g in DEFAULT_GPUS),
        help=f"使用的 GPU 编号，逗号分隔（默认 {','.join(str(g) for g in DEFAULT_GPUS)}）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果 JSON 文件路径（默认自动生成带时间戳的文件名）",
    )
    parser.add_argument(
        "--llm-retry",
        action="store_true",
        default=False,
        help="失败时调用 LLM（ducc -p）兜底修复，默认关闭（并行场景下会增加耗时）",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="CPU-only worker 数量，用于处理 oom_risk=high 的模型"
             "（默认自动设为 CPU 核数的一半，最多 16）",
    )
    parser.add_argument(
        "--max-model-size-b",
        type=str,
        default="auto",
        help="允许抽取的最大模型参数量（单位：十亿）。"
             "'auto' 根据机器总 RAM / worker 数自动计算（默认）；"
             "也可手动指定，如 '10' 表示最大 10B。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1000,
        help="单个模型 subprocess 执行超时（秒），默认 1000s。"
             "建议按批次设置：≤1B→180, 2~3B→300, 4~7B→500, 8~13B→800。",
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

    # --- Resolve max_model_size_b ---
    cpu_workers_count = args.cpu_workers or max(1, os.cpu_count() // 2)
    total_workers = len(gpus) + cpu_workers_count
    if args.max_model_size_b == "auto":
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / 1024 ** 3
        except ImportError:
            total_ram_gb = 256.0  # conservative fallback
        max_model_size_b = total_ram_gb * 0.7 / total_workers / 4
        print(f"[INFO] max_model_size_b=auto → {max_model_size_b:.1f}B "
              f"(RAM={total_ram_gb:.0f}GB, workers={total_workers})")
    else:
        max_model_size_b = float(args.max_model_size_b)
        print(f"[INFO] max_model_size_b={max_model_size_b:.1f}B (manually set)")

    # --- 预分析 OOM 风险，GPU 模型优先入队，CPU 模型后入队 ---
    import json as _json
    from pathlib import Path as _Path
    from graph_net.agent.metadata_analyzer.config_metadata_analyzer import (
        ConfigMetadataAnalyzer,
    )

    _analyzer = ConfigMetadataAnalyzer()
    _models_cache = _Path(workspace) / "models"

    def _get_oom_risk(model_id: str) -> str:
        safe_id = model_id.replace("/", "--")
        # HF snapshot cache 目录格式：models--org--model/snapshots/hash
        for snap in _models_cache.glob(f"models--{safe_id}/snapshots/*/config.json"):
            try:
                cfg = _json.loads(snap.read_text())
                return _analyzer._estimate_oom_risk(cfg)
            except Exception:
                pass
        return "low"  # 未下载过的模型默认 low，下载后由模板决定

    gpu_models, cpu_models = [], []
    for mid in model_ids:
        if _get_oom_risk(mid) == "high":
            cpu_models.append(mid)
        else:
            gpu_models.append(mid)

    print(
        f"[INFO] Queue split: {len(gpu_models)} GPU models + "
        f"{len(cpu_models)} CPU models (oom_risk=high)"
    )

    # --- 将 GPU 模型先入队，CPU 模型后入队 ---
    gpu_task_queue: multiprocessing.Queue = multiprocessing.Queue()
    for mid in gpu_models:
        gpu_task_queue.put(mid)

    cpu_task_queue: multiprocessing.Queue = multiprocessing.Queue()
    for mid in cpu_models:
        cpu_task_queue.put(mid)

    # --- Launch GPU workers ---
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    processes = []

    n_cpu_workers = min(
        args.cpu_workers if args.cpu_workers is not None
        else max(1, (os.cpu_count() or 4) // 2),  # 默认：核数/2，至少1
        16,                                         # 上限16，避免过度竞争内存
        max(1, len(cpu_models)),                    # 不超过实际 CPU 任务数
    )
    total_workers = len(gpus) + (n_cpu_workers if cpu_models else 0)
    start_time = datetime.now()
    print(
        f"\n[START] {start_time.strftime('%Y-%m-%d %H:%M:%S')} — "
        f"launching {len(gpus)} GPU workers + {n_cpu_workers if cpu_models else 0} CPU workers\n"
    )

    for gpu_id in gpus:
        p = multiprocessing.Process(
            target=_worker,
            args=(
                gpu_id,
                gpu_task_queue,
                result_queue,
                workspace,
                args.hf_token,
                len(model_ids),
                args.llm_retry,
                max_model_size_b,
                args.timeout,
            ),
            name=f"worker-gpu{gpu_id}",
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Launch CPU workers ---
    if cpu_models:
        for i in range(n_cpu_workers):
            p = multiprocessing.Process(
                target=_cpu_worker,
                args=(
                    i,
                    cpu_task_queue,
                    result_queue,
                    workspace,
                    args.hf_token,
                    len(model_ids),
                    args.llm_retry,
                    max_model_size_b,
                    args.timeout,
                ),
                name=f"worker-cpu{i}",
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
    from graph_net.agent.utils.workspace_manager import WorkspaceManager as _WorkspaceManager
    _ws = _WorkspaceManager(workspace)
    output_file = (
        args.output
        or str(_ws.logs_and_lists_dir / f"parallel_extract_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
    )
    _save_results(results, output_file)

    # --- Print summary ---
    _print_summary(results)
    print(f"\n[DONE] Total elapsed: {elapsed_total:.0f}s")

    return 0 if results["success_rate"] > 0 else 1


if __name__ == "__main__":
    # multiprocessing 在 Linux 上默认 fork，显式设置 spawn 避免 CUDA fork 问题
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
