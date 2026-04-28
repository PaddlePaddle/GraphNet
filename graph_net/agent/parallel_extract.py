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
        agent = GraphNetAgent(workspace=workspace, hf_token=hf_token, llm_retry=llm_retry)
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

    # --- 将所有模型填入共享任务队列 ---
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
                args.llm_retry,
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
    # multiprocessing 在 Linux 上默认 fork，显式设置 spawn 避免 CUDA fork 问题
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
