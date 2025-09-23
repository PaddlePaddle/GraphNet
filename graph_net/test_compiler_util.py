import sys
import json
import time
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class DurationBox:
    value: int


@contextmanager
def naive_timer(duration_box, synchronizer_func):
    synchronizer_func()
    start = time.time()
    yield
    synchronizer_func()
    end = time.time()
    duration_box.value = (end - start) * 1000  # Store in milliseconds


def get_timing_stats(elapsed_times):
    stats = {
        "mean": float(f"{np.mean(elapsed_times):.6g}"),
        "std": float(f"{np.std(elapsed_times):.6g}"),
        "min": float(f"{np.min(elapsed_times):.6g}"),
        "max": float(f"{np.max(elapsed_times):.6g}"),
    }
    return stats


def print_times_and_speedup(args, eager_stats, compiled_stats):
    print(
        f"{args.log_prompt} [Performance][eager]: {json.dumps(eager_stats)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"{args.log_prompt} [Performance][compiled]: {json.dumps(compiled_stats)}",
        file=sys.stderr,
        flush=True,
    )

    e2e_speedup = 0
    gpu_speedup = 0

    eager_e2e_time_ms = eager_stats.get("e2e", {}).get("mean", 0)
    compiled_e2e_time_ms = compiled_stats.get("e2e", {}).get("mean", 0)

    if eager_e2e_time_ms > 0 and compiled_e2e_time_ms > 0:
        e2e_speedup = eager_e2e_time_ms / compiled_e2e_time_ms

    if "cuda" in args.device:
        eager_gpu_time_ms = eager_stats.get("gpu", {}).get("mean", 0)
        compiled_gpu_time_ms = compiled_stats.get("gpu", {}).get("mean", 0)

        if eager_gpu_time_ms > 0 and compiled_gpu_time_ms > 0:
            gpu_speedup = eager_gpu_time_ms / compiled_gpu_time_ms

    if e2e_speedup > 0:
        print(
            f"{args.log_prompt} [Speedup][e2e]: {e2e_speedup:.5f}",
            file=sys.stderr,
            flush=True,
        )

    if "cuda" in args.device and gpu_speedup > 0:
        print(
            f"{args.log_prompt} [Speedup][gpu]: {gpu_speedup:.5f}",
            file=sys.stderr,
            flush=True,
        )
