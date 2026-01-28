import torch
import sys
from graph_net_bench import test_compiler_util


def measure_performance(model_call, args, compiler):
    stats = {}
    outs = model_call()
    # Warmup runs
    for _ in range(args.warmup):
        model_call()
    compiler.synchronize()

    print(
        f"[Profiling] Warm up {args.warmup}, Trials {args.trials}",
        file=sys.stderr,
        flush=True,
    )

    if "cuda" in args.device:
        torch.cuda.empty_cache()
        executor = CUDATrialExecutor(model_call, compiler)
    else:
        executor = NoneCUDATrialExecutor(model_call, compiler)

    timings = run_benchmark(args.trials, executor)

    stats = {
        name: test_compiler_util.get_timing_stats(values)
        for name, values in timings.items()
    }

    return outs, stats


def run_benchmark(trials, executor):
    results = {}

    for i in range(trials):
        timings = executor.run_one_trial()

        for k, v in timings.items():
            results.setdefault(k, []).append(v)

        log_trial(i + 1, timings)

    return results


def log_trial(idx, timings):
    msg = ", ".join(f"{k}={v:.5f} ms" for k, v in timings.items())
    print(f"Trial {idx}: {msg}", file=sys.stderr, flush=True)


class BaseTrialExecutor:
    def __init__(self, model_call, compiler):
        self.model_call = model_call
        self.compiler = compiler

    def run_one_trial(self):
        raise NotImplementedError


class NoneCUDATrialExecutor(BaseTrialExecutor):
    def run_one_trial(self):
        duration_box = test_compiler_util.DurationBox(-1)
        with test_compiler_util.naive_timer(duration_box, self.compiler.synchronize):
            self.model_call()
        return {"e2e": duration_box.value}


class CUDATrialExecutor(BaseTrialExecutor):
    def run_one_trial(self):
        duration_box = test_compiler_util.DurationBox(-1)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with test_compiler_util.naive_timer(duration_box, self.compiler.synchronize):
            start_event.record()
            self.model_call()
            end_event.record()
            self.compiler.synchronize()

        gpu_time = start_event.elapsed_time(end_event)

        return {
            "e2e": duration_box.value,
            "gpu": gpu_time,
        }
