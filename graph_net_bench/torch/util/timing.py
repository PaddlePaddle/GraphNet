import torch
import platform
import sys
from graph_net_bench import test_compiler_util


def get_hardward_name(device):
    hardware_name = "unknown"
    if "cuda" in device:
        hardware_name = torch.cuda.get_device_name(device)
    elif device == "cpu":
        hardware_name = platform.processor()
    return hardware_name


def get_compiler_version(compiler):
    if compiler in ["inductor", "nope", "unstable_to_stable"]:
        return torch.__version__
    elif compiler in ["tvm", "xla", "tensorrt", "bladedisc"]:
        # Assuming compiler object has a version attribute
        return f"{compiler.capitalize()} {compiler.version}"
    return "unknown"


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
        e2e_times, gpu_times = [], []
        for i in range(args.trials):
            duration_box = test_compiler_util.DurationBox(-1)
            with test_compiler_util.naive_timer(duration_box, compiler.synchronize):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                model_call()
                end_event.record()
                compiler.synchronize()

            gpu_time_ms = start_event.elapsed_time(end_event)
            e2e_times.append(duration_box.value)
            gpu_times.append(gpu_time_ms)
            print(
                f"Trial {i + 1}: e2e={duration_box.value:.5f} ms, gpu={gpu_time_ms:.5f} ms",
                file=sys.stderr,
                flush=True,
            )

        stats["e2e"] = test_compiler_util.get_timing_stats(e2e_times)
        stats["gpu"] = test_compiler_util.get_timing_stats(gpu_times)
    else:
        e2e_times = []
        for i in range(args.trials):
            duration_box = test_compiler_util.DurationBox(-1)
            with test_compiler_util.naive_timer(duration_box, compiler.synchronize):
                model_call()
            e2e_times.append(duration_box.value)
            print(
                f"Trial {i + 1}: e2e={duration_box.value:.5f} ms",
                file=sys.stderr,
                flush=True,
            )
        stats["e2e"] = test_compiler_util.get_timing_stats(e2e_times)

    return outs, stats
