import argparse
import importlib.util
import paddle
from pathlib import Path
import sys
import os
from dataclasses import dataclass
from contextlib import contextmanager
import time
import math
import numpy as np
import platform
import traceback
import subprocess
import re

from graph_net.paddle import utils
from graph_net import path_utils
from graph_net import test_compiler_util

from graph_net.paddle.backend.graph_compiler_backend import GraphCompilerBackend
from graph_net.paddle.backend.cinn_backend import CinnBackend
from graph_net.paddle.backend.nope_backend import NopeBackend


registry_backend = {
    "cinn": CinnBackend(),
    "nope": NopeBackend(),
}


def get_compiler_backend(args) -> GraphCompilerBackend:
    assert args.compiler in registry_backend, f"Unknown compiler: {args.compiler}"
    return registry_backend[args.compiler]


def init_env(args):
    if test_compiler_util.is_gpu_device(args.device):
        paddle.set_flags({"FLAGS_cudnn_exhaustive_search": 1})


def get_compile_framework_version(args):
    if args.compiler in ["cinn", "nope"]:
        return paddle.__version__
    return "unknown"


def check_and_print_gpu_utilization(compiler):
    if paddle.device.is_compiled_with_cuda():
        device_id = int(paddle.device.get_device().split(":")[-1])
        device_count = paddle.device.cuda.device_count()
        gpu_util, mem_util = test_compiler_util.get_device_utilization(
            device_id, device_count, compiler.synchronize
        )
        if gpu_util is not None and mem_util is not None:
            print(
                f"Device status: gpu_id {device_id}, gpu_util {gpu_util:.2f}%, mem_util {mem_util:.2f}%",
                file=sys.stderr,
                flush=True,
            )


def load_class_from_file(file_path: str, class_name: str):
    file = Path(file_path).resolve()
    module_name = file.stem

    with open(file_path, "r", encoding="utf-8") as f:
        original_code = f.read()
    import_stmt = "import paddle"
    modified_code = f"{import_stmt}\n{original_code}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    compiled_code = compile(modified_code, filename=file, mode="exec")
    exec(compiled_code, module.__dict__)

    model_class = getattr(module, class_name, None)
    return model_class


def get_model(model_path):
    model_class = load_class_from_file(
        f"{model_path}/model.py", class_name="GraphModule"
    )
    return model_class()


def get_static_model(args, model):
    static_model = paddle.jit.to_static(
        model,
        input_spec=utils.get_input_spec(args.model_path),
        full_graph=True,
        backend=None,
    )
    static_model.eval()
    program = static_model.forward.concrete_program.main_program
    return static_model


def measure_performance(model_call, args, compiler, profile=False):
    runtime_seed = 1024
    stats = {}

    paddle.seed(runtime_seed)
    outs = model_call()

    # Warmup runs
    warmup_e2e_times = []
    for _ in range(args.warmup):
        duration_box = test_compiler_util.DurationBox(-1)
        with test_compiler_util.naive_timer(duration_box, compiler.synchronize):
            model_call()
        warmup_e2e_times.append(duration_box.value)
    compiler.synchronize()

    # Ensure the measuring time is not less than 100ms.
    min_trials = int(100 / np.mean(warmup_e2e_times[1:]))
    trials = max(args.trials, min_trials)

    hardware_name = test_compiler_util.get_hardward_name(args)
    print(
        f"[Profiling] Using device: {args.device} {hardware_name}, warm up {args.warmup}, trials {trials}",
        file=sys.stderr,
        flush=True,
    )

    if profile:
        import paddle.profiler as profiler

        p = profiler.Profiler()
        p.start()

    if test_compiler_util.is_gpu_device(args.device):
        """
        Acknowledgement: We evaluate the performance on both end-to-end and GPU-only timings,
        With reference to methods only based on CUDA events from KernelBench in https://github.com/ScalingIntelligence/KernelBench
        """

        e2e_times = []
        gpu_times = []

        for i in range(trials):
            # End-to-end timing (naive_timer)
            duration_box = test_compiler_util.DurationBox(-1)
            with test_compiler_util.naive_timer(duration_box, compiler.synchronize):
                # GPU-only timing (CUDA Events)
                start_event = paddle.device.Event(enable_timing=True)
                end_event = paddle.device.Event(enable_timing=True)

                start_event.record()
                model_call()
                end_event.record()

            if profile:
                p.step()

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
    else:  # CPU or other devices
        e2e_times = []
        for i in range(trials):
            duration_box = test_compiler_util.DurationBox(-1)
            with test_compiler_util.naive_timer(duration_box, compiler.synchronize):
                model_call()

            if profile:
                p.step()

            e2e_times.append(duration_box.value)
            print(
                f"Trial {i + 1}: e2e={duration_box.value:.4f} ms",
                file=sys.stderr,
                flush=True,
            )
        stats["e2e"] = test_compiler_util.get_timing_stats(e2e_times)

    if profile:
        p.stop()
        p.summary()

    return outs, stats


def check_outputs(args, expected_out, compiled_out):
    if isinstance(expected_out, paddle.Tensor):
        expected_out = [expected_out]
    if isinstance(compiled_out, paddle.Tensor):
        compiled_out = [compiled_out]

    eager_dtypes = [None] * len(expected_out)
    for i, tensor in enumerate(expected_out):
        eager_dtypes[i] = (
            str(tensor.dtype).replace("paddle.", "") if tensor is not None else "None"
        )

    compiled_dtypes = [None] * len(compiled_out)
    for i, tensor in enumerate(compiled_out):
        compiled_dtypes[i] = (
            str(tensor.dtype).replace("paddle.", "") if tensor is not None else "None"
        )

    type_match = test_compiler_util.check_output_datatype(
        args, eager_dtypes, compiled_dtypes
    )

    eager_shapes = [None] * len(expected_out)
    for i, tensor in enumerate(expected_out):
        eager_shapes[i] = tensor.shape if tensor is not None else None

    compiled_shapes = [None] * len(compiled_out)
    for i, tensor in enumerate(compiled_out):
        compiled_shapes[i] = tensor.shape if tensor is not None else None

    shape_match = test_compiler_util.check_output_shape(
        args, eager_shapes, compiled_shapes
    )

    def transfer_to_float(origin_outputs):
        outputs = []
        for item in origin_outputs:
            if (
                item is not None
                and isinstance(item, paddle.Tensor)
                and item.dtype not in [paddle.float32, paddle.float64]
            ):
                item = item.astype("float32")
            outputs.append(item)
        return outputs

    if type_match and shape_match:
        test_compiler_util.check_equal(
            args,
            expected_out,
            compiled_out,
            cmp_equal_func=utils.get_cmp_equal,
        )

        expected_out_fp32 = transfer_to_float(expected_out)
        compiled_out_fp32 = transfer_to_float(compiled_out)
        test_compiler_util.check_allclose(
            args,
            expected_out_fp32,
            compiled_out_fp32,
            cmp_all_close_func=utils.get_cmp_all_close,
            cmp_max_diff_func=utils.get_cmp_max_diff,
            cmp_mean_diff_func=utils.get_cmp_mean_diff,
            cmp_max_relative_diff_func=utils.get_cmp_max_relative_diff,
            cmp_mean_relative_diff_func=utils.get_cmp_mean_relative_diff,
        )


def test_single_model(args):
    compiler = get_compiler_backend(args)
    check_and_print_gpu_utilization(compiler)

    input_dict = utils.get_input_dict(args.model_path)
    model = get_model(args.model_path)
    model.eval()

    hardware_name = test_compiler_util.get_hardward_name(args)

    test_compiler_util.print_basic_config(
        args, hardware_name, get_compile_framework_version(args)
    )

    # Run on eager mode
    eager_success = False
    eager_time_stats = {}
    try:
        print("Run model in eager mode.", file=sys.stderr, flush=True)
        static_model = get_static_model(args, model)
        expected_out, eager_time_stats = measure_performance(
            lambda: static_model(**input_dict), args, compiler, profile=False
        )
        eager_success = True
    except Exception as e:
        print(
            f"Run model in eager mode failed: {str(e)}\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )

    # Run on compiling mode
    compiled_success = False
    compiled_time_stats = {}
    try:
        print("Run model in compiled mode.", file=sys.stderr, flush=True)
        input_spec = utils.get_input_spec(args.model_path)
        compiled_model = compiler(model, input_spec)
        compiled_out, compiled_time_stats = measure_performance(
            lambda: compiled_model(**input_dict), args, compiler, profile=False
        )
        compiled_success = True
    except Exception as e:
        print(
            f"Run model in compiled mode failed: {str(e)}\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )

    test_compiler_util.print_running_status(args, eager_success, compiled_success)
    if eager_success and compiled_success:
        check_outputs(args, expected_out, compiled_out)

    test_compiler_util.print_times_and_speedup(
        args, eager_time_stats, compiled_time_stats
    )


def main(args):
    assert os.path.isdir(args.model_path)
    assert args.compiler in {"cinn", "nope"}
    assert args.device in ["cuda", "dcu", "xpu", "cpu"]

    initalize_seed = 123
    test_compiler_util.set_seed(random_seed=initalize_seed)

    if path_utils.is_single_model_dir(args.model_path):
        test_single_model(args)
    else:
        test_compiler_util.test_multi_models(args, "paddle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test compiler performance.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model file(s), each subdirectory containing graph_net.json will be regarded as a model",
    )
    parser.add_argument(
        "--compiler",
        type=str,
        required=False,
        default="cinn",
        help="Path to customized compiler python file",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device for testing the compiler (e.g., 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--warmup", type=int, required=False, default=5, help="Number of warmup steps"
    )
    parser.add_argument(
        "--trials", type=int, required=False, default=10, help="Number of timing trials"
    )
    parser.add_argument(
        "--log-prompt",
        type=str,
        required=False,
        default="graph-net-test-compiler-log",
        help="Log prompt for performance log filtering.",
    )
    parser.add_argument(
        "--allow-list",
        type=str,
        required=False,
        default=None,
        help="Path to samples list, each line contains a sample path",
    )
    args = parser.parse_args()
    main(args=args)
