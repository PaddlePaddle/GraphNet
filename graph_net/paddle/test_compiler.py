import argparse
import importlib.util
import paddle
from pathlib import Path
import sys
import os
from dataclasses import dataclass
from contextlib import contextmanager
import time
import numpy as np
import random
import platform
import traceback

from graph_net.paddle import utils
from graph_net import path_utils
from graph_net import test_compiler_utils
from graph_net.benchmark_result import BenchmarkResult


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


def get_synchronizer_func(args):
    return paddle.device.synchronize


def get_model(args):
    model_class = load_class_from_file(
        f"{args.model_path}/model.py", class_name="GraphModule"
    )
    return model_class()


def get_input_dict(args):
    inputs_params = utils.load_converted_from_text(f"{args.model_path}")
    params = inputs_params["weight_info"]
    inputs = inputs_params["input_info"]

    params.update(inputs)
    state_dict = {k: utils.replay_tensor(v) for k, v in params.items()}
    return state_dict


def get_input_spec(args):
    inputs_params_list = utils.load_converted_list_from_text(f"{args.model_path}")
    input_spec = [None] * len(inputs_params_list)
    for i, v in enumerate(inputs_params_list):
        dtype = v["info"]["dtype"]
        shape = v["info"]["shape"]
        input_spec[i] = paddle.static.InputSpec(shape, dtype)
    return input_spec


def get_compiled_model(args, model):
    input_spec = get_input_spec(args)
    build_strategy = paddle.static.BuildStrategy()
    compiled_model = paddle.jit.to_static(
        model,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )
    compiled_model.eval()
    return compiled_model


def measure_performance(model_call, args, synchronizer_func):
    stats = {}

    # Warmup runs
    for _ in range(args.warmup):
        outs = model_call()
    synchronizer_func()

    if "cuda" in args.device:
        """
        Acknowledgement: We evaluate the performance on both end-to-end and GPU-only timings,
        With reference to methods only based on CUDA events from KernelBench in https://github.com/ScalingIntelligence/KernelBench
        """
        hardware_name = paddle.device.cuda.get_device_name(0)
        print(
            f"{args.log_prompt} [Profiling] Using device: {args.device} {hardware_name}, warm up {args.warmup}, trials {args.trials}"
        )

        e2e_times = []
        gpu_times = []

        for i in range(args.trials):
            # End-to-end timing (naive_timer)
            duration_box = test_compiler_utils.DurationBox(-1)
            with test_compiler_utils.naive_timer(duration_box, synchronizer_func):
                # GPU-only timing (CUDA Events)
                start_event = paddle.device.Event(enable_timing=True)
                end_event = paddle.device.Event(enable_timing=True)

                start_event.record()
                outs = model_call()
                end_event.record()

            gpu_time_ms = start_event.elapsed_time(end_event)
            e2e_times.append(duration_box.value)
            gpu_times.append(gpu_time_ms)
            print(
                f"Trial {i + 1}: e2e={duration_box.value:.5f} ms, gpu={gpu_time_ms:.5f} ms"
            )

        stats["e2e"] = test_compiler_utils.get_timing_stats(e2e_times)
        stats["gpu"] = test_compiler_utils.get_timing_stats(gpu_times)
    else:  # CPU or other devices
        hardware_name = platform.processor()
        print(
            f"[Profiling] Using device: {args.device} {hardware_name}, warm up {args.warmup}, trials {args.trials}"
        )

        e2e_times = []
        for i in range(args.trials):
            duration_box = test_compiler_utils.DurationBox(-1)
            with test_compiler_utils.naive_timer(duration_box, synchronizer_func):
                outs = model_call()
            print(f"Trial {i + 1}: e2e={duration_box.value:.4f} ms")
            e2e_times.append(duration_box.value)
        stats["e2e"] = test_compiler_utils.get_timing_stats(e2e_times)

    return outs, stats


def init_benchmark_result(args):
    if args.device == "cuda":
        hardware = paddle.device.cuda.get_device_name(0)
    elif args.device == "cpu":
        hardware = platform.processor()
    else:
        hardware = "unknown"

    if args.compiler == "cinn":
        compile_framework_version = paddle.__version__
    else:
        compile_framework_version = "unknown"

    result_data = BenchmarkResult(
        args=args,
        framework="PaddlePaddle",
        hardware=hardware,
        compile_framework_version=compile_framework_version,
    )
    return result_data


def check_outputs(args, expected_out, compiled_out):
    if isinstance(expected_out, paddle.Tensor):
        expected_out = [expected_out]
    if isinstance(compiled_out, paddle.Tensor):
        compiled_out = [compiled_out]

    eager_output_dtypes = [None] * len(expected_out)
    for i, tensor in enumerate(expected_out):
        if tensor is not None:
            eager_output_dtypes[i] = str(tensor.dtype)

    compiled_output_dtypes = [None] * len(compiled_out)
    for i, tensor in enumerate(compiled_out):
        if tensor is not None:
            compiled_output_dtypes[i] = str(tensor.dtype)

    is_output_consistent = len(expected_out) == len(compiled_out)
    for a, b in zip(expected_out, compiled_out):
        if (a is None and b is not None) or (a is not None and b is None):
            is_output_consistent = False
        if a is not None and b is not None and a.dtype != b.dtype:
            is_output_consistent = False

    def regular_outputs(origin_outputs):
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

    expected_out = regular_outputs(expected_out)
    compiled_out = regular_outputs(compiled_out)

    def print_cmp(key, func, **kwargs):
        try:
            cmp_ret = func(expected_out, compiled_out, **kwargs)
        except Exception as e:
            cmp_ret = f"{key} failed: {str(e)}\n{traceback.format_exc()}"
        print(
            f"{args.log_prompt} {key} model_path:{args.model_path} {cmp_ret}",
            file=sys.stderr,
        )

    print(
        f"{args.log_prompt} output_dtypes model_path:{args.model_path} eager:{eager_output_dtypes} compiled:{compiled_output_dtypes}",
        file=sys.stderr,
    )
    print_cmp("cmp.equal", get_cmp_equal)
    print_cmp("cmp.all_close_atol8_rtol8", get_cmp_all_close, atol=1e-8, rtol=1e-8)
    print_cmp("cmp.all_close_atol8_rtol5", get_cmp_all_close, atol=1e-8, rtol=1e-5)
    print_cmp("cmp.all_close_atol5_rtol5", get_cmp_all_close, atol=1e-5, rtol=1e-5)
    print_cmp("cmp.all_close_atol3_rtol2", get_cmp_all_close, atol=1e-3, rtol=1e-2)
    print_cmp("cmp.all_close_atol2_rtol1", get_cmp_all_close, atol=1e-2, rtol=1e-1)
    print_cmp("cmp.max_diff", get_cmp_max_diff)
    print_cmp("cmp.mean_diff", get_cmp_mean_diff)
    print_cmp("cmp.diff_count_atol8_rtol8", get_cmp_diff_count, atol=1e-8, rtol=1e-8)
    print_cmp("cmp.diff_count_atol8_rtol5", get_cmp_diff_count, atol=1e-8, rtol=1e-5)
    print_cmp("cmp.diff_count_atol5_rtol5", get_cmp_diff_count, atol=1e-5, rtol=1e-5)
    print_cmp("cmp.diff_count_atol3_rtol2", get_cmp_diff_count, atol=1e-3, rtol=1e-2)
    print_cmp("cmp.diff_count_atol2_rtol1", get_cmp_diff_count, atol=1e-2, rtol=1e-1)


def test_single_model(args):
    synchronizer_func = get_synchronizer_func(args)
    input_dict = get_input_dict(args)
    model = get_model(args)
    model.eval()

    # Run on eager mode
    running_eager_success = False
    try:
        print("Run model in eager mode.")
        expected_out, eager_time_stats = measure_performance(
            lambda: model(**input_dict), args, synchronizer_func
        )
        running_eager_success = True
    except Exception as e:
        print(f"Run model in eager mode failed: {str(e)}\n{traceback.format_exc()}")

    # Run on compiling mode
    running_compiled_success = False
    try:
        print("Run model in compiled mode.")
        compiled_model = get_compiled_model(args, model)
        compiled_out, compiled_time_stats = measure_performance(
            lambda: compiled_model(**input_dict), args, synchronizer_func
        )
        running_compiled_success = True
    except Exception as e:
        print(f"Run model in compiled mode failed: {str(e)}\n{traceback.format_exc()}")

    if running_eager_success and running_compiled_success:
        check_outputs(args, expected_out, compiled_out)

        test_compiler_utils.print_times_and_speedup(
            args, eager_time_stats, compiled_time_stats
        )


def get_cmp_equal(expected_out, compiled_out):
    return " ".join(
        str(int(paddle.equal_all(a, b))) for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_all_close(expected_out, compiled_out, atol, rtol):
    return " ".join(
        str(int(paddle.allclose(a, b, atol=atol, rtol=rtol)))
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_max_diff(expected_out, compiled_out):
    return " ".join(
        str(paddle.max(paddle.abs(a - b)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_mean_diff(expected_out, compiled_out):
    return " ".join(
        str(paddle.mean(paddle.abs(a - b)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_diff_count(expected_out, compiled_out, atol, rtol):
    return " ".join(
        str(paddle.sum(~paddle.isclose(a, b, atol=atol, rtol=rtol)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def test_multi_models(args):
    for model_path in path_utils.get_recursively_model_path(args.model_path):
        cmd = "".join(
            [
                sys.executable,
                "-m graph_net.paddle.test_compiler",
                f"--model-path {model_path}",
                f"--compiler {args.compiler}",
                f"--device {args.device}",
                f"--warmup {args.warmup}",
                f"--trials {args.trials}",
                f"--log-prompt {args.log_prompt}",
                f"--output-dir {args.output_dir}",
            ]
        )
        cmd_ret = os.system(cmd)
        assert cmd_ret == 0, f"{cmd_ret=}, {cmd=}"


def main(args):
    assert os.path.isdir(args.model_path)
    assert args.compiler == "cinn"

    random_seed = 123
    paddle.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    if path_utils.is_single_model_dir(args.model_path):
        test_single_model(args)
    else:
        test_multi_models(args)


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
        "--output-dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save the structured JSON result file.",
    )
    args = parser.parse_args()
    main(args=args)
