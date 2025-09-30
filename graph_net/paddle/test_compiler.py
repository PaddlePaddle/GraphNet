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
import random
import platform
import traceback

from graph_net.paddle import utils
from graph_net import path_utils
from graph_net import test_compiler_util


def set_seed(random_seed):
    paddle.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def get_hardward_name(args):
    if args.device == "cuda":
        hardware = paddle.device.cuda.get_device_name(0)
    elif args.device == "cpu":
        hardware = platform.processor()
    else:
        hardware = "unknown"
    return hardware


def get_compile_framework_version(args):
    if args.compiler == "cinn":
        compile_framework_version = paddle.__version__
    else:
        compile_framework_version = "unknown"
    return compile_framework_version


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
    program = compiled_model.forward.concrete_program.main_program
    return compiled_model


def get_static_model(args, model):
    static_model = paddle.jit.to_static(
        model,
        input_spec=get_input_spec(args),
        full_graph=True,
        backend=None,
    )
    static_model.eval()
    program = static_model.forward.concrete_program.main_program
    return static_model


def measure_performance(model_call, args, synchronizer_func, profile=False):
    runtime_seed = 1024
    stats = {}

    paddle.seed(runtime_seed)
    outs = model_call()

    # Warmup runs
    for _ in range(args.warmup):
        model_call()
    synchronizer_func()

    hardware_name = get_hardward_name(args)
    print(
        f"[Profiling] Using device: {args.device} {hardware_name}, warm up {args.warmup}, trials {args.trials}",
        file=sys.stderr,
        flush=True,
    )

    if "cuda" in args.device:
        """
        Acknowledgement: We evaluate the performance on both end-to-end and GPU-only timings,
        With reference to methods only based on CUDA events from KernelBench in https://github.com/ScalingIntelligence/KernelBench
        """

        e2e_times = []
        gpu_times = []

        if profile:
            paddle.base.core.nvprof_start()
        for i in range(args.trials):
            # End-to-end timing (naive_timer)
            duration_box = test_compiler_util.DurationBox(-1)
            with test_compiler_util.naive_timer(duration_box, synchronizer_func):
                # GPU-only timing (CUDA Events)
                start_event = paddle.device.Event(enable_timing=True)
                end_event = paddle.device.Event(enable_timing=True)

                start_event.record()
                model_call()
                end_event.record()

            gpu_time_ms = start_event.elapsed_time(end_event)
            e2e_times.append(duration_box.value)
            gpu_times.append(gpu_time_ms)
            print(
                f"Trial {i + 1}: e2e={duration_box.value:.5f} ms, gpu={gpu_time_ms:.5f} ms",
                file=sys.stderr,
                flush=True,
            )
        if profile:
            paddle.base.core.nvprof_stop()

        stats["e2e"] = test_compiler_util.get_timing_stats(e2e_times)
        stats["gpu"] = test_compiler_util.get_timing_stats(gpu_times)
    else:  # CPU or other devices
        e2e_times = []
        for i in range(args.trials):
            duration_box = test_compiler_util.DurationBox(-1)
            with test_compiler_util.naive_timer(duration_box, synchronizer_func):
                model_call()
            print(f"Trial {i + 1}: e2e={duration_box.value:.4f} ms")
            e2e_times.append(duration_box.value)
        stats["e2e"] = test_compiler_util.get_timing_stats(e2e_times)

    return outs, stats


def check_outputs(args, expected_out, compiled_out):
    if isinstance(expected_out, paddle.Tensor):
        expected_out = [expected_out]
    if isinstance(compiled_out, paddle.Tensor):
        compiled_out = [compiled_out]

    eager_dtypes = [None] * len(expected_out)
    for i, tensor in enumerate(expected_out):
        eager_dtypes[i] = (
            str(tensor.dtype).replace("paddle.", "") if tensor is not None else "none"
        )

    compiled_dtypes = [None] * len(compiled_out)
    for i, tensor in enumerate(compiled_out):
        compiled_dtypes[i] = (
            str(tensor.dtype).replace("paddle.", "") if tensor is not None else "none"
        )

    type_match = test_compiler_util.check_output_datatype(
        args, eager_dtypes, compiled_dtypes
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

    if type_match:
        test_compiler_util.check_equal(
            args,
            expected_out,
            compiled_out,
            cmp_equal_func=get_cmp_equal,
        )

        expected_out_fp32 = transfer_to_float(expected_out)
        compiled_out_fp32 = transfer_to_float(compiled_out)
        test_compiler_util.check_allclose(
            args,
            expected_out_fp32,
            compiled_out_fp32,
            cmp_all_close_func=get_cmp_all_close,
            cmp_max_diff_func=get_cmp_max_diff,
            cmp_mean_diff_func=get_cmp_mean_diff,
            cmp_max_relative_diff_func=get_cmp_max_relative_diff,
            cmp_mean_relative_diff_func=get_cmp_mean_relative_diff,
        )


def test_single_model(args):
    synchronizer_func = get_synchronizer_func(args)
    input_dict = get_input_dict(args)
    model = get_model(args)
    model.eval()

    test_compiler_util.print_basic_config(
        args, get_hardward_name(args), get_compile_framework_version(args)
    )

    # Run on eager mode
    eager_success = False
    try:
        print("Run model in eager mode.")
        static_model = get_static_model(args, model)
        expected_out, eager_time_stats = measure_performance(
            lambda: static_model(**input_dict), args, synchronizer_func, profile=False
        )
        eager_success = True
    except Exception as e:
        print(f"Run model in eager mode failed: {str(e)}\n{traceback.format_exc()}")

    # Run on compiling mode
    compiled_success = False
    try:
        print("Run model in compiled mode.")
        compiled_model = get_compiled_model(args, model)
        compiled_out, compiled_time_stats = measure_performance(
            lambda: compiled_model(**input_dict), args, synchronizer_func, profile=False
        )
        compiled_success = True
    except Exception as e:
        print(f"Run model in compiled mode failed: {str(e)}\n{traceback.format_exc()}")

    test_compiler_util.print_running_status(args, eager_success, compiled_success)
    if eager_success and compiled_success:
        check_outputs(args, expected_out, compiled_out)

        test_compiler_util.print_times_and_speedup(
            args, eager_time_stats, compiled_time_stats
        )


def get_cmp_equal(expected_out, compiled_out):
    def convert(x):
        if x.dtype in [paddle.float16, paddle.bfloat16]:
            return x.astype("float32")
        elif x.dtype in [paddle.uint8, paddle.int8, paddle.int16]:
            return x.astype("int32")
        return x

    return " ".join(
        str(int(paddle.equal_all(convert(a), convert(b))))
        for a, b in zip(expected_out, compiled_out)
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


def get_cmp_max_relative_diff(expected_out, compiled_out):
    epsilon = 1e-8
    return " ".join(
        str(paddle.max(paddle.abs(a - b) / (paddle.abs(a) + epsilon)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_mean_relative_diff(expected_out, compiled_out):
    epsilon = 1e-8
    return " ".join(
        str(paddle.mean(paddle.abs(a - b) / (paddle.abs(a) + epsilon)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_diff_count(expected_out, compiled_out, atol, rtol):
    return " ".join(
        str(paddle.sum(~paddle.isclose(a, b, atol=atol, rtol=rtol)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def test_multi_models(args):
    verified_samples = None
    if args.verified_samples_list_path is not None:
        assert os.path.isfile(args.verified_samples_list_path)
        graphnet_root = path_utils.get_graphnet_root()
        print(f"graphnet_root: {graphnet_root}")
        verified_samples = []
        with open(args.verified_samples_list_path, "r") as f:
            for line in f.readlines():
                verified_samples.append(os.path.join(graphnet_root, line.strip()))

    sample_idx = 0
    failed_samples = []
    for model_path in path_utils.get_recursively_model_path(args.model_path):
        if verified_samples is None or os.path.abspath(model_path) in verified_samples:
            print(f"[{sample_idx}] test_compiler, model_path: {model_path}")
            cmd = " ".join(
                [
                    sys.executable,
                    "-m graph_net.paddle.test_compiler",
                    f"--model-path {model_path}",
                    f"--compiler {args.compiler}",
                    f"--device {args.device}",
                    f"--warmup {args.warmup}",
                    f"--trials {args.trials}",
                    f"--log-prompt {args.log_prompt}",
                ]
            )
            cmd_ret = os.system(cmd)
            # assert cmd_ret == 0, f"{cmd_ret=}, {cmd=}"
            if cmd_ret != 0:
                failed_samples.append(model_path)
            sample_idx += 1

    print(
        f"Totally {sample_idx} verified samples, failed {len(failed_samples)} samples."
    )
    for model_path in failed_samples:
        print(f"- {model_path}")


def main(args):
    assert os.path.isdir(args.model_path)
    assert args.compiler == "cinn"

    initalize_seed = 123
    set_seed(random_seed=initalize_seed)

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
        "--verified-samples-list-path",
        type=str,
        required=False,
        default=None,
        help="Path to model file(s), each subdirectory containing graph_net.json will be regarded as a model",
    )
    args = parser.parse_args()
    main(args=args)
