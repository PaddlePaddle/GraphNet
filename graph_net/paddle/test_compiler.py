import argparse
import importlib.util
import paddle
from pathlib import Path
import sys
import os
import numpy as np
import random
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


def set_seed(random_seed):
    paddle.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def init_env(args):
    if test_compiler_util.is_gpu_device(args.device):
        paddle.set_flags({"FLAGS_cudnn_exhaustive_search": 1})


def get_hardward_name(args):
    hardware = "unknown"
    if test_compiler_util.is_gpu_device(args.device):
        hardware = paddle.device.cuda.get_device_name(0)
    elif args.device == "xpu":
        try:
            output = subprocess.check_output(["xpu-smi", "-L"], text=True)
            hardware = next(
                match.group(2)
                for line in output.splitlines()
                if (
                    match := re.match(
                        r"XPU\s+(\d+):\s+(.+?)\s+\(UUID:\s*([^)]+)\)", line
                    )
                )
            )
        except Exception:
            pass
    elif args.device == "cpu":
        hardware = platform.processor()
    return hardware


def get_compile_framework_version(args):
    if args.compiler in ["cinn", "nope"]:
        return paddle.__version__
    return "unknown"


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


def get_input_dict(model_path):
    inputs_params = utils.load_converted_from_text(f"{model_path}")
    params = inputs_params["weight_info"]
    inputs = inputs_params["input_info"]

    params.update(inputs)
    state_dict = {k: utils.replay_tensor(v) for k, v in params.items()}
    return state_dict


def get_input_spec(model_path):
    inputs_params_list = utils.load_converted_list_from_text(f"{model_path}")
    input_spec = [None] * len(inputs_params_list)
    for i, v in enumerate(inputs_params_list):
        dtype = v["info"]["dtype"]
        shape = v["info"]["shape"]
        input_spec[i] = paddle.static.InputSpec(shape, dtype)
    return input_spec


def get_static_model(args, model):
    static_model = paddle.jit.to_static(
        model,
        input_spec=get_input_spec(args.model_path),
        full_graph=True,
        backend=None,
    )
    static_model.eval()
    program = static_model.forward.concrete_program.main_program  # noqa
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

    hardware_name = get_hardward_name(args)
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
    def _flatten_outputs_to_list(outs):
        flattened_outs = outs
        if isinstance(outs, paddle.Tensor):
            flattened_outs = [outs]
        else:
            flattened_outs = [
                x
                for out in outs
                for x in (out if isinstance(out, (tuple, list)) else (out,))
            ]
        return flattened_outs

    expected_out = _flatten_outputs_to_list(expected_out)
    compiled_out = _flatten_outputs_to_list(compiled_out)

    def _get_output_dtypes(outs):
        dtypes = [
            str(tensor.dtype).replace("paddle.", "")
            if isinstance(tensor, paddle.Tensor)
            else None
            for i, tensor in enumerate(outs)
        ]
        return dtypes

    eager_dtypes = _get_output_dtypes(expected_out)
    compiled_dtypes = _get_output_dtypes(compiled_out)
    type_match = test_compiler_util.check_output_datatype(
        args, eager_dtypes, compiled_dtypes
    )

    def _get_output_shapes(outs):
        shapes = [
            tensor.shape if isinstance(tensor, paddle.Tensor) else None
            for i, tensor in enumerate(outs)
        ]
        return shapes

    eager_shapes = _get_output_shapes(expected_out)
    compiled_shapes = _get_output_shapes(compiled_out)
    shape_match = test_compiler_util.check_output_shape(
        args, eager_shapes, compiled_shapes
    )

    def _transfer_to_float(origin_outputs):
        outputs = []
        for item in origin_outputs:
            if isinstance(item, paddle.Tensor) and item.dtype not in [
                paddle.float32,
                paddle.float64,
            ]:
                item = item.astype("float32")
            outputs.append(item)
        return outputs

    if type_match and shape_match:
        test_compiler_util.check_equal(
            args,
            expected_out,
            compiled_out,
            cmp_equal_func=get_cmp_equal,
        )

        expected_out_fp32 = _transfer_to_float(expected_out)
        compiled_out_fp32 = _transfer_to_float(compiled_out)
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


def test_single_model(args):
    model_path = os.path.normpath(args.model_path)
    test_compiler_util.print_with_log_prompt(
        "[Processing]", model_path, args.log_prompt
    )

    compiler = get_compiler_backend(args)
    check_and_print_gpu_utilization(compiler)

    input_dict = get_input_dict(model_path)
    model = get_model(model_path)
    model.eval()

    test_compiler_util.print_basic_config(
        args, get_hardward_name(args), get_compile_framework_version(args)
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
        input_spec = get_input_spec(model_path)
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


def get_format_str(f):
    if (abs(f) > 1e5 or abs(f) < 1e-5) and abs(f) != 0.0:
        return str(f"{f:.5E}")
    else:
        return str(f"{f:.5f}")


def get_cmp_max_diff(expected_out, compiled_out):
    return " ".join(
        get_format_str(paddle.max(paddle.abs(a - b)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_mean_diff(expected_out, compiled_out):
    return " ".join(
        get_format_str(paddle.mean(paddle.abs(a - b)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_max_relative_diff(expected_out, compiled_out):
    epsilon = 1e-8
    return " ".join(
        get_format_str(paddle.max(paddle.abs(a - b) / (paddle.abs(a) + epsilon)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_mean_relative_diff(expected_out, compiled_out):
    epsilon = 1e-8
    return " ".join(
        get_format_str(
            paddle.mean(paddle.abs(a - b) / (paddle.abs(a) + epsilon)).item()
        )
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_diff_count(expected_out, compiled_out, atol, rtol):
    return " ".join(
        str(paddle.sum(~paddle.isclose(a, b, atol=atol, rtol=rtol)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def test_multi_models(args):
    test_samples = test_compiler_util.get_allow_samples(args.allow_list)

    sample_idx = 0
    failed_samples = []
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    for model_path in path_utils.get_recursively_model_path(args.model_path):
        if test_samples is None or os.path.abspath(model_path) in test_samples:
            print(
                f"[{sample_idx}] {module_name}, model_path: {model_path}",
                file=sys.stderr,
                flush=True,
            )
            cmd = " ".join(
                [
                    sys.executable,
                    f"-m graph_net.paddle.{module_name}",
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
        f"Totally {sample_idx} verified samples, failed {len(failed_samples)} samples.",
        file=sys.stderr,
        flush=True,
    )
    for model_path in failed_samples:
        print(f"- {model_path}", file=sys.stderr, flush=True)


def main(args):
    assert os.path.isdir(args.model_path)
    assert args.compiler in {"cinn", "nope"}
    assert args.device in ["cuda", "dcu", "xpu", "cpu"]

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
        "--allow-list",
        type=str,
        required=False,
        default=None,
        help="Path to samples list, each line contains a sample path",
    )
    args = parser.parse_args()
    main(args=args)
