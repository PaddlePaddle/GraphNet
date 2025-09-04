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

from graph_net.paddle import utils
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

    param_dtypes = set()
    for name, info in params.items():
        dtype = str(info["info"]["dtype"])
        if dtype not in param_dtypes:
            param_dtypes.add(dtype)

    input_dtypes = set()
    for name, info in inputs.items():
        dtype = str(info["info"]["dtype"])
        if dtype not in input_dtypes:
            input_dtypes.add(dtype)

    params.update(inputs)
    state_dict = {k: utils.replay_tensor(v) for k, v in params.items()}
    return state_dict, list(input_dtypes), list(param_dtypes)


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


def regular_item(item):
    assert isinstance(item, paddle.Tensor)
    if item.dtype not in [paddle.float32, paddle.float64]:
        item = item.astype("float32")
    return item


def count_number_of_ops(args, model, eager_mode):
    if eager_mode:
        static_model = paddle.jit.to_static(
            model,
            input_spec=get_input_spec(args),
            full_graph=True,
            backend=None,
        )
        static_model.eval()
        program = static_model.forward.concrete_program.main_program
    else:
        program = model.forward.concrete_program.main_program
        print(program)

    num_ops = 0
    for block in program.blocks:
        for op in block.ops:
            if op.name() != "pd_op.data" and not op.name().startswith("builtin."):
                num_ops += 1
    print(f"Totally {num_ops} ops.")
    print("")
    return num_ops


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
            duration_box = DurationBox(-1)
            with naive_timer(duration_box, synchronizer_func):
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
                f"Trial {i + 1}: e2e={duration_box.value:.4f} ms, gpu={gpu_time_ms:.5g} ms"
            )

        stats["e2e"] = get_timing_stats(e2e_times)
        stats["gpu"] = get_timing_stats(gpu_times)
    else:  # CPU or other devices
        hardware_name = platform.processor()
        print(
            f"[Profiling] Using device: {args.device} {hardware_name}, warm up {args.warmup}, trials {args.trials}"
        )

        e2e_times = []
        for i in range(args.trials):
            duration_box = DurationBox(-1)
            with naive_timer(duration_box, compiler.synchronize):
                outs = model_call()
            print(f"Trial {i + 1}: e2e={duration_box.value:.4f} ms")
            e2e_times.append(duration_box.value)
        stats["e2e"] = get_timing_stats(e2e_times)

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


def test_single_model(args):
    synchronizer_func = get_synchronizer_func(args)
    input_dict, input_dtypes, param_dtypes = get_input_dict(args)
    model = get_model(args)
    model.eval()

    # Collect model information
    num_eager_ops = count_number_of_ops(args, model, eager_mode=True)

    # Initialize benchmark result
    result_data = init_benchmark_result(args)
    result_data.update_model_info(num_eager_ops, input_dtypes, param_dtypes)

    # Run on eager mode
    expected_out, eager_time_stats = measure_performance(
        lambda: model(**input_dict), args, synchronizer_func
    )

    # Run on compiling mode
    compiled_model = get_compiled_model(args, model)
    compiled_out, compiled_time_stats = measure_performance(
        lambda: compiled_model(**input_dict), args, synchronizer_func
    )

    if isinstance(expected_out, paddle.Tensor):
        expected_out = [expected_out]
        compiled_out = [compiled_out]
    if isinstance(expected_out, list) or isinstance(expected_out, tuple):
        output_dtypes = []
        for a, b in zip(expected_out, compiled_out):
            if (a is None and b is not None) or (a is not None and b is None):
                raise ValueError("Both expected_out and compiled_out must be not None.")
            if a is not None and b is not None:
                assert (
                    a.dtype == b.dtype
                ), f"expected_out's dtype ({a.dtype}) is not the same as compiled_out's dtype {b.dtype}."
                output_dtypes.append(str(a.dtype))
        result_data.update_corrrectness("num_outpus", len(output_dtypes))
        result_data.update_corrrectness("output_dtyps", output_dtypes)

        # Remove all None in outputs
        expected_out = [x for x in expected_out if x is not None]
        compiled_out = [x for x in compiled_out if x is not None]
        expected_out = [
            regular_item(item)
            for item in expected_out
            if item is not None and np.array(item).size != 0
        ]
        compiled_out = [
            regular_item(item)
            for item in compiled_out
            if item is not None and np.array(item).size != 0
        ]
    else:
        raise ValueError("Illegal return value.")

    def print_cmp(key, func, **kwargs):
        cmp_ret = func(expected_out, compiled_out, **kwargs)
        result_data.update_corrrectness(key, cmp_ret)
        print(
            f"{args.log_prompt} {key} model_path:{args.model_path} {cmp_ret}",
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

    print(
        f"{args.log_prompt} information model_path:{args.model_path} {num_eager_ops} ops, param_dtypes:{param_dtypes}, input_dtypes:{input_dtypes}",
        file=sys.stderr,
    )

    result_data.update_performance(eager_time_stats, compiled_time_stats)
    duration_log = (
        f"{args.log_prompt} [Duration] "
        f"eager_e2e:{result_data.eager_e2e_time_ms:.4f} ms compiled_e2e:{result_data.compiled_e2e_time_ms:.4f} ms"
    )
    speedup_log = (
        f"{args.log_prompt} [Speedup] " f"e2e_speedup:{result_data.e2e_speedup:.4f}"
    )

    if "cuda" in args.device:
        duration_log += f" eager_gpu:{result_data.eager_gpu_time_ms:.4f} ms compiled_gpu:{result_data.compiled_gpu_time_ms:.4f} ms"
        speedup_log += f" gpu_speedup:{result_data.gpu_speedup:.4f}"

    print(duration_log, file=sys.stderr)
    print(speedup_log, file=sys.stderr)

    if args.output_dir:
        result_data.write_to_json(args.output_dir)


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
    for model_path in get_recursively_model_path(args.model_path):
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


def get_recursively_model_path(root_dir):
    for sub_dir in get_immediate_subdirectory_paths(root_dir):
        if is_single_model_dir(sub_dir):
            yield sub_dir
        else:
            yield from get_recursively_model_path(sub_dir)


def get_immediate_subdirectory_paths(parent_dir):
    return [
        sub_dir
        for name in os.listdir(parent_dir)
        for sub_dir in [os.path.join(parent_dir, name)]
        if os.path.isdir(sub_dir)
    ]


def is_single_model_dir(model_dir):
    return os.path.isfile(f"{model_dir}/graph_net.json")


def main(args):
    assert os.path.isdir(args.model_path)
    assert args.compiler == "cinn"

    random_seed = 123
    paddle.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    if is_single_model_dir(args.model_path):
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
