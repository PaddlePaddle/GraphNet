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
    if isinstance(item, paddle.Tensor) and (item.dtype == paddle.bfloat16):
        item = np.array(item.astype("float32"))
    else:
        item = np.array(item)
    if item.dtype == np.bool_:
        item = item.astype("float32")
    return item


def count_number_of_ops(args, model):
    static_model = paddle.jit.to_static(
        model,
        input_spec=get_input_spec(args),
        full_graph=True,
        backend=None,
    )
    static_model.eval()
    program = model.forward.concrete_program.main_program
    # print(program)

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
    duration_box.value = end - start


def time_execution_with_cuda_event(
    model_call, synchronizer_func, num_warmup=3, num_trials=10, profile=False
):
    outs = None

    # warmups
    for _ in range(num_warmup):
        outs = model_call()
    synchronizer_func()

    elapsed_times = []
    if profile:
        paddle.base.core.nvprof_start()

    # actual trials
    for trial in range(num_trials):
        # create event marker default is not interprocess
        start_event = paddle.device.Event(enable_timing=True)
        end_event = paddle.device.Event(enable_timing=True)

        start_event.record()
        outs = model_call()
        end_event.record()
        synchronizer_func()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_times.append(elapsed_time_ms)
    if profile:
        paddle.base.core.nvprof_stop()
    elapsed_times = elapsed_times[num_trials // 2 :]
    return outs, np.mean(elapsed_times)


def time_execution_naive(model_call, synchronizer_func, num_warmup=3, num_trials=10):
    outs = None

    # warmups
    for _ in range(num_warmup):
        outs = model_call()

    # actual trials
    duration_box = DurationBox(-1)
    with naive_timer(duration_box, synchronizer_func):
        for i in range(num_trials):
            outs = model_call()
    return outs, duration_box.value * 1000 / float(num_trials)


def measure_performance(model_call, synchronizer_func, args, profile=False):
    if not args.use_naive_timer:
        outs, times = time_execution_with_cuda_event(
            model_call,
            synchronizer_func=synchronizer_func,
            num_warmup=args.warmup,
            num_trials=args.trials,
            profile=profile,
        )
    else:
        outs, times = time_execution_naive(
            model_call,
            synchronizer_func=synchronizer_func,
            num_warmup=args.warmup,
            num_trials=args.trials,
        )
    return outs, times


def init_benchmark_result(args):
    if args.device == "cuda":
        hardware = paddle.device.cuda.get_device_name(0)
    elif args.device == "cpu":
        hardware = platform.processor()
    else:
        hardware = "unknown"

    if args.compiler == "CINN":
        compile_framework_version = paddle.__version__
    else:
        compile_framework_version = "unknown"

    result_data = BenchmarkResult(
        args=args,
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
    num_ops = count_number_of_ops(args, model)

    # Initialize benchmark result
    result_data = init_benchmark_result(args)
    result_data.update_model_info(num_ops, input_dtypes, param_dtypes)

    # Run on eager mode
    expected_out, eager_time_ms = measure_performance(
        lambda: model(**input_dict), synchronizer_func, args, profile=False
    )

    # Run on compiling mode
    compiled_model = get_compiled_model(args, model)
    compiled_out, compiled_time_ms = measure_performance(
        lambda: compiled_model(**input_dict), synchronizer_func, args, profile=False
    )

    if isinstance(expected_out, paddle.Tensor):
        expected_out = [expected_out]
        compiled_out = [compiled_out]
    if isinstance(expected_out, list) or isinstance(expected_out, tuple):
        for a, b in zip(expected_out, compiled_out):
            if (a is None and b is not None) or (a is not None and b is None):
                raise ValueError("Both expected_out and compiled_out must be not None.")
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
        f"{args.log_prompt} information model_path:{args.model_path} {num_ops} ops, param_dtypes:{param_dtypes}, input_dtypes:{input_dtypes}",
        file=sys.stderr,
    )
    print(
        f"{args.log_prompt} duration model_path:{args.model_path} eager:{eager_time_ms:.5f} ms, compiled:{compiled_time_ms:.5f} ms, speedup:{eager_time_ms / compiled_time_ms:.3f}",
        file=sys.stderr,
    )

    result_data.update_performance(eager_time_ms, compiled_time_ms)
    if args.output_dir:
        result_data.write_to_json(args.output_dir)


def get_cmp_equal(expected_out, compiled_out):
    return " ".join(
        str(int(np.sum(np.equal(a, b)))) for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_all_close(expected_out, compiled_out, atol, rtol):
    return " ".join(
        str(int(np.allclose(a, b, atol=atol, rtol=rtol)))
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_max_diff(expected_out, compiled_out):
    return " ".join(
        str(np.max(np.abs(a - b)).item()) for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_mean_diff(expected_out, compiled_out):
    return " ".join(
        str(np.mean(np.abs(a - b)).item()) for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_diff_count(expected_out, compiled_out, atol, rtol):
    return " ".join(
        str(np.sum(~np.isclose(a, b, atol=atol, rtol=rtol)).item())
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
                f"--warmup {args.warmup}",
                f"--trials {args.trials}",
                f"--log-prompt {args.log_prompt}",
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
    assert args.compiler == "CINN"

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
        default="CINN",
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
        "--use-naive-timer",
        action="store_true",
        default=False,
        help="Use naive timer for permance measuring.",
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
