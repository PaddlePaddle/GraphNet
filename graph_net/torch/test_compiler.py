from . import utils
import argparse
import importlib.util
import inspect
import torch
from pathlib import Path
from typing import Type, Any
import sys
from graph_net.torch.extractor import extract
import os
import os.path
from dataclasses import dataclass
from contextlib import contextmanager
import time
import json
import numpy as np
import platform

try:
    import torch_tensorrt
except ImportError:
    torch_tensorrt = None


class GraphCompilerBackend:
    def __call__(self, model):
        raise NotImplementedError()

    def synchronize(self):
        raise NotImplementedError()


class InductorBackend(GraphCompilerBackend):
    def __call__(self, model):
        return torch.compile(model, backend="inductor")

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class TensorRTBackend(GraphCompilerBackend):
    def __call__(self, model):
        return torch.compile(model, backend="tensorrt")

    def synchronize(self):
        torch.cuda.synchronize()


registry_backend = {
    "inductor": InductorBackend(),
    "tensorrt": TensorRTBackend(),
}


def load_class_from_file(
    args: argparse.Namespace, class_name: str
) -> Type[torch.nn.Module]:
    file_path = f"{args.model_path}/model.py"
    file = Path(file_path).resolve()
    module_name = file.stem

    with open(file_path, "r", encoding="utf-8") as f:
        model_code = f.read()
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    compiled_code = compile(model_code, filename=file, mode="exec")
    exec(compiled_code, module.__dict__)

    model_class = getattr(module, class_name, None)
    return model_class


def get_compiler_backend(args) -> GraphCompilerBackend:
    assert args.compiler in registry_backend, f"Unknown compiler: {args.compiler}"
    return registry_backend[args.compiler]


def get_model(args):
    model_class = load_class_from_file(args, class_name="GraphModule")
    model = model_class().to(torch.device(args.device))
    # for param in model.parameters():
    #     param.requires_grad_(False)
    return model


def get_input_dict(args):
    inputs_params = utils.load_converted_from_text(f"{args.model_path}")
    params = inputs_params["weight_info"]
    return {
        k: utils.replay_tensor(v).to(torch.device(args.device))
        for k, v in params.items()
    }


@dataclass
class DurationBox:
    value: float


@contextmanager
def naive_timer(duration_box, synchronizer_func):
    synchronizer_func()
    start = time.time()
    yield
    synchronizer_func()
    end = time.time()
    duration_box.value = (end - start) * 1000  # Store in milliseconds


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Acknowledgement: We introduce evaluation method in https://github.com/ScalingIntelligence/KernelBench to enhance function.

    Time a CUDA kernel function over multiple trials using torch.cuda.Event

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    # Actual trials
    for trial in range(num_trials):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device=device)

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def time_execution_naive(
    model_call, synchronizer_func, num_warmup: int = 3, num_trials: int = 10
):
    print(
        f"[Profiling] Using device: {args.device} {platform.processor()}, warm up {num_warmup}, trials {num_trials}"
    )
    for _ in range(num_warmup):
        model_call()

    times = []
    for i in range(num_trials):
        duration_box = DurationBox(-1)
        with naive_timer(duration_box, synchronizer_func):
            model_call()
        print(f"Trial {i + 1}: {duration_box.value:.2f} ms")
        times.append(duration_box.value)
    return times


def get_timing_stats(elapsed_times: list[float]):
    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
    }
    return stats


def measure_performance(model_call, args, compiler):
    if args.device == "cuda":
        times = time_execution_with_cuda_event(
            model_call,
            num_warmup=args.warmup,
            num_trials=args.trials,
            device=torch.device("cuda:0"),
        )
    else:
        times = time_execution_naive(
            model_call,
            compiler.synchronize,
            num_warmup=args.warmup,
            num_trials=args.trials,
        )
    return get_timing_stats(times)


def test_single_model(args):
    compiler = get_compiler_backend(args)
    input_dict = get_input_dict(args)
    model = get_model(args)
    compiled_model = compiler(model)

    eager_stats = {}
    compiled_stats = {}

    result_data = {
        "configuration": {
            "model": os.path.basename(os.path.normpath(args.model_path)),
            "device": args.device,
            "hardware": None,
            "compiler": args.compiler,
            "compile_framework_version": None,
            "warmup": args.warmup,
            "trials": args.trials,
        },
        "correctness": {},
        "performance": {
            "eager": {},
            "compiled": {},
            "speedup": {},
        },
    }

    if args.device == "cuda":
        result_data["configuration"]["hardware"] = torch.cuda.get_device_name(0)
    elif args.device == "cpu":
        result_data["configuration"]["hardware"] = platform.processor()
    else:
        result_data["configuration"]["hardware"] = "unknown"

    if args.compiler == "inductor":
        result_data["configuration"]["compile_framework_version"] = torch.__version__
    elif args.compiler == "tensorrt":
        result_data["configuration"][
            "compile_framework_version"
        ] = f"TensorRT {torch_tensorrt.version}"
    else:
        result_data["configuration"]["compiler_version"] = "unknown"

    eager_model_call = lambda: model(**input_dict)
    compiled_model_call = lambda: compiled_model(**input_dict)

    eager_stats = measure_performance(eager_model_call, args, compiler)
    compiled_stats = measure_performance(compiled_model_call, args, compiler)

    expected_out = eager_model_call()
    compiled_out = compiled_model_call()

    def print_and_store_cmp(key, func, **kwargs):
        cmp_ret = func(expected_out, compiled_out, **kwargs)
        result_data["correctness"][key] = cmp_ret
        print(
            f"{args.log_prompt} {key} model_path:{args.model_path} {cmp_ret}",
            file=sys.stderr,
        )

    print_and_store_cmp("equal", get_cmp_equal)
    print_and_store_cmp(
        "all_close_atol8_rtol8", get_cmp_all_close, atol=1e-8, rtol=1e-8
    )
    print_and_store_cmp(
        "all_close_atol8_rtol5", get_cmp_all_close, atol=1e-8, rtol=1e-5
    )
    print_and_store_cmp(
        "all_close_atol5_rtol5", get_cmp_all_close, atol=1e-5, rtol=1e-5
    )
    print_and_store_cmp(
        "all_close_atol3_rtol2", get_cmp_all_close, atol=1e-3, rtol=1e-2
    )
    print_and_store_cmp(
        "all_close_atol2_rtol1", get_cmp_all_close, atol=1e-2, rtol=1e-1
    )
    print_and_store_cmp("max_diff", get_cmp_max_diff)
    print_and_store_cmp("mean_diff", get_cmp_mean_diff)
    print_and_store_cmp(
        "diff_count_atol8_rtol8", get_cmp_diff_count, atol=1e-8, rtol=1e-8
    )
    print_and_store_cmp(
        "diff_count_atol8_rtol5", get_cmp_diff_count, atol=1e-8, rtol=1e-5
    )
    print_and_store_cmp(
        "diff_count_atol5_rtol5", get_cmp_diff_count, atol=1e-5, rtol=1e-5
    )
    print_and_store_cmp(
        "diff_count_atol3_rtol2", get_cmp_diff_count, atol=1e-3, rtol=1e-2
    )
    print_and_store_cmp(
        "diff_count_atol2_rtol1", get_cmp_diff_count, atol=1e-2, rtol=1e-1
    )

    eager_time_ms = eager_stats["mean"]
    compiled_time_ms = compiled_stats["mean"]

    result_data["performance"]["eager"] = eager_stats
    result_data["performance"]["compiled"] = compiled_stats
    if eager_time_ms > 0 and compiled_time_ms > 0:
        result_data["performance"]["speedup"] = eager_time_ms / compiled_time_ms

    print(
        f"{args.log_prompt} duration model_path:{args.model_path} eager:{eager_time_ms:.4f} compiled:{compiled_time_ms:.4f}",
        file=sys.stderr,
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_name = result_data["configuration"]["model"]
        compiler_name = args.compiler
        file_path = os.path.join(args.output_dir, f"{model_name}_{compiler_name}.json")
        with open(file_path, "w") as f:
            json.dump(result_data, f, indent=4)
        print(f"Result saved to {file_path}", file=sys.stderr)


def get_cmp_equal(expected_out, compiled_out):
    return " ".join(
        str(int(torch.equal(a, b))) for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_all_close(expected_out, compiled_out, atol, rtol):
    return " ".join(
        str(int(torch.allclose(a, b, atol=atol, rtol=rtol)))
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_max_diff(expected_out, compiled_out):
    return " ".join(
        str(torch.max(torch.abs(a - b)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_mean_diff(expected_out, compiled_out):
    return " ".join(
        str(torch.mean(torch.abs(a - b)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_diff_count(expected_out, compiled_out, atol, rtol):
    return " ".join(
        str(torch.sum(~torch.isclose(a, b, atol=atol, rtol=rtol)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def test_multi_models(args):
    for model_path in get_recursively_model_path(args.model_path):
        cmd_list = [
            sys.executable,
            "-m",
            "graph_net.torch.test_compiler",
            "--model-path",
            model_path,
            "--compiler",
            args.compiler,
            "--warmup",
            str(args.warmup),
            "--trials",
            str(args.trials),
            "--log-prompt",
            args.log_prompt,
            "--device",
            args.device,
        ]
        if args.output_dir:
            cmd_list.extend(["--output-dir", args.output_dir])

        cmd = " ".join(cmd_list)
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
        default="inductor",
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
        "--warmup", type=int, required=False, default=3, help="Number of warmup steps"
    )
    parser.add_argument(
        "--trials", type=int, required=False, default=5, help="Number of timing trials"
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
