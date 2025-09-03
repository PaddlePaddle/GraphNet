from . import utils
import argparse
import importlib.util
import inspect
import torch
from pathlib import Path
from typing import Type, Any, List, Dict, Callable
import sys
import os
import os.path
from dataclasses import dataclass
from contextlib import contextmanager
import time
import json
import numpy as np
import platform
from graph_net.torch.backend.graph_compiler_backend import GraphCompilerBackend
from graph_net.torch.backend.tvm_backend import TvmBackend
from graph_net.torch.backend.inductor_backend import InductorBackend
from graph_net.torch.backend.tensorrt_backend import TensorRTBackend
from graph_net.torch.backend.blade_disc_backend import BladeDISCBackend

registry_backend = {
    "tvm": TvmBackend(),
    "inductor": InductorBackend(),
    "tensorrt": TensorRTBackend(),
    "bladedisc": BladeDISCBackend(),
}


def load_class_from_file(
    args: argparse.Namespace, class_name: str
) -> Type[torch.nn.Module]:
    file_path = f"{args.model_path}/model.py"
    file = Path(file_path).resolve()
    module_name = file.stem

    with open(file_path, "r", encoding="utf-8") as f:
        model_code = f.read()
    model_code = utils.modify_code_by_device(model_code, args.device)
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
    for tensor_meta in params.values():
        if hasattr(tensor_meta, "device"):
            tensor_meta.device = args.device
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


def get_timing_stats(elapsed_times: List[float]):
    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
    }
    return stats


def measure_performance(model_call, args, compiler):
    stats = {}

    # Warmup runs
    for _ in range(args.warmup):
        model_call()
    compiler.synchronize()

    if "cuda" in args.device:
        """
        Acknowledgement: We evaluate the performance on both end-to-end and GPU-only timings,
        With reference to methods only based on CUDA events from KernelBench in https://github.com/ScalingIntelligence/KernelBench
        """

        device = torch.device(args.device)
        hardware_name = torch.cuda.get_device_name(device)
        print(
            f"{args.log_prompt} [Profiling] Using device: {args.device} {hardware_name}, warm up {args.warmup}, trials {args.trials}"
        )

        e2e_times = []
        gpu_times = []

        for i in range(args.trials):
            # End-to-end timing (naive_timer)
            duration_box = DurationBox(-1)
            with naive_timer(duration_box, compiler.synchronize):
                # GPU-only timing (CUDA Events)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                model_call()

                end_event.record()
                torch.cuda.synchronize(device=device)

            gpu_time_ms = start_event.elapsed_time(end_event)
            e2e_times.append(duration_box.value)
            gpu_times.append(gpu_time_ms)
            print(
                f"Trial {i + 1}: e2e={duration_box.value:.2f} ms, gpu={gpu_time_ms:.3g} ms"
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
                model_call()
            print(f"Trial {i + 1}: e2e={duration_box.value:.2f} ms")
            e2e_times.append(duration_box.value)
        stats["e2e"] = get_timing_stats(e2e_times)

    return stats


def test_single_model(args):
    compiler = get_compiler_backend(args)
    input_dict = get_input_dict(args)
    model = get_model(args)
    compiled_model = compiler(model)

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

    if "cuda" in args.device:
        result_data["configuration"]["hardware"] = torch.cuda.get_device_name(
            args.device
        )
    elif args.device == "cpu":
        result_data["configuration"]["hardware"] = platform.processor()
    else:
        result_data["configuration"]["hardware"] = "unknown"

    if args.compiler == "inductor":
        result_data["configuration"]["compile_framework_version"] = torch.__version__
    elif args.compiler == "tvm":
        result_data["configuration"][
            "compile_framework_version"
        ] = f"Tvm {compiler.version}"
    elif args.compiler == "tensorrt":
        result_data["configuration"][
            "compile_framework_version"
        ] = f"TensorRT {compiler.version}"
    elif args.compiler == "bladedisc":
        result_data["configuration"][
            "compile_framework_version"
        ] = f"BladeDISC {compiler.version}"
    else:
        result_data["configuration"]["compiler_version"] = "unknown"

    eager_model_call = lambda: model(**input_dict)
    compiled_model_call = lambda: compiled_model(**input_dict)

    eager_stats = measure_performance(eager_model_call, args, compiler)
    compiled_stats = measure_performance(compiled_model_call, args, compiler)

    result_data["performance"]["eager"] = eager_stats
    result_data["performance"]["compiled"] = compiled_stats

    expected_out = eager_model_call()
    compiled_out = compiled_model_call()

    if not isinstance(expected_out, tuple):
        expected_out = (expected_out,)
    if not isinstance(compiled_out, tuple):
        compiled_out = (compiled_out,)

    def print_and_store_cmp(key, func, **kwargs):
        cmp_ret = func(expected_out, compiled_out, **kwargs)
        result_data["correctness"][key] = cmp_ret
        print(
            f"{args.log_prompt} {key} model_path:{args.model_path} {cmp_ret}",
            file=sys.stderr,
        )

    print_and_store_cmp("[equal]", get_cmp_equal)
    print_and_store_cmp(
        "[all_close_atol8_rtol8]", get_cmp_all_close, atol=1e-8, rtol=1e-8
    )
    print_and_store_cmp(
        "[all_close_atol8_rtol5]", get_cmp_all_close, atol=1e-8, rtol=1e-5
    )
    print_and_store_cmp(
        "[all_close_atol5_rtol5]", get_cmp_all_close, atol=1e-5, rtol=1e-5
    )
    print_and_store_cmp(
        "[all_close_atol3_rtol2]", get_cmp_all_close, atol=1e-3, rtol=1e-2
    )
    print_and_store_cmp(
        "[all_close_atol2_rtol1]", get_cmp_all_close, atol=1e-2, rtol=1e-1
    )
    print_and_store_cmp("[max_diff]", get_cmp_max_diff)
    print_and_store_cmp("[mean_diff]", get_cmp_mean_diff)
    print_and_store_cmp(
        "[diff_count_atol8_rtol8]", get_cmp_diff_count, atol=1e-8, rtol=1e-8
    )
    print_and_store_cmp(
        "[diff_count_atol8_rtol5]", get_cmp_diff_count, atol=1e-8, rtol=1e-5
    )
    print_and_store_cmp(
        "[diff_count_atol5_rtol5]", get_cmp_diff_count, atol=1e-5, rtol=1e-5
    )
    print_and_store_cmp(
        "[diff_count_atol3_rtol2]", get_cmp_diff_count, atol=1e-3, rtol=1e-2
    )
    print_and_store_cmp(
        "[diff_count_atol2_rtol1]", get_cmp_diff_count, atol=1e-2, rtol=1e-1
    )

    eager_e2e_time_ms = eager_stats.get("e2e", {}).get("mean", 0)
    compiled_e2e_time_ms = compiled_stats.get("e2e", {}).get("mean", 0)

    e2e_speedup = 0
    if eager_e2e_time_ms > 0 and compiled_e2e_time_ms > 0:
        e2e_speedup = eager_e2e_time_ms / compiled_e2e_time_ms
        result_data["performance"]["speedup"]["e2e"] = e2e_speedup

    duration_log = (
        f"{args.log_prompt} [Duration] "
        f"eager_e2e:{eager_e2e_time_ms:.4f} compiled_e2e:{compiled_e2e_time_ms:.4f}"
    )
    speedup_log = f"{args.log_prompt} [Speedup] " f"e2e_speedup:{e2e_speedup:.4f}"

    if "cuda" in args.device:
        eager_gpu_time_ms = eager_stats.get("gpu", {}).get("mean", 0)
        compiled_gpu_time_ms = compiled_stats.get("gpu", {}).get("mean", 0)

        gpu_speedup = 0
        if eager_gpu_time_ms > 0 and compiled_gpu_time_ms > 0:
            gpu_speedup = eager_gpu_time_ms / compiled_gpu_time_ms
            result_data["performance"]["speedup"]["gpu"] = gpu_speedup

        duration_log += f" eager_gpu:{eager_gpu_time_ms:.4f} compiled_gpu:{compiled_gpu_time_ms:.4f}"
        speedup_log += f" gpu_speedup:{gpu_speedup:.4f}"

    print(duration_log, file=sys.stderr)
    print(speedup_log, file=sys.stderr)

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
        # Transform to float to handle LongTensor output of some models, which cannnot be processed with torch.max().
        str(torch.max(torch.abs(a.float() - b.float())).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_mean_diff(expected_out, compiled_out):
    return " ".join(
        # To handle LongTensor
        str(torch.mean(torch.abs(a.float() - b.float())).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_diff_count(expected_out, compiled_out, atol, rtol):
    results = []
    for a, b in zip(expected_out, compiled_out):
        # To handle LongTensor
        if a.is_floating_point() and b.is_floating_point():
            diff_count = torch.sum(~torch.isclose(a, b, atol=atol, rtol=rtol)).item()
        else:
            diff_count = torch.sum(a != b).item()
        results.append(str(diff_count))
    return " ".join(results)


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
