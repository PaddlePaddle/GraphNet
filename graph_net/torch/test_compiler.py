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
import random
import numpy as np
import platform
from graph_net.torch.backend.graph_compiler_backend import GraphCompilerBackend
from graph_net.torch.backend.tvm_backend import TvmBackend
from graph_net.torch.backend.xla_backend import XlaBackend
from graph_net.torch.backend.inductor_backend import InductorBackend
from graph_net.torch.backend.tensorrt_backend import TensorRTBackend
from graph_net.torch.backend.blade_disc_backend import BladeDISCBackend
from graph_net.torch.backend.nope_backend import NopeBackend
from graph_net.torch.backend.unstable_to_stable_backend import UnstableToStableBackend
from graph_net.torch.backend.range_decomposer_validator_backend import (
    RangeDecomposerValidatorBackend,
)
from graph_net.test_compiler_util import generate_allclose_configs
from graph_net import test_compiler_util
from graph_net import path_utils


registry_backend = {
    "tvm": TvmBackend(),
    "xla": XlaBackend(),
    "inductor": InductorBackend(),
    "tensorrt": TensorRTBackend(),
    "bladedisc": BladeDISCBackend(),
    "nope": NopeBackend(),
    "unstable_to_stable": UnstableToStableBackend(),
    "range_decomposer_validator": RangeDecomposerValidatorBackend(),
}


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)


def get_hardward_name(args):
    hardware_name = "unknown"
    if "cuda" in args.device:
        hardware_name = torch.cuda.get_device_name(args.device)
    elif args.device == "cpu":
        hardware_name = platform.processor()
    return hardware_name


def get_compile_framework_version(args):
    if args.compiler in ["inductor", "nope", "unstable_to_stable"]:
        return torch.__version__
    elif args.compiler in ["tvm", "xla", "tensorrt", "bladedisc"]:
        # Assuming compiler object has a version attribute
        return f"{args.compiler.capitalize()} {compiler.version}"
    return "unknown"


def load_class_from_file(
    args: argparse.Namespace, class_name: str, device: str
) -> Type[torch.nn.Module]:
    file_path = f"{args.model_path}/model.py"
    file = Path(file_path).resolve()
    module_name = file.stem

    with open(file_path, "r", encoding="utf-8") as f:
        model_code = f.read()
    model_code = utils.modify_code_by_device(model_code, device)
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    compiled_code = compile(model_code, filename=file, mode="exec")
    exec(compiled_code, module.__dict__)

    model_class = getattr(module, class_name, None)
    setattr(model_class, "__graph_net_file_path__", file_path)
    setattr(model_class, "__graph_net_device__", device)
    return model_class


def get_compiler_backend(args) -> GraphCompilerBackend:
    assert args.compiler in registry_backend, f"Unknown compiler: {args.compiler}"
    return registry_backend[args.compiler]


def get_model(args):
    device = "xla" if args.compiler == "xla" else args.device

    # device: Torch device object specifying the target device for model loading (e.g., 'cuda', 'cpu', 'xla')
    model_class = load_class_from_file(args, class_name="GraphModule", device=device)
    model = model_class().to(torch.device(args.device))
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


def measure_performance(model_call, args, compiler):
    stats = {}

    # Warmup runs
    for _ in range(args.warmup):
        model_call()
    compiler.synchronize()

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

        for i in range(args.trials):
            # End-to-end timing (naive_timer)
            duration_box = test_compiler_util.DurationBox(-1)
            with test_compiler_util.naive_timer(duration_box, compiler.synchronize):
                # GPU-only timing (CUDA Events)
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

    else:  # CPU or other devices
        e2e_times = []
        for i in range(args.trials):
            duration_box = test_compiler_util.DurationBox(-1)
            with test_compiler_util.naive_timer(duration_box, compiler.synchronize):
                model_call()
            print(
                f"Trial {i + 1}: e2e={duration_box.value:.5f} ms",
                file=sys.stderr,
                flush=True,
            )
            e2e_times.append(duration_box.value)
        stats["e2e"] = test_compiler_utilget_timing_stats(e2e_times)

    return stats


def test_single_model(args):
    compiler = get_compiler_backend(args)
    input_dict = get_input_dict(args)
    model = get_model(args)

    test_compiler_util.print_basic_config(
        args, get_hardward_name(args), get_compile_framework_version(args)
    )

    runtime_seed = 1024
    eager_failure = False
    expected_out = None
    eager_types = []
    eager_stats = {}

    try:
        eager_model_call = lambda: model(**input_dict)
        eager_stats = measure_performance(eager_model_call, args, compiler)

        torch.manual_seed(runtime_seed)
        expected_out = eager_model_call()
        if not isinstance(expected_out, tuple):
            expected_out = (expected_out,)
    except (TypeError, RuntimeError) as e:
        print(f"Eager model execution failed: {str(e)}", file=sys.stderr)
        eager_failure = True

    compiled_failure = False
    compiled_model = None
    compiled_types = []
    compiled_stats = {}

    try:
        compiled_model = compiler(model)
        torch.manual_seed(runtime_seed)
        compiled_model_call = lambda: compiled_model(**input_dict)
        compiled_stats = measure_performance(compiled_model_call, args, compiler)

        compiled_out = compiled_model_call()
        if not isinstance(compiled_out, tuple):
            compiled_out = (compiled_out,)
        if args.compiler == "xla":
            compiled_out = tuple(item.to("cpu").to("cuda") for item in compiled_out)
    except (TypeError, RuntimeError) as e:
        print(f"Compiled model execution failed: {str(e)}", file=sys.stderr)
        compiled_failure = True

    if eager_failure:
        print(f"{args.log_prompt} [Result] status: failed", file=sys.stderr, flush=True)
        print(
            f"{args.log_prompt} [Fail due to eager model execution error.]",
            file=sys.stderr,
            flush=True,
        )
    elif compiled_failure:
        print(f"{args.log_prompt} [Result] status: failed", file=sys.stderr, flush=True)
        print(
            f"{args.log_prompt} [Fail due to compiled model execution error.]",
            file=sys.stderr,
            flush=True,
        )
    else:
        compare_correctness(expected_out, compiled_out, args)

        print(
            f"{args.log_prompt} [Result] status: success", file=sys.stderr, flush=True
        )

        test_compiler_util.print_times_and_speedup(args, eager_stats, compiled_stats)


def print_and_store_cmp(key, cmp_func, args, expected_out, compiled_out, **kwargs):
    cmp_ret = cmp_func(expected_out, compiled_out, **kwargs)
    print(
        f"{args.log_prompt} [Correctness]{key}: {cmp_ret}",
        file=sys.stderr,
        flush=True,
    )
    return cmp_ret


def compare_correctness(expected_out, compiled_out, args):
    eager_dtypes = [
        (
            str(x.dtype).replace("torch.", "")
            if isinstance(x, torch.Tensor)
            else type(x).__name__
        )
        for x in expected_out
    ]
    compiled_dtypes = [
        (
            str(x.dtype).replace("torch.", "")
            if isinstance(x, torch.Tensor)
            else type(x).__name__
        )
        for x in compiled_out
    ]

    # datatype check
    type_match = test_compiler_util.check_output_datatype(
        args, eager_dtypes, compiled_dtypes
    )

    if type_match:
        test_compiler_util.check_equal(
            args,
            expected_out,
            compiled_out,
            cmp_equal_func=get_cmp_equal,
        )

        test_compiler_util.check_allclose(
            args,
            expected_out,
            compiled_out,
            cmp_all_close_func=get_cmp_all_close,
            cmp_max_diff_func=get_cmp_max_diff,
            cmp_mean_diff_func=get_cmp_mean_diff,
        )


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
                    f"-m graph_net.torch.{module_name}",
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
        "--allow-list",
        type=str,
        required=False,
        default=None,
        help="Path to samples list, each line contains a sample path",
    )
    args = parser.parse_args()
    main(args=args)
