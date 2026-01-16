from . import utils
import argparse
import importlib.util
import torch
from pathlib import Path
from typing import Type
import sys
import os
import traceback
import json
import random
import numpy as np
import platform
import base64
from contextlib import redirect_stdout, redirect_stderr

from graph_net_bench.torch.backend.graph_compiler_backend import GraphCompilerBackend
from graph_net_bench.torch.backend.tvm_backend import TvmBackend
from graph_net_bench.torch.backend.xla_backend import XlaBackend
from graph_net_bench.torch.backend.inductor_backend import InductorBackend
from graph_net_bench.torch.backend.tensorrt_backend import TensorRTBackend
from graph_net_bench.torch.backend.blade_disc_backend import BladeDISCBackend
from graph_net_bench.torch.backend.nope_backend import NopeBackend
from graph_net_bench.torch.backend.pass_mgr_backend import PassMgrBackend
from graph_net_bench.torch.backend.unstable_to_stable_backend import (
    UnstableToStableBackend,
)
from graph_net_bench.torch.backend.range_decomposer_validator_backend import (
    RangeDecomposerValidatorBackend,
)
from graph_net_bench.torch.backend.graph_variable_renamer_validator_backend import (
    GraphVariableRenamerValidatorBackend,
)
from graph_net_bench import test_compiler_util


compiler_backend_name2class = {
    "tvm": TvmBackend,
    "xla": XlaBackend,
    "inductor": InductorBackend,
    "tensorrt": TensorRTBackend,
    "bladedisc": BladeDISCBackend,
    "nope": NopeBackend,
    "pass_mgr": PassMgrBackend,
    "unstable_to_stable": UnstableToStableBackend,
    "range_decomposer_validator": RangeDecomposerValidatorBackend,
    "graph_variable_renamer_validator": GraphVariableRenamerValidatorBackend,
}


def register_op_lib(op_lib):
    if op_lib == "flaggems":
        import flag_gems

        flag_gems.enable()
    else:
        pass


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)


def get_hardward_name(device):
    hardware_name = "unknown"
    if "cuda" in device:
        hardware_name = torch.cuda.get_device_name(device)
    elif args.device == "cpu":
        hardware_name = platform.processor()
    return hardware_name


def get_compiler_version(compiler):
    if compiler in ["inductor", "nope", "unstable_to_stable"]:
        return torch.__version__
    elif compiler in ["tvm", "xla", "tensorrt", "bladedisc"]:
        # Assuming compiler object has a version attribute
        return f"{compiler.capitalize()} {compiler.version}"
    return "unknown"


def load_class_from_file(
    model_path: str, class_name: str, device: str
) -> Type[torch.nn.Module]:
    file_path = f"{model_path}/model.py"
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


def convert_to_dict(config_str):
    if config_str is None or config_str == "None":
        return {}
    config_str = base64.b64decode(config_str).decode("utf-8")
    config = json.loads(config_str)
    assert isinstance(config, dict), f"config should be a dict. {config_str=}"
    return config


def get_compiler_backend(args) -> GraphCompilerBackend:
    assert (
        args.compiler in compiler_backend_name2class
    ), f"Unknown compiler: {args.compiler}"
    backend_class = compiler_backend_name2class[args.compiler]
    backend_config = (
        convert_to_dict(args.backend_config) if args.backend_config is not None else {}
    )
    return backend_class(backend_config)


def get_model(args):
    device = "xla" if args.compiler == "xla" else args.device

    # device: Torch device object specifying the target device for model loading (e.g., 'cuda', 'cpu', 'xla')
    model_class = load_class_from_file(
        args.model_path, class_name="GraphModule", device=device
    )
    model = model_class().to(torch.device(args.device))
    return model


def get_input_dict(args):
    inputs_params = utils.load_converted_from_text(f"{args.model_path}")
    params = inputs_params["weight_info"]
    for tensor_meta in params.values():
        if "device" in tensor_meta["info"]:
            tensor_meta["info"]["device"] = args.device
    return {
        k: utils.replay_tensor(v).to(torch.device(args.device))
        for k, v in params.items()
    }


def measure_performance(model_call, args, compiler):
    stats = {}
    outs = model_call()

    # Warmup runs
    for _ in range(args.warmup):
        model_call()
    compiler.synchronize()

    hardware_name = get_hardward_name(args.device)
    print(
        f"[Profiling] Using device: {args.device} {hardware_name}, warm up {args.warmup}, trials {args.trials}",
        file=sys.stderr,
        flush=True,
    )

    if "cuda" in args.device:
        torch.cuda.empty_cache()
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
        stats["e2e"] = test_compiler_util.get_timing_stats(e2e_times)

    return outs, stats


def eval_single_model_with_single_backend(args):
    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    log_path = utils.get_log_path(args.output_path, args.model_path)
    output_dump_path = utils.get_output_path(args.output_path, args.model_path)
    print(f"Log path: {log_path}", file=sys.stderr, flush=True)
    print(f"Outputs path: {output_dump_path}", file=sys.stderr, flush=True)

    with open(log_path, "w", encoding="utf-8") as log_f:
        with redirect_stdout(log_f), redirect_stderr(log_f):
            compiler = get_compiler_backend(args)

            input_dict = get_input_dict(args)
            model = get_model(args)
            model.eval()

            test_compiler_util.print_with_log_prompt(
                "[Config] seed:", args.seed, args.log_prompt
            )

            test_compiler_util.print_basic_config(
                args,
                get_hardward_name(args.device),
                get_compiler_version(args.compiler),
            )

            test_compiler_util.print_with_log_prompt(
                "[Config] op_lib:", args.op_lib, args.log_prompt
            )

            success = False
            time_stats = {}
            try:
                compiled_model = compiler(model)

                def model_call():
                    return compiled_model(**input_dict)

                outputs, time_stats = measure_performance(model_call, args, compiler)
                success = True
            except Exception as e:
                print(
                    f"Run model failed: {str(e)}\n{traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )

            test_compiler_util.print_running_status(args, success)
            if success:
                torch.save(outputs, str(output_dump_path))
            test_compiler_util.print_with_log_prompt(
                "[Performance][eager]:", json.dumps(time_stats), args.log_prompt
            )

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
        print(content, file=sys.stderr, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single Backend Performance Evaluation"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default=None,
        help="Path to model file(s), each subdirectory containing graph_net.json will be regarded as a model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=False,
        default="/tmp/test_save",
        help="Path to save outputs",
    )
    parser.add_argument("--seed", type=int, required=False, default=123)
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
    parser.add_argument("--op-lib", type=str, required=False, default=None)
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
        default="graph-net-bench-log",
        help="Log prompt for performance log filtering.",
    )
    parser.add_argument(
        "--model-path-prefix",
        type=str,
        required=False,
        default=None,
        help="Prefix path to model path list",
    )
    parser.add_argument(
        "--backend-config",
        type=str,
        required=False,
        default=None,
        help="base64 encode configuration json.",
    )
    args = parser.parse_args()
    eval_single_model_with_single_backend(args=args)
