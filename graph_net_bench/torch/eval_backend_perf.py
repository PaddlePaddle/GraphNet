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
import types
from contextlib import redirect_stdout, redirect_stderr
from graph_net_bench.torch.backend.graph_compiler_backend import GraphCompilerBackend
from graph_net_bench import test_compiler_util


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
    elif device == "cpu":
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


def get_compiler_backend(args) -> GraphCompilerBackend:
    """
    Dynamically load backend class based on args.compiler
    """
    compiler_name = args.compiler.lower()
    module_name = f"graph_net_bench.torch.backend.{compiler_name}_backend"

    try:
        module = __import__(module_name, fromlist=[f"{compiler_name.title()}Backend"])

        class_name = (
            f"{''.join(part.title() for part in compiler_name.split('_'))}Backend"
        )

        backend_class = None
        if hasattr(module, class_name):
            backend_class = getattr(module, class_name)
        else:
            raise ImportError(f"No valid backend class found in {module_name}")

    except ImportError as e:
        raise ImportError(f"Failed to import backend module for '{compiler_name}': {e}")

    backend_config = (
        test_compiler_util.convert_to_dict(args.backend_config)
        if args.backend_config is not None
        else {}
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

    print(
        f"[Profiling] Warm up {args.warmup}, Trials {args.trials}",
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
    check_and_complete_args(args)
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

            test_compiler_util.print_config(
                args,
                get_hardward_name(args.device),
                get_compiler_version(args.compiler),
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


def check_and_complete_args(args):
    """
    Ensure all required arguments are present with default values if missing
    """
    defaults = {
        "model_path": None,  # Model path
        "output_path": None,  # Log and output directory
        "seed": 123,  # Random seed
        "compiler": "inductor",  # Compiler name
        "device": "cuda",  # Device for testing the compiler (e.g., 'cpu' or 'cuda')
        "op_lib": None,  # Operator library
        "warmup": 3,  # Number of warmup steps
        "trials": 5,  # Number of timing trials
        "log_prompt": "graph-net-bench-log",  # Log prompt for performance log filtering
        "model_path_prefix": None,  # Prefix path to model path in args.model-path
        "backend_config": None,  # backend configuration json
    }

    for key, default in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, default)


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
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=None,
        help="base64 encode configuration json.",
    )
    args = parser.parse_args()
    mut_args = types.SimpleNamespace(
        model_path=args.model_path,
        output_path=args.output_path,
        **test_compiler_util.convert_to_dict(args.config),
    )
    eval_single_model_with_single_backend(mut_args)
