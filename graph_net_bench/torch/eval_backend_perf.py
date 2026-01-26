"""Single Backend Performance Evaluation Script."""

import argparse
import importlib.util
import json
import os
import platform
import random
import sys
import traceback
import types
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple, Type, Optional

import numpy as np
import torch

from . import utils
from graph_net_bench import test_compiler_util
from graph_net_bench.torch.backend.graph_compiler_backend import GraphCompilerBackend

_ARG_DEFAULTS: Dict[str, Any] = {
    "model_path": None,
    "output_path": None,
    "seed": 123,
    "compiler": "inductor",
    "device": "cuda",
    "op_lib": None,
    "warmup": 3,
    "trials": 5,
    "log_prompt": "graph-net-bench-log",
    "model_path_prefix": None,
    "backend_config": None,
}


def register_op_lib(op_lib: Optional[str]) -> None:
    """Register operator library if specified."""
    if op_lib != "flaggems":
        return
    import flag_gems

    flag_gems.enable()


def set_seed(random_seed: int) -> None:
    """Set random seed for reproducibility across all frameworks."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if not torch.cuda.is_available():
        return
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def get_hardware_name(device: str) -> str:
    """Get hardware name based on device type."""
    if "cuda" in device:
        return torch.cuda.get_device_name(device)
    if device == "cpu":
        return platform.processor()
    return "unknown"


def get_compiler_version(compiler_name: str) -> str:
    """Get version string for the given compiler.

    Args:
        compiler_name: Name of the compiler (e.g., 'inductor', 'tvm').

    Returns:
        Version string or 'unknown' if not determinable.
    """
    torch_based_compilers = {"inductor", "nope", "unstable_to_stable"}
    if compiler_name in torch_based_compilers:
        return torch.__version__
    # TODO: For external compilers, version detection would require runtime introspection
    # which is not reliably available here. Return a placeholder.
    return "unknown"


def _read_and_modify_model_code(file_path: str, device: str) -> str:
    """Read model file and modify code for target device."""
    with open(file_path, "r", encoding="utf-8") as f:
        model_code = f.read()
    return utils.modify_code_by_device(model_code, device)


def _create_module_from_code(
    module_name: str, code: str, file_path: Path
) -> types.ModuleType:
    """Create a module by executing code."""
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    compiled_code = compile(code, filename=file_path, mode="exec")
    exec(compiled_code, module.__dict__)
    return module


def load_class_from_file(
    model_path: str, class_name: str, device: str
) -> Type[torch.nn.Module]:
    """Dynamically load a model class from file.

    Args:
        model_path: Directory containing model.py.
        class_name: Name of the class to load.
        device: Target device for code modification.

    Returns:
        The loaded model class with metadata attributes set.

    Raises:
        AttributeError: If class_name not found in module.
    """
    file_path = f"{model_path}/model.py"
    resolved_path = Path(file_path).resolve()
    module_name = resolved_path.stem

    model_code = _read_and_modify_model_code(file_path, device)
    module = _create_module_from_code(module_name, model_code, resolved_path)

    model_class = getattr(module, class_name)
    model_class.__graph_net_file_path__ = file_path
    model_class.__graph_net_device__ = device
    return model_class


def _build_backend_class_name(compiler_name: str) -> str:
    """Convert compiler name to PascalCase backend class name."""
    return "".join(part.title() for part in compiler_name.split("_")) + "Backend"


def _load_backend_class(compiler_name: str) -> Type[GraphCompilerBackend]:
    """Load backend class by compiler name."""
    module_name = f"graph_net_bench.torch.backend.{compiler_name}_backend"
    class_name = _build_backend_class_name(compiler_name)

    module = __import__(module_name, fromlist=[class_name])
    if not hasattr(module, class_name):
        raise ImportError(
            f"No valid backend class '{class_name}' found in {module_name}"
        )
    return getattr(module, class_name)


def get_compiler_backend(args) -> GraphCompilerBackend:
    """Dynamically load and instantiate backend class based on args.compiler."""
    backend_class = _load_backend_class(args.compiler.lower())
    backend_config = test_compiler_util.convert_to_dict(args.backend_config) or {}
    return backend_class(backend_config)


def get_model(args) -> torch.nn.Module:
    """Load and prepare model for evaluation."""
    load_device = "xla" if args.compiler == "xla" else args.device
    model_class = load_class_from_file(args.model_path, "GraphModule", load_device)
    return model_class().to(torch.device(args.device))


def _update_tensor_device(params: Dict[str, Any], device: str) -> None:
    """Update device info in tensor metadata in-place."""
    for tensor_meta in params.values():
        if "device" in tensor_meta["info"]:
            tensor_meta["info"]["device"] = device


def get_input_dict(args) -> Dict[str, torch.Tensor]:
    """Load and prepare input tensors for model evaluation.

    Args:
        args: Arguments containing model_path and device settings.

    Returns:
        Dictionary mapping parameter names to tensors on target device.
    """
    inputs_params = utils.load_converted_from_text(args.model_path)
    params = inputs_params["weight_info"]
    _update_tensor_device(params, args.device)

    target_device = torch.device(args.device)
    return {k: utils.replay_tensor(v).to(target_device) for k, v in params.items()}


def _run_warmup(model_call: Callable, warmup_count: int, sync_fn: Callable) -> None:
    """Execute warmup runs."""
    for _ in range(warmup_count):
        model_call()
    sync_fn()


def _measure_single_trial_cuda(
    model_call: Callable, sync_fn: Callable
) -> Tuple[float, float]:
    """Measure a single trial on CUDA device.

    Returns:
        Tuple of (e2e_time_ms, gpu_time_ms).
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    duration_box = test_compiler_util.DurationBox(-1)

    with test_compiler_util.naive_timer(duration_box, sync_fn):
        start_event.record()
        model_call()
        end_event.record()
        sync_fn()

    gpu_time_ms = start_event.elapsed_time(end_event)
    return duration_box.value, gpu_time_ms


def _measure_single_trial_cpu(model_call: Callable, sync_fn: Callable) -> float:
    """Measure a single trial on CPU or other devices.

    Returns:
        End-to-end time in milliseconds.
    """
    duration_box = test_compiler_util.DurationBox(-1)
    with test_compiler_util.naive_timer(duration_box, sync_fn):
        model_call()
    return duration_box.value


def _run_cuda_trials(
    model_call: Callable, trials: int, sync_fn: Callable
) -> Dict[str, Any]:
    """Run multiple timing trials on CUDA device."""
    torch.cuda.empty_cache()
    e2e_times: List[float] = []
    gpu_times: List[float] = []

    for i in range(trials):
        e2e_time, gpu_time = _measure_single_trial_cuda(model_call, sync_fn)
        e2e_times.append(e2e_time)
        gpu_times.append(gpu_time)
        print(
            f"Trial {i + 1}: e2e={e2e_time:.5f} ms, gpu={gpu_time:.5f} ms",
            file=sys.stderr,
            flush=True,
        )

    return {
        "e2e": test_compiler_util.get_timing_stats(e2e_times),
        "gpu": test_compiler_util.get_timing_stats(gpu_times),
    }


def _run_cpu_trials(
    model_call: Callable, trials: int, sync_fn: Callable
) -> Dict[str, Any]:
    """Run multiple timing trials on CPU or other devices."""
    e2e_times: List[float] = []

    for i in range(trials):
        e2e_time = _measure_single_trial_cpu(model_call, sync_fn)
        e2e_times.append(e2e_time)
        print(
            f"Trial {i + 1}: e2e={e2e_time:.5f} ms",
            file=sys.stderr,
            flush=True,
        )

    return {"e2e": test_compiler_util.get_timing_stats(e2e_times)}


def measure_performance(
    model_call: Callable, args, compiler
) -> Tuple[Any, Dict[str, Any]]:
    """Measure model inference performance.

    Args:
        model_call: Callable that executes the model.
        args: Arguments containing device, warmup, and trials settings.
        compiler: Compiler backend with synchronize method.

    Returns:
        Tuple of (model_outputs, timing_stats).
    """
    outs = model_call()
    _run_warmup(model_call, args.warmup, compiler.synchronize)

    print(
        f"[Profiling] Warm up {args.warmup}, Trials {args.trials}",
        file=sys.stderr,
        flush=True,
    )

    is_cuda = "cuda" in args.device
    if is_cuda:
        stats = _run_cuda_trials(model_call, args.trials, compiler.synchronize)
    else:
        stats = _run_cpu_trials(model_call, args.trials, compiler.synchronize)

    return outs, stats


def _compile_and_benchmark(
    args, compiler: GraphCompilerBackend, model: torch.nn.Module, input_dict: Dict
) -> Tuple[bool, Any, Dict[str, Any]]:
    """Compile model and run performance benchmark.

    Returns:
        Tuple of (success, outputs, time_stats).
    """
    try:
        compiled_model = compiler(model)

        def model_call():
            return compiled_model(**input_dict)

        outputs, time_stats = measure_performance(model_call, args, compiler)
        return True, outputs, time_stats
    except Exception as e:
        print(
            f"Run model failed: {str(e)}\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        return False, None, {}


def _run_evaluation_core(args) -> Tuple[bool, Any, Dict[str, Any]]:
    """Core evaluation logic: load model, compile, and benchmark."""
    compiler = get_compiler_backend(args)
    input_dict = get_input_dict(args)
    model = get_model(args)
    model.eval()

    test_compiler_util.print_config(
        args,
        get_hardware_name(args.device),
        get_compiler_version(args.compiler),
    )

    return _compile_and_benchmark(args, compiler, model, input_dict)


def _finalize_evaluation(
    args,
    success: bool,
    outputs: Any,
    time_stats: Dict[str, Any],
    output_dump_path: Path,
) -> None:
    """Finalize evaluation: save outputs and print status."""
    test_compiler_util.print_running_status(args, success)
    if success:
        torch.save(outputs, str(output_dump_path))
    test_compiler_util.print_with_log_prompt(
        "[Performance][eager]:", json.dumps(time_stats), args.log_prompt
    )


def _print_log_file(log_path: Path) -> None:
    """Read and print log file content to stderr."""
    print(Path(log_path).read_text(encoding="utf-8"), file=sys.stderr, flush=True)


def eval_single_model_with_single_backend(args) -> None:
    """Evaluate a single model with a single compiler backend."""
    check_and_complete_args(args)
    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    log_path = utils.get_log_path(args.output_path, args.model_path)
    output_dump_path = utils.get_output_path(args.output_path, args.model_path)
    print(f"Log path: {log_path}", file=sys.stderr, flush=True)
    print(f"Outputs path: {output_dump_path}", file=sys.stderr, flush=True)

    with open(log_path, "w", encoding="utf-8") as log_f:
        with redirect_stdout(log_f), redirect_stderr(log_f):
            success, outputs, time_stats = _run_evaluation_core(args)
            _finalize_evaluation(args, success, outputs, time_stats, output_dump_path)

    _print_log_file(log_path)


def check_and_complete_args(args) -> None:
    """Ensure all required arguments are present with default values if missing.

    Args:
        args: Namespace object to be validated and completed in-place.
    """
    for key, default in _ARG_DEFAULTS.items():
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
