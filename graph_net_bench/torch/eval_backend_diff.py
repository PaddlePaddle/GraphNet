"""Backend Performance Difference Evaluation Script.

Compares outputs and performance between reference and target compiler backends.
"""

import argparse
import os
import sys
import traceback
import types
from typing import Any, List, Optional, Tuple

import torch

from graph_net_bench import path_utils
from graph_net_bench import test_compiler_util
from .runner import RunnerConfig, RunResult, create_runner

_DEFAULT_REF_DIR = "/tmp/eval_perf_diff/reference"
_DEFAULT_TARGET_DIR = "/tmp/eval_perf_diff/target"


def _get_dtype_name(value: Any) -> str:
    """Extract dtype name from tensor or type name from other objects."""
    if isinstance(value, torch.Tensor):
        return str(value.dtype).replace("torch.", "")
    return type(value).__name__


def _extract_dtypes(outputs: List[Any]) -> List[str]:
    """Extract dtype/type names from a list of outputs."""
    return [_get_dtype_name(x) for x in outputs]


def compare_correctness(
    expected_out: List[torch.Tensor],
    compiled_out: List[torch.Tensor],
    args,
) -> None:
    """Compare correctness between expected and compiled outputs.

    Args:
        expected_out: List of expected output tensors.
        compiled_out: List of compiled output tensors.
        args: Arguments containing log_prompt and other settings.
    """
    eager_dtypes = _extract_dtypes(expected_out)
    compiled_dtypes = _extract_dtypes(compiled_out)

    type_match = test_compiler_util.check_output_datatype(
        args, eager_dtypes, compiled_dtypes
    )
    if not type_match:
        return

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


def get_cmp_equal(
    expected_out: List[torch.Tensor], compiled_out: List[torch.Tensor]
) -> str:
    """Get space-separated string of equality check results (1=equal, 0=not)."""
    return " ".join(
        str(int(torch.equal(a, b))) for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_all_close(
    expected_out: List[torch.Tensor],
    compiled_out: List[torch.Tensor],
    atol: float,
    rtol: float,
) -> str:
    """Get space-separated string of allclose check results."""
    return " ".join(
        str(int(torch.allclose(a, b, atol=atol, rtol=rtol)))
        for a, b in zip(compiled_out, expected_out)
    )


def _compute_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute absolute difference, converting to float for LongTensor compatibility."""
    return torch.abs(a.float() - b.float())


def get_cmp_max_diff(
    expected_out: List[torch.Tensor], compiled_out: List[torch.Tensor]
) -> str:
    """Get space-separated string of max absolute differences."""
    return " ".join(
        str(torch.max(_compute_abs_diff(a, b)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_mean_diff(
    expected_out: List[torch.Tensor], compiled_out: List[torch.Tensor]
) -> str:
    """Get space-separated string of mean absolute differences."""
    return " ".join(
        str(torch.mean(_compute_abs_diff(a, b)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def _count_diff_elements(
    a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float
) -> int:
    """Count number of differing elements between two tensors."""
    if a.is_floating_point() and b.is_floating_point():
        return torch.sum(~torch.isclose(a, b, atol=atol, rtol=rtol)).item()
    return torch.sum(a != b).item()


def get_cmp_diff_count(
    expected_out: List[torch.Tensor],
    compiled_out: List[torch.Tensor],
    atol: float,
    rtol: float,
) -> str:
    """Get space-separated string of element difference counts."""
    return " ".join(
        str(_count_diff_elements(a, b, atol, rtol))
        for a, b in zip(expected_out, compiled_out)
    )


def _has_model_file(path: str) -> bool:
    """Check if directory contains model.py."""
    return os.path.exists(os.path.join(path, "model.py"))


def _get_model_paths_from_list(
    model_path_list: str, model_path_prefix: str
) -> List[str]:
    """Get model paths from a list file with prefix."""
    assert os.path.isdir(model_path_prefix), f"Not a directory: {model_path_prefix}"
    assert os.path.isfile(model_path_list), f"Not a file: {model_path_list}"

    test_samples = test_compiler_util.get_allow_samples(
        model_path_list, model_path_prefix
    )
    return [
        os.path.join(model_path_prefix, rel_path)
        for rel_path in test_samples
        if _has_model_file(os.path.join(model_path_prefix, rel_path))
    ]


def _get_model_paths_from_dir(
    model_path: str, model_path_list: Optional[str], model_path_prefix: Optional[str]
) -> List[str]:
    """Get model paths by recursively scanning a directory."""
    assert os.path.isdir(model_path), f"Not a directory: {model_path}"

    test_samples = test_compiler_util.get_allow_samples(
        model_path_list, model_path_prefix
    )
    all_paths = path_utils.get_recursively_model_path(model_path)

    if test_samples is None:
        return list(all_paths)
    return [p for p in all_paths if os.path.abspath(p) in test_samples]


def _get_model_paths(
    args, model_path_prefix: Optional[str], use_model_list: bool
) -> List[str]:
    """Get list of model paths based on configuration."""
    if use_model_list:
        return _get_model_paths_from_list(args.model_path_list, model_path_prefix)
    return _get_model_paths_from_dir(
        args.model_path, args.model_path_list, model_path_prefix
    )


def _create_model_args(
    model_path: str, reference_config: str, target_config: str
) -> argparse.Namespace:
    """Create namespace for single model evaluation."""
    return argparse.Namespace(
        model_path=model_path,
        model_path_list=None,
        reference_config=reference_config,
        target_config=target_config,
    )


def _eval_single_model_safe(model_args: argparse.Namespace) -> bool:
    """Evaluate single model with exception handling.

    Returns:
        True if evaluation succeeded, False otherwise.
    """
    try:
        eval_single_model(model_args)
        return True
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        sys.exit(1)
    except Exception:
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        return False


def _print_evaluation_summary(total_count: int, failed_samples: List[str]) -> None:
    """Print summary of multi-model evaluation."""
    print(
        f"Totally {total_count} verified samples, failed {len(failed_samples)} samples.",
        file=sys.stderr,
        flush=True,
    )
    for model_path in failed_samples:
        print(f"- {model_path}", file=sys.stderr, flush=True)


def eval_multi_models(
    args,
    model_path_prefix: Optional[str] = None,
    use_model_list: bool = False,
) -> None:
    """Evaluate multiple models and collect results."""
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    model_paths = _get_model_paths(args, model_path_prefix, use_model_list)
    failed_samples: List[str] = []

    for sample_idx, model_path in enumerate(model_paths):
        print(
            f"[{sample_idx}] {module_name}, model_path: {model_path}",
            file=sys.stderr,
            flush=True,
        )
        model_args = _create_model_args(
            model_path, args.reference_config, args.target_config
        )
        success = _eval_single_model_safe(model_args)
        if not success:
            failed_samples.append(model_path)

    _print_evaluation_summary(len(model_paths), failed_samples)


def _parse_runner_configs(args) -> Tuple[RunnerConfig, RunnerConfig]:
    """Parse reference and target runner configurations."""
    return (
        RunnerConfig.from_dict(
            test_compiler_util.convert_to_dict(args.reference_config)
        ),
        RunnerConfig.from_dict(test_compiler_util.convert_to_dict(args.target_config)),
    )


def _log_runner_info(ref_config: RunnerConfig, target_config: RunnerConfig) -> None:
    """Log runner type information."""
    for label, cfg in [("Reference", ref_config), ("Target", target_config)]:
        print(
            f"[eval_backend_diff] {label} runner: {cfg.strategy.runner_type.value}",
            file=sys.stderr,
            flush=True,
        )


def _run_and_validate(
    runner, model_path: str, output_dir: str, label: str
) -> RunResult:
    """Run model and validate result."""
    result = runner.run(model_path, output_dir)
    if not result.success:
        raise RuntimeError(f"{label} run failed: {result.error_message}")
    return result


def eval_single_model(args) -> None:
    """Evaluate single model using Runner abstraction.

    Supports local, process, and remote execution via runner_type in config.
    """
    ref_runner_config, target_runner_config = _parse_runner_configs(args)
    _log_runner_info(ref_runner_config, target_runner_config)

    ref_runner = create_runner(ref_runner_config)
    target_runner = create_runner(target_runner_config)

    ref_result = _run_and_validate(
        ref_runner, args.model_path, _DEFAULT_REF_DIR, "Reference"
    )
    target_result = _run_and_validate(
        target_runner, args.model_path, _DEFAULT_TARGET_DIR, "Target"
    )

    compare_results(ref_result, target_result, ref_runner_config)


def compare_results(
    ref_result: RunResult, target_result: RunResult, config: RunnerConfig
) -> None:
    """Compare outputs and performance between reference and target results.

    Args:
        ref_result: Result from reference runner.
        target_result: Result from target runner.
        config: Runner configuration for logging settings.
    """
    if ref_result.outputs is None or target_result.outputs is None:
        print("[Warning] Cannot compare: missing outputs", file=sys.stderr)
        return

    dummy_args = types.SimpleNamespace(
        log_prompt=config.execution.log_prompt,
        compiler=config.execution.compiler,
        device=config.execution.device,
    )

    compare_correctness(ref_result.outputs, target_result.outputs, dummy_args)
    test_compiler_util.print_times_and_speedup(
        dummy_args, ref_result.time_stats, target_result.time_stats
    )


def main(args: argparse.Namespace) -> None:
    """Main entry point for backend difference evaluation.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If model_path is invalid.
    """
    ref_config = test_compiler_util.convert_to_dict(args.reference_config)
    model_path_prefix = ref_config.get("model_path_prefix")

    if args.model_path_list and model_path_prefix:
        eval_multi_models(args, model_path_prefix, use_model_list=True)
        return

    if not os.path.isdir(args.model_path):
        raise ValueError(f"Invalid model path: {args.model_path}")

    if path_utils.is_single_model_dir(args.model_path):
        eval_single_model(args)
        return

    eval_multi_models(args, model_path_prefix, use_model_list=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Backend Performance Difference."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default=None,
        help="Path to model file(s), each subdirectory containing graph_net.json will be regarded as a model",
    )
    parser.add_argument(
        "--model-path-list",
        type=str,
        required=False,
        default=None,
        help="Path to samples list, each line contains a sample path",
    )
    parser.add_argument(
        "--reference-config",
        type=str,
        required=True,
        help="base64 encode reference config json.",
    )
    parser.add_argument(
        "--target-config",
        type=str,
        required=True,
        help="base64 encode target config json.",
    )
    args = parser.parse_args()
    main(args=args)
