import argparse
import torch
import sys
import os
import os.path
import traceback
import types
from graph_net_bench import test_compiler_util
from graph_net_bench import path_utils
from .runner import RunnerConfig, RunResult, create_runner


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
        for a, b in zip(compiled_out, expected_out)
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


def _get_model_paths(args, model_path_prefix, use_model_list):
    if use_model_list:
        assert os.path.isdir(model_path_prefix) and os.path.isfile(args.model_path_list)

        test_samples = test_compiler_util.get_allow_samples(
            args.model_path_list, model_path_prefix
        )
        model_paths = [
            os.path.join(model_path_prefix, rel_model_path)
            for rel_model_path in test_samples
            if os.path.exists(
                os.path.join(model_path_prefix, rel_model_path, "model.py")
            )
        ]
    else:
        assert os.path.isdir(args.model_path)

        test_samples = test_compiler_util.get_allow_samples(
            args.model_path_list, model_path_prefix
        )
        model_paths = [
            model_path
            for model_path in path_utils.get_recursively_model_path(args.model_path)
            if test_samples is None or os.path.abspath(model_path) in test_samples
        ]

    return model_paths


def _create_model_args(model_path, reference_config, target_config):
    args = argparse.Namespace()
    args.model_path = model_path
    args.model_path_list = None
    args.reference_config = reference_config
    args.target_config = target_config
    return args


def eval_multi_models(args, model_path_prefix=None, use_model_list=False):
    module_name = os.path.splitext(os.path.basename(__file__))[0]

    model_paths = _get_model_paths(args, model_path_prefix, use_model_list)
    failed_samples = []
    for sample_idx, model_path in enumerate(model_paths):
        print(
            f"[{sample_idx}] {module_name}, model_path: {model_path}",
            file=sys.stderr,
            flush=True,
        )

        model_args = argparse.Namespace()
        model_args.model_path = model_path
        model_args.model_path_list = None
        model_args.reference_config = args.reference_config
        model_args.target_config = args.target_config

        try:
            eval_single_model(model_args)
            success = True
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(1)
        except Exception:
            print("\n--- Full Traceback ---")
            traceback.print_exc()
            success = False

        if not success:
            failed_samples.append(model_path)

    print(
        f"Totally {len(model_paths)} verified samples, failed {len(failed_samples)} samples.",
        file=sys.stderr,
        flush=True,
    )
    if failed_samples:
        for model_path in failed_samples:
            print(f"- {model_path}", file=sys.stderr, flush=True)


def eval_single_model(args):
    """
    Unified evaluation using Runner abstraction.
    Supports local, process, and remote execution via runner_type in config.
    """
    ref_dir = "/tmp/eval_perf_diff/reference"
    target_dir = "/tmp/eval_perf_diff/target"

    ref_config_dict = test_compiler_util.convert_to_dict(args.reference_config)
    target_config_dict = test_compiler_util.convert_to_dict(args.target_config)

    ref_runner_config = RunnerConfig.from_dict(ref_config_dict)
    target_runner_config = RunnerConfig.from_dict(target_config_dict)

    ref_runner = create_runner(ref_runner_config)
    target_runner = create_runner(target_runner_config)

    print(
        f"[eval_backend_diff] Reference runner: {ref_runner_config.strategy.runner_type.value}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[eval_backend_diff] Target runner: {target_runner_config.strategy.runner_type.value}",
        file=sys.stderr,
        flush=True,
    )

    ref_result = ref_runner.run(args.model_path, ref_dir)
    if not ref_result.success:
        raise RuntimeError(f"Reference run failed: {ref_result.error_message}")

    target_result = target_runner.run(args.model_path, target_dir)
    if not target_result.success:
        raise RuntimeError(f"Target run failed: {target_result.error_message}")

    compare_results(ref_result, target_result, ref_runner_config)


def compare_results(
    ref_result: RunResult, target_result: RunResult, config: RunnerConfig
):
    """Compare outputs and performance between reference and target results."""
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


def main(args):
    ref_config = test_compiler_util.convert_to_dict(args.reference_config)
    model_path_prefix = ref_config.get("model_path_prefix")

    if args.model_path_list and model_path_prefix:
        eval_multi_models(args, model_path_prefix, use_model_list=True)
    elif os.path.isdir(args.model_path):
        if path_utils.is_single_model_dir(args.model_path):
            eval_single_model(args)
        else:
            eval_multi_models(args, model_path_prefix, use_model_list=False)
    else:
        raise ValueError(f"Invalid model path: {args.model_path}")


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
