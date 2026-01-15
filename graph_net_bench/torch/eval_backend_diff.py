from . import utils
import subprocess
import argparse
import torch
import sys
import os
import os.path
import traceback
import json
import base64
import types
from graph_net_bench import test_compiler_util
from graph_net_bench import path_utils


def convert_to_dict(config_str):
    if config_str in {None, "", "null", "None"}:
        return {}
    config_str = base64.b64decode(config_str).decode("utf-8")
    config = json.loads(config_str)
    assert isinstance(config, dict), f"config should be a dict. {config_str=}"
    return config


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


def parse_time_stats_from_reference_log(log_path):
    assert os.path.isfile(
        log_path
    ), f"{log_path} does not exist or is not a regular file."

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "[Performance][eager]" in line:
                start = line.find("{")
                end = line.rfind("}")
                time_stats = json.loads(line[start : end + 1])
    return time_stats


def eval_multi_models(args, model_path_prefix):
    test_samples = test_compiler_util.get_allow_samples(
        args.model_path_list, model_path_prefix
    )

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
                    f"-m graph_net_bench.torch.{module_name}",
                    f"--model-path {model_path}",
                    f"--config {args.config}",
                ]
            )
            try:
                process = subprocess.Popen(cmd, shell=True)
                cmd_ret = process.wait()
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                sys.exit(1)
            except Exception:
                print("\n--- Full Traceback ---")
                traceback.print_exc()
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


def eval_multi_models_with_prefix(args, model_path_prefix):
    assert os.path.isdir(model_path_prefix)
    assert os.path.isfile(args.model_path_list)
    test_samples = test_compiler_util.get_allow_samples(
        args.model_path_list, model_path_prefix
    )
    py_module_name = os.path.splitext(os.path.basename(__file__))[0]
    for rel_model_path in test_samples:
        model_path = os.path.join(model_path_prefix, rel_model_path)
        if not os.path.exists(model_path):
            continue
        if not os.path.exists(os.path.join(model_path, "model.py")):
            continue
        cmd = " ".join(
            [
                sys.executable,
                f"-m graph_net_bench.torch.{py_module_name}",
                f"--model-path {model_path}",
                f"--config {args.config}",
            ]
        )
        try:
            process = subprocess.Popen(cmd, shell=True)
            process.wait()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(1)
        except Exception:
            print("\n--- Full Traceback ---")
            traceback.print_exc()


def compare_perf_diff(args, model_path, ref_dir, target_dir):
    # A
    ref_dump_path = utils.get_output_path(ref_dir, model_path)
    ref_out = torch.load(str(ref_dump_path))

    ref_log_path = utils.get_log_path(ref_dir, model_path)
    ref_time_stats = parse_time_stats_from_reference_log(ref_log_path)

    # B
    target_dump_path = utils.get_output_path(target_dir, model_path)
    target_out = torch.load(str(target_dump_path))

    target_log_path = utils.get_log_path(target_dir, model_path)
    target_time_stats = parse_time_stats_from_reference_log(target_log_path)

    compare_correctness(ref_out, target_out, args)

    test_compiler_util.print_times_and_speedup(args, ref_time_stats, target_time_stats)


def eval_single_model(args):
    ref_dir = "/tmp/eval_perf_diff/A"
    target_dir = "/tmp/eval_perf_diff/B"

    EvalCfg = types.SimpleNamespace(
        ref_env=types.SimpleNamespace(**convert_to_dict(args.config)["ref_env"]),
        target_env=types.SimpleNamespace(**convert_to_dict(args.config)["target_env"]),
    )

    ref_args = build_sub_args(EvalCfg.ref_env)
    target_args = build_sub_args(EvalCfg.target_env)

    run_sub_process(ref_args, args.model_path, ref_dir)
    run_sub_process(target_args, args.model_path, target_dir)
    compare_perf_diff(ref_args, args.model_path, ref_dir, target_dir)


def run_sub_process(env_args, model_path, output_path):
    cmd = [sys.executable, "-m", "graph_net_bench.torch.eval_backend_perf"]
    args_pairs = [
        ("--model-path", model_path),
        ("--output-path", output_path),
        ("--seed", str(env_args.seed)),
        ("--compiler", env_args.compiler),
        ("--device", env_args.device),
        ("--op-lib", env_args.op_lib),
        ("--warmup", str(env_args.warmup)),
        ("--trials", str(env_args.trials)),
        ("--log-prompt", env_args.log_prompt),
        ("--model-path-prefix", env_args.model_path_prefix),
        ("--config", env_args.backend_config),
    ]

    for arg_name, arg_value in args_pairs:
        if arg_value is not None:
            cmd.extend([arg_name, arg_value])

    subprocess.run(cmd, check=True)


def build_sub_args(env_ns: types.SimpleNamespace) -> argparse.Namespace:
    sub = argparse.Namespace()
    sub.seed = getattr(env_ns, "seed", 123)
    sub.compiler = getattr(env_ns, "compiler", None)
    sub.device = getattr(env_ns, "device", None)
    sub.op_lib = getattr(env_ns, "op_lib", None)
    sub.warmup = getattr(env_ns, "warmup", 3)
    sub.trials = getattr(env_ns, "trials", 5)
    sub.log_prompt = getattr(env_ns, "log_prompt", None)
    sub.model_path_prefix = getattr(env_ns, "model_path_prefix", None)
    sub.backend_config = getattr(env_ns, "backend_config", None)
    return sub


def main(args):
    config_dict = convert_to_dict(args.config)
    model_path_prefix = config_dict["ref_env"]["model_path_prefix"]
    if args.model_path_list is not None and model_path_prefix is not None:
        eval_multi_models_with_prefix(args, model_path_prefix)
        return
    assert os.path.isdir(args.model_path)

    if path_utils.is_single_model_dir(args.model_path):
        eval_single_model(args)
    else:
        eval_multi_models(args, model_path_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test compiler performance.")
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
        "--config",
        type=str,
        required=False,
        default=None,
        help="base64 encode configuration json.",
    )
    args = parser.parse_args()
    main(args=args)
