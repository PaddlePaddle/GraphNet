import argparse
import os
import sys
import types

import torch
from graph_net_bench import path_utils
from graph_net_bench import test_compiler_util
from graph_net import model_path_util
from graph_net_bench.torch import utils, eval_backend_perf, eval_backend_diff


def parse_config_from_reference_log(log_path):
    assert os.path.isfile(
        log_path
    ), f"{log_path} does not exist or is not a regular file."

    config = {}
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "[Processing]" in line:
                model_path = line.split("[Processing]")[1].strip()
                config["model_path"] = model_path
            if "[Config]" in line:
                config_line = line.split("[Config]")[1].strip()
                key, value = config_line.split(": ")
                config[key.strip()] = value.strip()
    return config


def get_ref_config_from_log(args, model_path):
    """Extract config from reference log file."""
    ref_log = utils.get_log_path(args.reference_dir, model_path)
    config = parse_config_from_reference_log(ref_log)
    return config


def convert_args_for_eval_backend(args, output_path):
    """Convert test_target_device args to eval_backend_perf args format."""
    return types.SimpleNamespace(
        model_path=args.model_path,
        output_path=output_path,
        seed=args.seed,
        compiler=args.compiler,
        device=args.device,
        op_lib=args.op_lib,
        warmup=args.warmup,
        trials=args.trials,
        log_prompt=args.log_prompt,
        backend_config=getattr(args, "config", None),
    )


def test_single_model(args):
    target_dir = "/tmp/eval_device_diff/target"

    ref_config = get_ref_config_from_log(args, args.model_path)
    vars(args)["compiler"] = ref_config.get("compiler")
    vars(args)["trials"] = int(ref_config.get("trials"))
    vars(args)["warmup"] = int(ref_config.get("warmup"))
    vars(args)["seed"] = int(ref_config.get("seed"))

    eval_args = convert_args_for_eval_backend(args, target_dir)
    eval_backend_perf.eval_single_model_with_single_backend(eval_args)

    ref_dump = utils.get_output_path(args.reference_dir, args.model_path)
    ref_out = torch.load(str(ref_dump))
    ref_log = utils.get_log_path(args.reference_dir, args.model_path)
    ref_time_stats = eval_backend_diff.parse_time_stats_from_reference_log(ref_log)

    target_dump = utils.get_output_path(target_dir, args.model_path)
    target_out = torch.load(str(target_dump))
    target_log = utils.get_log_path(target_dir, args.model_path)
    target_time_stats = eval_backend_diff.parse_time_stats_from_reference_log(
        target_log
    )

    eval_backend_diff.compare_correctness(ref_out, target_out, eval_args)
    test_compiler_util.print_times_and_speedup(args, ref_time_stats, target_time_stats)


def is_reference_log_exist(reference_dir, model_path):
    log_path = utils.get_log_path(reference_dir, model_path)
    return os.path.isfile(log_path)


def test_multi_models(args):
    assert os.path.isdir(args.reference_dir)

    test_samples = model_path_util.get_allow_samples(args.allow_list)

    sample_idx = 0
    failed_samples = []
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    for model_path in path_utils.get_recursively_model_path(args.model_path):
        if not is_reference_log_exist(args.reference_dir, model_path):
            continue

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
                    f"--device {args.device}",
                    f"--op-lib {args.op_lib}",
                    f"--log-prompt {args.log_prompt}",
                    f"--reference-dir {args.reference_dir}",
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
    assert args.device in ["cuda", "dcu", "xpu", "cpu"]

    if path_utils.is_single_model_dir(args.model_path):
        if args.op_lib == "origin":
            ref_config = get_ref_config_from_log(args, args.model_path)
            vars(args)["op_lib"] = ref_config.get("op_lib")
            print(
                f"{args.log_prompt} [Config] op_lib: {args.op_lib}",
                file=sys.stderr,
                flush=True,
            )
        else:
            eval_backend_perf.register_op_lib(args.op_lib)

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
        "--reference-dir",
        type=str,
        required=True,
        help="Directory to load reference stats log and outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device for testing the compiler (e.g., 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--op-lib",
        type=str,
        required=False,
        default="default",
        help="Customized operator library (eg. default, flaggems or origin)",
    )
    parser.add_argument(
        "--log-prompt",
        type=str,
        required=False,
        default="graph-net-test-device-log",
        help="Log prompt for performance log filtering.",
    )
    parser.add_argument(
        "--allow-list",
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
        help="Path to compiler configuration file or a JSON string",
    )
    args = parser.parse_args()
    main(args=args)
