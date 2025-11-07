import argparse
import importlib.util
import paddle
import time
import numpy as np
import random
import os
from pathlib import Path
import json
import re
import sys
import traceback
from graph_net import test_compiler_util
from graph_net.paddle import utils
from graph_net.paddle import test_compiler
from graph_net import path_utils
from graph_net import test_compiler_util


def read_config(log_path):
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


def read_time_stats(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "[Performance][eager]" in line:
                start = line.find("{")
                end = line.rfind("}")
                time_stats = json.loads(line[start : end + 1])
    return time_stats


def test_single_model(args):
    compiler = test_compiler.get_compiler_backend(args)
    test_compiler.check_and_print_gpu_utilization(compiler)

    input_dict = test_compiler.get_input_dict(args.model_path)
    model = test_compiler.get_model(args.model_path)
    model.eval()

    test_compiler_util.print_basic_config(
        args,
        test_compiler.get_hardward_name(args),
        test_compiler.get_compile_framework_version(args),
    )

    success = False
    time_stats = {}
    try:
        input_spec = test_compiler.get_input_spec(args.model_path)
        compiled_model = compiler(model, input_spec)
        outputs, time_stats = test_compiler.measure_performance(
            lambda: compiled_model(**input_dict), args, compiler, profile=False
        )
        success = True
    except Exception as e:
        print(
            f"Run model failed: {str(e)}\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )

    test_compiler_util.print_running_status(args, success)

    model_name = test_compiler_util.get_model_name(args.model_path)
    if test_compiler_util.get_subgraph_tag(args.model_path):
        model_name += "_" + test_compiler_util.get_subgraph_tag(args.model_path)

    ref_dump = Path(args.reference_dir) / f"{model_name}.pdout"
    ref_log = Path(args.reference_dir) / f"{model_name}.log"
    ref_out = paddle.load(str(ref_dump))
    ref_time_stats = read_time_stats(ref_log)

    if success:
        test_compiler.check_outputs(args, ref_out, outputs)

    test_compiler_util.print_times_and_speedup(args, ref_time_stats, time_stats)

    return 0


def find_log_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".log"):
                yield os.path.join(root, file)


def main(args):
    assert os.path.isdir(args.reference_dir)

    sample_idx = 0
    failed_samples = []

    for log_file in find_log_files(args.reference_dir):
        config = read_config(log_file)
        model_path = config.get("model_path")
        vars(args)["model_path"] = model_path
        vars(args)["compiler"] = config.get("compiler")
        vars(args)["trials"] = int(config.get("trials"))
        vars(args)["warmup"] = int(config.get("warmup"))
        test_compiler.set_seed(random_seed=int(config.get("seed")))

        print(
            f"[{sample_idx}] test_device, model_path: {model_path}",
            file=sys.stderr,
            flush=True,
        )
        if test_single_model(args) != 0:
            failed_samples.append(model_path)
        sample_idx += 1

    print(
        f"Totally {sample_idx} verified samples, failed {len(failed_samples)} samples.",
        file=sys.stderr,
        flush=True,
    )
    for model_path in failed_samples:
        print(f"- {model_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test compiler performance.")
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
        "--log-prompt",
        type=str,
        required=False,
        default="graph-net-test-device-log",
        help="Log prompt for performance log filtering.",
    )
    args = parser.parse_args()
    main(args=args)
