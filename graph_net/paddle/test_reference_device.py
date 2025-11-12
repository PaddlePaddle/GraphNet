import argparse
import importlib.util
import paddle
import time
import numpy as np
import random
import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import json
import re
import sys
import traceback
from graph_net import test_compiler_util
from graph_net.paddle import utils
from graph_net.paddle import test_compiler
from graph_net import path_utils
from graph_net import test_compiler_util


def test_single_model(args):
    model_name = test_compiler_util.get_model_name(args.model_path)
    if test_compiler_util.get_subgraph_tag(args.model_path):
        model_name += "_" + test_compiler_util.get_subgraph_tag(args.model_path)
    ref_log = Path(args.reference_dir) / f"{model_name}.log"
    with open(ref_log, "w", encoding="utf-8") as log_f:
        with redirect_stdout(log_f), redirect_stderr(log_f):
            compiler = test_compiler.get_compiler_backend(args)
            test_compiler.check_and_print_gpu_utilization(compiler)

            input_dict = test_compiler.get_input_dict(args.model_path)
            model = test_compiler.get_model(args.model_path)
            model.eval()

            test_compiler_util.print_with_log_prompt(
                "[Config] seed:", args.seed, args.log_prompt
            )

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
                    lambda: compiled_model(**input_dict),
                    args,
                    compiler,
                    profile=False,
                )
                success = True
            except Exception as e:
                print(
                    f"Run model failed: {str(e)}\n{traceback.format_exc()}",
                    file=sys.stderr,
                    flush=True,
                )

            test_compiler_util.print_running_status(args, success)
            if success:
                ref_dump = Path(args.reference_dir) / f"{model_name}.pdout"
                paddle.save(outputs, str(ref_dump))
            test_compiler_util.print_with_log_prompt(
                "[Performance][eager]:", json.dumps(time_stats), args.log_prompt
            )


def test_multi_models(args):
    test_samples = test_compiler_util.get_allow_samples(args.allow_list)

    sample_idx = 0
    failed_samples = []
    for model_path in path_utils.get_recursively_model_path(args.model_path):
        if test_samples is None or os.path.abspath(model_path) in test_samples:
            print(
                f"[{sample_idx}] test_compiler, model_path: {model_path}",
                file=sys.stderr,
                flush=True,
            )
            cmd = " ".join(
                [
                    sys.executable,
                    "-m graph_net.paddle.test_reference_device",
                    f"--model-path {model_path}",
                    f"--compiler {args.compiler}",
                    f"--device {args.device}",
                    f"--warmup {args.warmup}",
                    f"--trials {args.trials}",
                    f"--log-prompt {args.log_prompt}",
                    f"--seed {args.seed}",
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
    assert args.compiler in {"cinn", "nope"}
    assert args.device in ["cuda"]

    test_compiler.set_seed(random_seed=args.seed)

    ref_dump_dir = Path(args.reference_dir)
    ref_dump_dir.mkdir(parents=True, exist_ok=True)

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
        default="cinn",
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
        "--warmup", type=int, required=False, default=5, help="Number of warmup steps"
    )
    parser.add_argument(
        "--trials", type=int, required=False, default=10, help="Number of timing trials"
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
        "--seed",
        type=int,
        required=False,
        default=123,
        help="Random seed (default: 123)",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        required=True,
        help="Directory to save reference stats log and outputs",
    )
    args = parser.parse_args()
    main(args=args)
