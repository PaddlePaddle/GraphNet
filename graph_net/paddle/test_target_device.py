import argparse
import os
import json
import sys
import traceback

import paddle
from graph_net_bench import path_utils
from graph_net import test_compiler_util
from graph_net.paddle import test_compiler, test_reference_device


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


def update_args_and_set_seed(args, model_path):
    ref_log = test_reference_device.get_reference_log_path(
        args.reference_dir, model_path
    )
    config = parse_config_from_reference_log(ref_log)
    vars(args)["model_path"] = model_path
    vars(args)["compiler"] = config.get("compiler")
    vars(args)["trials"] = int(config.get("trials"))
    vars(args)["warmup"] = int(config.get("warmup"))
    test_compiler.set_seed(random_seed=int(config.get("seed")))
    return args


def test_single_model(args):
    model_path = os.path.normpath(args.model_path)
    test_compiler_util.print_with_log_prompt(
        "[Processing]", model_path, args.log_prompt
    )
    args = update_args_and_set_seed(args, model_path)

    compiler = test_compiler.get_compiler_backend(args)
    test_compiler.check_and_print_gpu_utilization(compiler)

    input_dict = test_compiler.get_input_dict(model_path)
    model = test_compiler.get_model(model_path)
    model.eval()

    test_compiler_util.print_basic_config(
        args,
        test_compiler.get_hardward_name(args),
        test_compiler.get_compile_framework_version(args),
    )

    success = False
    time_stats = {}
    try:
        input_spec = test_compiler.get_input_spec(model_path)
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

    model_name = test_compiler_util.get_model_name(model_path)
    if test_compiler_util.get_subgraph_tag(model_path):
        model_name += "_" + test_compiler_util.get_subgraph_tag(model_path)

    ref_dump = test_reference_device.get_reference_output_path(
        args.reference_dir, model_path
    )
    ref_out = paddle.load(str(ref_dump))

    ref_log = test_reference_device.get_reference_log_path(
        args.reference_dir, model_path
    )
    ref_time_stats = parse_time_stats_from_reference_log(ref_log)

    if success:
        test_compiler.check_outputs(args, ref_out, outputs)

    test_compiler_util.print_times_and_speedup(args, ref_time_stats, time_stats)


def is_reference_log_exist(reference_dir, model_path):
    log_path = test_reference_device.get_reference_log_path(reference_dir, model_path)
    return os.path.isfile(log_path)


def test_multi_models(args):
    assert os.path.isdir(args.reference_dir)

    test_samples = test_compiler_util.get_allow_samples(args.allow_list)

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
                    f"-m graph_net.paddle.{module_name}",
                    f"--model-path {model_path}",
                    f"--device {args.device}",
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

    test_compiler.init_env(args)

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
    parser.add_argument(
        "--allow-list",
        type=str,
        required=False,
        default=None,
        help="Path to samples list, each line contains a sample path",
    )
    args = parser.parse_args()
    main(args=args)
