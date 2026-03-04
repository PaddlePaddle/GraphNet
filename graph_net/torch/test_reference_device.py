import argparse
import os
import sys
import types
import torch
from pathlib import Path

from graph_net_bench import path_utils
from graph_net import model_path_util
from graph_net_bench.torch import eval_backend_perf


def convert_args_for_eval_backend(args):
    """Convert test_reference_device args to eval_backend_perf args format."""
    return types.SimpleNamespace(
        model_path=args.model_path,
        output_path=args.reference_dir,
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
    eval_args = convert_args_for_eval_backend(args)
    eval_backend_perf.eval_single_model_with_single_backend(eval_args)


def test_multi_models(args):
    test_samples = model_path_util.get_allow_samples(args.allow_list)

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
                    f"-m graph_net.torch.{module_name}",
                    f"--model-path {model_path}",
                    f"--compiler {args.compiler}",
                    f"--device {args.device}",
                    f"--op-lib {args.op_lib}",
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
    assert args.device in ["cuda", "cpu"]

    eval_backend_perf.set_seed(args.seed)
    torch.set_default_device(args.device)

    ref_dump_dir = Path(args.reference_dir)
    ref_dump_dir.mkdir(parents=True, exist_ok=True)

    if path_utils.is_single_model_dir(args.model_path):
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
        "--compiler",
        type=str,
        required=False,
        default="inductor",
        help="Compiler backend to use",
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
        help="Customized operator library (eg. default, flaggems)",
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
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=None,
        help="Path to compiler configuration file or a JSON string",
    )
    args = parser.parse_args()
    main(args=args)
