import argparse
import os
import sys
import tempfile
import paddle
from datetime import datetime

from graph_net_rpc.sample_remote_executor import SampleRemoteExecutor
from graph_net_bench import path_utils
from graph_net_bench import test_compiler_util
from graph_net import model_path_util
from graph_net.paddle import (
    test_compiler,
    test_target_device,
    test_reference_device,
    test_remote_reference_device,
)


def test_single_model_remote(args):
    model_path = os.path.normpath(args.model_path)
    test_compiler_util.print_with_log_prompt(
        "[Processing]", model_path, args.log_prompt
    )

    ref_log = test_reference_device.get_reference_log_path(
        args.reference_dir, model_path
    )
    ref_dump = test_reference_device.get_reference_output_path(
        args.reference_dir, model_path
    )
    print(f"Reference log path: {ref_log}", file=sys.stderr, flush=True)
    print(f"Reference outputs path: {ref_dump}", file=sys.stderr, flush=True)

    rpc_cmd = test_remote_reference_device.build_remote_rpc_cmd(args)
    executor = SampleRemoteExecutor(machine=args.machine, port=args.port)
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] Sending request to {args.machine}:{args.port}...",
            file=sys.stderr,
            flush=True,
        )
        print(f"Remote rpc_cmd: {rpc_cmd}", file=sys.stderr, flush=True)

        files_dict = executor.execute(args.model_path, rpc_cmd)
        print(f"Returned keys: {files_dict.keys()}", file=sys.stderr, flush=True)

        (
            log_filename,
            out_filename,
        ) = test_remote_reference_device.parse_returned_output_and_log_filename(
            files_dict, ref_log, ref_dump
        )

        recieved_log = files_dict.get(log_filename, None)
        recieved_outputs = files_dict.get(out_filename, None)

        outputs, time_stats = None, None
        with tempfile.TemporaryDirectory() as temp_dir:
            if recieved_log:
                temp_log_path = os.path.join(temp_dir, os.path.basename(ref_log))
                with open(temp_log_path, "wb") as f:
                    f.write(recieved_log)
                print(f"Saved log to {temp_log_path}", file=sys.stderr, flush=True)
                time_stats = test_target_device.parse_time_stats_from_log_file(
                    temp_log_path
                )

            if recieved_outputs:
                temp_output_path = os.path.join(temp_dir, os.path.basename(ref_dump))
                with open(temp_output_path, "wb") as f:
                    f.write(recieved_outputs)
                print(
                    f"Saved outputs to {temp_output_path}", file=sys.stderr, flush=True
                )
                outputs = paddle.load(temp_output_path)

        if recieved_log is not None:
            # print log to stderr
            try:
                filtered_log = "\n".join(
                    line
                    for line in recieved_log.decode("utf-8").splitlines()
                    if "[Processing]" not in line
                )
                print(filtered_log, file=sys.stderr, flush=True)
            except Exception:
                print(
                    f"Warning: Failed to decode remote log as utf-8; printing bytes length only {len(recieved_log)}",
                    file=sys.stderr,
                    flush=True,
                )

        (
            ref_out,
            ref_time_stats,
        ) = test_target_device.get_reference_output_and_time_stats(
            model_path, args.reference_dir
        )
        if outputs is not None and ref_out is not None:
            test_compiler.check_outputs(args, ref_out, outputs)

        test_compiler_util.print_times_and_speedup(args, ref_time_stats, time_stats)
        print("Remote execution completed successfully!", file=sys.stderr)

    except Exception as e:
        print(f"Remote execution failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        executor.close()


def test_multi_models_remote(args):
    assert os.path.isdir(args.reference_dir)

    test_samples = model_path_util.get_allow_samples(args.allow_list)

    sample_idx = 0
    failed_samples = []
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    for model_path in path_utils.get_recursively_model_path(args.model_path):
        if test_samples is None or os.path.abspath(model_path) in test_samples:
            print(
                f"[{sample_idx}] {module_name}, model_path: {model_path}",
                file=sys.stderr,
            )

            cmd = " ".join(
                [
                    sys.executable,
                    f"-m graph_net.paddle.{module_name}",
                    f"--model-path {model_path}",
                    f"--compiler {args.compiler}",
                    f"--device {args.device}",
                    f"--warmup {args.warmup}",
                    f"--trials {args.trials}",
                    f"--log-prompt {args.log_prompt}",
                    f"--seed {args.seed}",
                    f"--reference-dir {args.reference_dir}",
                    f"--machine {args.machine}",
                    f"--port {args.port}",
                ]
            ).strip()

            cmd_ret = os.system(cmd)
            if cmd_ret != 0:
                failed_samples.append(model_path)
            sample_idx += 1

    print(
        f"Totally {sample_idx} verified samples, failed {len(failed_samples)} samples.",
        file=sys.stderr,
    )
    for model_path in failed_samples:
        print(f"- {model_path}", file=sys.stderr)


def main(args):
    assert os.path.isdir(args.model_path)
    assert args.device in ["cuda", "dcu", "xpu", "cpu"]

    test_compiler.init_env(args)

    if path_utils.is_single_model_dir(args.model_path):
        test_single_model_remote(args)
    else:
        test_multi_models_remote(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test target device via remote execution."
    )
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
        help="Directory to load reference stats log and outputs",
    )

    parser.add_argument("--machine", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50052)

    args = parser.parse_args()
    main(args=args)
