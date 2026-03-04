import argparse
import paddle
import os
import sys
from pathlib import Path

from graph_net_rpc.sample_remote_executor import SampleRemoteExecutor
from graph_net_bench import path_utils
from graph_net import model_path_util
from graph_net.paddle import test_reference_device


def build_remote_rpc_cmd(args) -> str:
    cmd = "python -m graph_net.paddle.test_reference_device"

    # Append required args for graph_net.paddle.test_reference_device style entrypoint.
    cmd += ' --model-path "$INPUT_WORKSPACE"'
    cmd += ' --reference-dir "$OUTPUT_WORKSPACE"'

    # Keep CLI consistent with local runner.
    cmd += f" --compiler {args.compiler}"
    cmd += f" --device {args.device}"
    cmd += f" --warmup {args.warmup}"
    cmd += f" --trials {args.trials}"
    cmd += f" --seed {args.seed}"
    cmd += f" --log-prompt {args.log_prompt}"
    return cmd


def parse_returned_output_and_log_filename(files_dict, log_path, output_dump_path):
    # returned files contain the log and output tensors (names may differ)
    log_filename = Path(log_path).name
    out_filename = Path(output_dump_path).name

    available_logs = sorted([k for k in files_dict.keys() if k.endswith(".log")])
    available_outs = sorted([k for k in files_dict.keys() if k.endswith(".pdout")])

    if log_filename not in files_dict and len(available_logs) == 1:
        log_filename = available_logs[0]
    if out_filename not in files_dict and len(available_outs) == 1:
        out_filename = available_outs[0]

    if log_filename not in available_logs:
        print(
            f"Warning: log file not found in remote output. "
            f"expected={Path(log_path).name}, available_logs={available_logs}",
            file=sys.stderr,
            flush=True,
        )
    if out_filename not in available_outs:
        print(
            f"Warning: output file not found in remote output. "
            f"expected={Path(output_dump_path).name}, available_outs={available_outs}",
            file=sys.stderr,
            flush=True,
        )

    return log_filename, out_filename


def test_single_model_remote(args):
    model_path = os.path.normpath(args.model_path)
    ref_log = test_reference_device.get_reference_log_path(
        args.reference_dir, model_path
    )
    ref_dump = test_reference_device.get_reference_output_path(
        args.reference_dir, model_path
    )
    print(f"Reference log path: {ref_log}", file=sys.stderr, flush=True)
    print(f"Reference outputs path: {ref_dump}", file=sys.stderr, flush=True)

    rpc_cmd = build_remote_rpc_cmd(args)
    executor = SampleRemoteExecutor(machine=args.machine, port=args.port)
    try:
        print(
            f"Sending request to {args.machine}:{args.port}...",
            file=sys.stderr,
            flush=True,
        )
        print(f"Remote rpc_cmd: {rpc_cmd}", file=sys.stderr, flush=True)

        files_dict = executor.execute(model_path, rpc_cmd)
        log_filename, out_filename = parse_returned_output_and_log_filename(
            files_dict, ref_log, ref_dump
        )

        if log_filename in files_dict:
            with open(ref_log, "wb") as f:
                f.write(files_dict[log_filename])
            print(f"Saved log to {ref_log}", file=sys.stderr, flush=True)

        if out_filename in files_dict:
            with open(ref_dump, "wb") as f:
                f.write(files_dict[out_filename])
            print(f"Saved outputs to {ref_dump}", file=sys.stderr, flush=True)

            try:
                outputs = paddle.load(ref_dump)
                print(
                    f"Loaded {len(outputs) if isinstance(outputs, tuple) else 1} output tensor(s)",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load outputs: {e}", file=sys.stderr, flush=True
                )

        if log_filename in files_dict:
            # print log to stderr
            try:
                print(
                    files_dict[log_filename].decode("utf-8"),
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                print(
                    "Warning: Failed to decode remote log as utf-8; printing bytes length only: "
                    f"{len(files_dict[log_filename])}",
                    file=sys.stderr,
                    flush=True,
                )

        print("Remote execution completed successfully!", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"Remote execution failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        executor.close()


def test_multi_models_remote(args):
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
        flush=True,
    )
    for model_path in failed_samples:
        print(f"- {model_path}", file=sys.stderr, flush=True)


def main(args):
    assert os.path.isdir(args.model_path)
    assert args.compiler in {"cinn", "nope"}
    assert args.device in ["cuda"]

    ref_dump_dir = Path(args.reference_dir)
    ref_dump_dir.mkdir(parents=True, exist_ok=True)

    if path_utils.is_single_model_dir(args.model_path):
        test_single_model_remote(args)
    else:
        test_multi_models_remote(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test reference device performance via remote execution."
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
        default="nope",
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

    parser.add_argument("--machine", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50052)
    args = parser.parse_args()
    main(args=args)
