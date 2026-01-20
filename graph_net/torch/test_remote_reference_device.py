#!/usr/bin/env python3
"""
Remote reference device test client.
Executes test_reference_device.py on remote server and retrieves outputs.
"""

import argparse
import torch
import os
import sys
from pathlib import Path

from graph_net_rpc.sample_remote_executor import SampleRemoteExecutor
from graph_net_bench import path_utils
from graph_net.torch import test_reference_device
from graph_net import model_path_util


def _build_remote_rpc_cmd(args) -> str:
    cmd = "python3 -m graph_net.torch.test_reference_device"

    # Append required args for graph_net.torch.test_reference_device style entrypoint.
    cmd += ' --model-path "$INPUT_WORKSPACE"'
    cmd += ' --reference-dir "$OUTPUT_WORKSPACE"'

    # Keep CLI consistent with local runner.
    cmd += f" --compiler {args.compiler}"
    cmd += f" --device {args.device}"
    cmd += f" --op-lib {args.op_lib}"
    cmd += f" --warmup {args.warmup}"
    cmd += f" --trials {args.trials}"
    cmd += f" --seed {args.seed}"

    if args.allow_list is not None:
        cmd += f" --allow-list {args.allow_list}"
    if args.config is not None:
        cmd += f" --config {args.config}"
    if getattr(args, "log_prompt", None):
        cmd += f" --log-prompt {args.log_prompt}"
    return cmd


def test_single_model_remote(args):
    ref_log = test_reference_device.get_reference_log_path(
        args.reference_dir, args.model_path
    )
    ref_dump = test_reference_device.get_reference_output_path(
        args.reference_dir, args.model_path
    )

    print(f"Reference log path: {ref_log}", file=sys.stderr, flush=True)
    print(f"Reference outputs path: {ref_dump}", file=sys.stderr, flush=True)

    rpc_cmd = _build_remote_rpc_cmd(args)
    executor = SampleRemoteExecutor(machine=args.machine, port=args.port)
    try:
        print(f"Sending request to {args.machine}:{args.port}...", file=sys.stderr)
        print(f"Remote rpc_cmd: {rpc_cmd}", file=sys.stderr)
        files_dict = executor.execute(args.model_path, rpc_cmd)

        # returned files contain the log and output tensors (names may differ)
        log_filename = Path(ref_log).name
        pth_filename = Path(ref_dump).name

        available_logs = sorted([k for k in files_dict.keys() if k.endswith(".log")])
        available_pths = sorted([k for k in files_dict.keys() if k.endswith(".pth")])

        if log_filename not in files_dict and len(available_logs) == 1:
            log_filename = available_logs[0]
        if pth_filename not in files_dict and len(available_pths) == 1:
            pth_filename = available_pths[0]

        if log_filename in files_dict:
            with open(ref_log, "wb") as f:
                f.write(files_dict[log_filename])
            print(f"Saved log to {ref_log}", file=sys.stderr)
        else:
            print(
                f"Warning: log file not found in remote output. "
                f"expected={Path(ref_log).name}, available_logs={available_logs}",
                file=sys.stderr,
            )

        if pth_filename in files_dict:
            with open(ref_dump, "wb") as f:
                f.write(files_dict[pth_filename])
            print(f"Saved outputs to {ref_dump}", file=sys.stderr)

            try:
                outputs = torch.load(ref_dump)
                print(
                    f"Loaded {len(outputs) if isinstance(outputs, tuple) else 1} output tensor(s)",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"Warning: Failed to load outputs: {e}", file=sys.stderr)
        else:
            print(
                f"Warning: output file not found in remote output. "
                f"expected={Path(ref_dump).name}, available_pths={available_pths}",
                file=sys.stderr,
            )

        # print log to stderr
        if log_filename in files_dict:
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
                )

        print("Remote execution completed successfully!", file=sys.stderr)

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
            )

            cmd = " ".join(
                [
                    sys.executable,
                    f"-m graph_net.torch.{module_name}",
                    f"--model-path {model_path}",
                    f"--machine {args.machine}",
                    f"--port {args.port}",
                    f"--compiler {args.compiler}",
                    f"--device {args.device}",
                    f"--op-lib {args.op_lib}",
                    f"--warmup {args.warmup}",
                    f"--trials {args.trials}",
                    f"--seed {args.seed}",
                    f"--reference-dir {args.reference_dir}",
                    (f"--allow-list {args.allow_list}" if args.allow_list else ""),
                    (f'--rpc-cmd "{args.rpc_cmd}"' if args.rpc_cmd else ""),
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

    ref_dump_dir = Path(args.reference_dir)
    ref_dump_dir.mkdir(parents=True, exist_ok=True)

    if path_utils.is_single_model_dir(args.model_path):
        test_single_model_remote(args)
    else:
        test_multi_models_remote(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test compiler performance via remote execution."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--compiler", type=str, default="inductor")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--op-lib", type=str, default="default")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reference-dir", type=str, required=True)
    parser.add_argument("--allow-list", type=str, default=None)
    parser.add_argument("--log-prompt", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--machine", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50052)

    args = parser.parse_args()
    main(args=args)
