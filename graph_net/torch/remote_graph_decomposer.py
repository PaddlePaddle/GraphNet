import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from graph_net_rpc.sample_remote_executor import SampleRemoteExecutor


def _build_remote_rpc_cmd(args) -> str:
    """
    Build the command string that will be executed on the remote server.

    The gRPC server will set:
      INPUT_WORKSPACE=<uploaded model dir>
      OUTPUT_WORKSPACE=<remote output dir>

    So we append required args with those env vars.
    """
    cmd = (args.rpc_cmd or "").strip()
    if not cmd:
        raise ValueError("rpc_cmd is empty")

    # For our recommended server entrypoint, append the required parameters.
    # We keep this logic permissive: if user provides a different rpc_cmd, they can
    # still rely on these appended args as long as their entrypoint accepts them.
    #
    # Important: The decorator-config is a (potentially long) base64 string; we wrap it
    # in double quotes. If the string itself contains quotes (shouldn't), user must handle it.
    cmd += ' --model-path "$INPUT_WORKSPACE"'
    cmd += f' --decorator-config "{args.decorator_config}"'

    # optional: let server choose log file name; we still pass it so server can avoid collisions
    if args.remote_log_file:
        cmd += f' --log-file "{args.remote_log_file}"'

    return cmd


def _write_files_dict_to_dir(files_dict: Dict[str, bytes], dst_dir: str) -> None:
    """
    Materialize returned {relative_path: bytes} into dst_dir/relative_path.
    """
    dst_root = Path(dst_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    for rel_path, content in files_dict.items():
        # Normalize tar paths (always forward slashes)
        rel_path = rel_path.lstrip("/")

        out_path = dst_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content)


def _print_remote_log(files_dict: Dict[str, bytes], log_name: Optional[str]) -> None:
    """
    Print remote log to stderr if found.
    """
    if not log_name:
        return

    # Try exact match first.
    if log_name in files_dict:
        data = files_dict[log_name]
    else:
        # Fallback: try to find any .txt or .log likely produced by server.
        candidates = sorted(
            [k for k in files_dict.keys() if k.endswith((".log", ".txt"))]
        )
        if len(candidates) == 1:
            data = files_dict[candidates[0]]
            log_name = candidates[0]
        else:
            print(
                f"[remote_subgraph_decomposer] Remote log not found. expected={log_name}, candidates={candidates}",
                file=sys.stderr,
                flush=True,
            )
            return

    try:
        text = data.decode("utf-8", errors="replace")
        print(f"\n===== Remote log: {log_name} =====", file=sys.stderr, flush=True)
        print(text, file=sys.stderr, flush=True)
        print("===== End remote log =====\n", file=sys.stderr, flush=True)
    except Exception as e:
        print(
            f"[remote_subgraph_decomposer] Failed to decode remote log {log_name}: {e} (bytes={len(data)})",
            file=sys.stderr,
            flush=True,
        )


def main(args):
    model_path = os.path.abspath(args.model_path)
    assert os.path.isdir(model_path), f"--model-path must be a directory: {model_path}"

    local_out_dir = os.path.abspath(args.output_dir)
    Path(local_out_dir).mkdir(parents=True, exist_ok=True)

    rpc_cmd = _build_remote_rpc_cmd(args)

    print(
        f"[remote_subgraph_decomposer] model_path: {model_path}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[remote_subgraph_decomposer] output_dir: {local_out_dir}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[remote_subgraph_decomposer] server: {args.machine}:{args.port}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[remote_subgraph_decomposer] remote rpc_cmd: {rpc_cmd}",
        file=sys.stderr,
        flush=True,
    )

    executor = SampleRemoteExecutor(machine=args.machine, port=args.port)
    try:
        files_dict = executor.execute(model_path, rpc_cmd)

        # Write everything returned by server OUTPUT_WORKSPACE into local_out_dir.
        _write_files_dict_to_dir(files_dict, local_out_dir)
        # Optionally print log
        _print_remote_log(files_dict, args.remote_log_file)

        print(
            f"[remote_subgraph_decomposer] Done. Materialized {len(files_dict)} file(s) under {local_out_dir}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        executor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remote subgraph decomposer client (torch)."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a *single model directory* to upload to the server.",
    )
    parser.add_argument(
        "--decorator-config",
        type=str,
        required=True,
        help="Base64-encoded decorator config JSON (same as passed to graph_net.torch.run_model).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Local output directory to materialize files returned from server OUTPUT_WORKSPACE. "
            "Typically this should be the pass workspace dir, e.g. <pass_dir> (so that it contains samples/...)."
        ),
    )
    parser.add_argument("--machine", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50052)

    parser.add_argument(
        "--rpc-cmd",
        type=str,
        default="python3 -m graph_net.torch.remote_graph_decomposer_entry",
        help=(
            "Command to execute on remote server. It should accept "
            "--model-path and --decorator-config (and optionally --log-file)."
        ),
    )

    parser.add_argument(
        "--remote-log-file",
        type=str,
        default="log_remote_graph_decomposer.txt",
        help="Remote log filename expected to be produced under OUTPUT_WORKSPACE (optional).",
    )

    args = parser.parse_args()
    main(args)
