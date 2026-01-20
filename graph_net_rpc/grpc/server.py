import os
import sys
import subprocess
import tempfile
import shutil
import tarfile
from concurrent import futures
from io import BytesIO
from pathlib import Path
from datetime import datetime
import grpc

from graph_net_rpc.grpc import message_pb2
from graph_net_rpc.grpc import message_pb2_grpc


class RemoteModelExecutorServicer(message_pb2_grpc.SampleRemoteExecutorServicer):
    def Execute(self, request, context):
        input_workspace = tempfile.mkdtemp(prefix="remote_input_")
        output_workspace = tempfile.mkdtemp(prefix="remote_output_")

        try:
            return self._execute_core(request, input_workspace, output_workspace)
        except Exception as e:
            import traceback

            return message_pb2.ExecutionReply(
                ret_code=-1, stderr=f"{e}\n{traceback.format_exc()}"
            )
        finally:
            shutil.rmtree(input_workspace, ignore_errors=True)
            shutil.rmtree(output_workspace, ignore_errors=True)

    def _execute_core(self, request, input_workspace: str, output_workspace: str):
        """Execute the RPC command core logic."""
        print(
            f"[RemoteModelExecutorServicer] {input_workspace=}, {output_workspace=}",
            flush=True,
            file=sys.stderr,
        )

        # Basic validation
        rpc_cmd = (request.rpc_cmd or "").strip()
        if not rpc_cmd:
            return message_pb2.ExecutionReply(ret_code=-1, stderr="rpc_cmd is empty")

        # Expect compressed input
        if request.rpc_input.WhichOneof("rpc_data_type") != "compressed_data":
            return message_pb2.ExecutionReply(
                ret_code=-1,
                stderr="rpc_input must be RpcData.compressed_data (tar.gz payload)",
            )

        # 1) decompress to input_workspace
        self._decompress_to_dir(request.rpc_input.compressed_data, input_workspace)

        env = os.environ.copy()
        env["INPUT_WORKSPACE"] = input_workspace
        env["OUTPUT_WORKSPACE"] = output_workspace
        # Ensure GraphNet repo root is on PYTHONPATH for the subprocess.
        repo_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[RemoteModelExecutorServicer][{timestamp}] Executing rpc_cmd: {rpc_cmd}",
            flush=True,
            file=sys.stderr,
        )
        try:
            proc = subprocess.run(
                rpc_cmd,
                shell=True,
                cwd=input_workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except KeyboardInterrupt:
            print("KeyboardInterrupt", flush=True, file=sys.stderr)
            exit(-1)
        except subprocess.TimeoutExpired as e:
            print(f"subprocess.TimeoutExpired: {e}", flush=True, file=sys.stderr)
            return message_pb2.ExecutionReply(
                ret_code=-1,
                stderr=f"Subprocess timed out after 300 seconds: {e}",
            )
        except Exception as e:
            print(f"Except: {e}", flush=True, file=sys.stderr)
            return message_pb2.ExecutionReply(
                ret_code=-5,
                stderr=f"[Subprocess error] {e}",
            )

        print(f"returncode: {proc.returncode}", flush=True, file=sys.stderr)
        print(f"stdout: {proc.stdout}", flush=True, file=sys.stderr)
        print(f"stderr: {proc.stderr}", flush=True, file=sys.stderr)

        if proc.returncode != 0:
            return message_pb2.ExecutionReply(
                ret_code=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr
                or f"rpc_cmd failed with returncode={proc.returncode}",
            )

        # 3) Pack OUTPUT_WORKSPACE to output.tar.gz
        output_tar_path = Path(output_workspace) / "output.tar.gz"
        print(f"Compressing {output_tar_path=}", flush=True, file=sys.stderr)
        with tarfile.open(output_tar_path, "w:gz") as tar:
            for file_path in Path(output_workspace).rglob("*"):
                if file_path.is_file() and file_path != output_tar_path:
                    arcname = file_path.relative_to(output_workspace)
                    print(
                        f"Add {file_path} to {output_tar_path}.",
                        flush=True,
                        file=sys.stderr,
                    )
                    tar.add(file_path, arcname=arcname)

        if not output_tar_path.exists():
            print(f"{output_tar_path=} no exists!", flush=True, file=sys.stderr)
            return message_pb2.ExecutionReply(
                ret_code=-1,
                stdout=proc.stdout or "",
                stderr=(proc.stderr or "")
                + f"\nNo output.tar.gz generated in {output_workspace}",
            )

        payload = output_tar_path.read_bytes()

        return message_pb2.ExecutionReply(
            ret_code=0,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            rpc_output=message_pb2.RpcData(
                compressed_data=message_pb2.CompressedData(
                    filename=output_tar_path.name,
                    original_size=len(payload),
                    payload=payload,
                    compression_algo="gzip",
                )
            ),
        )

    def _decompress_to_dir(
        self, compressed_data: message_pb2.CompressedData, dst_dir: str
    ) -> None:
        buffer = BytesIO(compressed_data.payload)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(path=dst_dir)


def serve(port=50052, max_workers=4):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    message_pb2_grpc.add_SampleRemoteExecutorServicer_to_server(
        RemoteModelExecutorServicer(),
        server,
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    print(f"Server started on port {port}...", flush=True, file=sys.stderr)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remote Model Server")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.max_workers)
