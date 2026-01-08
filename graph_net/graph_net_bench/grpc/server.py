import os
import sys
import subprocess
import tempfile
import shutil
import tarfile
from concurrent import futures
from io import BytesIO
from pathlib import Path

import grpc

import message_pb2
import message_pb2_grpc


class RemoteModelExecutorServicer(message_pb2_grpc.SampleRemoteExecutorServicer):

    def Execute(self, request, context):
        input_workspace = tempfile.mkdtemp(prefix="remote_input_")
        output_workspace = tempfile.mkdtemp(prefix="remote_output_")

        try:
            # 0) 基本校验
            if not request.rpc_cmd:
                return message_pb2.ExecutionReply(ret_code=-1, stderr="rpc_cmd is empty")

            if not request.HasField("output_file_name") or not request.output_file_name:
                return message_pb2.ExecutionReply(ret_code=-1, stderr="output_file_name is required")

            if request.rpc_input.WhichOneof("rpc_data_type") != "compressed_data":
                return message_pb2.ExecutionReply(
                    ret_code=-1,
                    stderr="rpc_input must be RpcData.compressed_data (tar.gz bytes)",
                )

            # 1) 解压输入到 input_workspace
            self._decompress_to_dir(request.rpc_input.compressed_data, input_workspace)

            # 2) 执行 rpc_cmd
            out_path = Path(output_workspace) / request.output_file_name

            env = os.environ.copy()
            env["INPUT_WORKSPACE"] = input_workspace
            env["OUTPUT_WORKSPACE"] = output_workspace
            env["OUTPUT_FILE_NAME"] = request.output_file_name
            env["OUTPUT_FILE_PATH"] = str(out_path)
            env["RANDOM_SEED"] = str(request.random_seed)
            # Add grpc directory to PYTHONPATH so message_pb2 can be imported
            grpc_dir = Path(__file__).parent.resolve()
            env["PYTHONPATH"] = f"{grpc_dir}:{env.get('PYTHONPATH', '')}"

            print(f"Executing rpc_cmd: {request.rpc_cmd}", file=sys.stderr)
            print(f"Working directory: {input_workspace}", file=sys.stderr)
            proc = subprocess.run(
                request.rpc_cmd,
                shell=True,
                cwd=input_workspace,
                env=env,
                capture_output=True,
                text=True,
            )

            print(f"returncode: {proc.returncode}", file=sys.stderr)
            print(f"stdout: {proc.stdout}", file=sys.stderr)
            print(f"stderr: {proc.stderr}", file=sys.stderr)

            if proc.returncode != 0:
                return message_pb2.ExecutionReply(
                    ret_code=proc.returncode,
                    stdout=proc.stdout or "",
                    stderr=proc.stderr or f"rpc_cmd failed with returncode={proc.returncode}",
                )

            # 3) 回读输出文件
            if not out_path.exists():
                print(f"Output file not found at {out_path}", file=sys.stderr)
                print(f"Contents of output_workspace: {list(Path(output_workspace).rglob('*'))}", file=sys.stderr)
                return message_pb2.ExecutionReply(
                    ret_code=-1,
                    stdout=proc.stdout or "",
                    stderr=(proc.stderr or "") + f"\nExpected output not found: {out_path}",
                )

            payload = out_path.read_bytes()

            return message_pb2.ExecutionReply(
                ret_code=0,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                rpc_output=message_pb2.RpcData(
                    compressed_data=message_pb2.CompressedData(
                        filename=request.output_file_name,
                        original_size=len(payload),
                        payload=payload,
                        compression_algo="raw",
                    )
                ),
            )

        except Exception as e:
            import traceback
            return message_pb2.ExecutionReply(ret_code=-1, stderr=f"{e}\n{traceback.format_exc()}")
        finally:
            shutil.rmtree(input_workspace, ignore_errors=True)
            shutil.rmtree(output_workspace, ignore_errors=True)

    def _decompress_to_dir(self, compressed_data: message_pb2.CompressedData, dst_dir: str) -> None:
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
    print(f"Server started on port {port}...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remote Model Server")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.max_workers)