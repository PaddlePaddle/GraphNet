import grpc
from concurrent import futures
import tempfile
import shutil
import tarfile
from pathlib import Path
from io import BytesIO
import subprocess
import os
import message_pb2
import message_pb2_grpc


class RemoteModelExecutorServicer(message_pb2_grpc.SampleRemoteExecutorServicer):

    def Execute(self, request, context):
        input_workspace = None
        output_workspace = None

        try:
            input_workspace = tempfile.mkdtemp(prefix="input_workspace_")
            output_workspace = tempfile.mkdtemp(prefix="output_workspace_")

            self._decompress_model(request.rpc_input.compressed_data, input_workspace)

            # 3. 构建 rpc_cmd 脚本路径
            rpc_cmd_path = os.path.join(input_workspace, request.rpc_cmd)

            # 4. 设置环境变量并执行 rpc_cmd 脚本
            env = os.environ.copy()
            env["INPUT_WORKSPACE"] = input_workspace
            env["OUTPUT_WORKSPACE"] = output_workspace
            env["OUTPUT_FILE_NAME"] = request.output_file_name

            result = subprocess.run(
                ["python3", rpc_cmd_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5分钟超时
            )

            if result.returncode != 0:
                return message_pb2.ExecutionReply(
                    ret_code=-1,
                    stderr=f"rpc_cmd 执行失败:\n{result.stderr}"
                )

            # 5. 读取 output_workspace/{output_file_name} 到 reply.rpc_output
            output_file_path = os.path.join(
                output_workspace,
                request.output_file_name
            )

            if not os.path.exists(output_file_path):
                return message_pb2.ExecutionReply(
                    ret_code=-1,
                    stderr=f"输出文件不存在: {output_file_path}"
                )

            # 读取 .npz 文件并返回
            return message_pb2.ExecutionReply(
                ret_code=0,
                stdout=result.stdout,
                stderr=result.stderr,
                rpc_output=message_pb2.RpcData(
                    npz_data=Path(output_file_path).read_bytes()
                )
            )

        except subprocess.TimeoutExpired:
            return message_pb2.ExecutionReply(
                ret_code=-1,
                stderr="rpc_cmd 执行超时（5分钟）"
            )

        except Exception as e:
            import traceback
            return message_pb2.ExecutionReply(
                ret_code=-1,
                stderr=f"{str(e)}\n{traceback.format_exc()}"
            )

        finally:
            # 6. 清理临时目录
            if input_workspace and os.path.exists(input_workspace):
                shutil.rmtree(input_workspace, ignore_errors=True)
            if output_workspace and os.path.exists(output_workspace):
                shutil.rmtree(output_workspace, ignore_errors=True)

    def _decompress_model(self, compressed_data, target_dir):
        buffer = BytesIO(compressed_data.payload)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(path=target_dir)


def serve(port=50052, max_workers=4):
    """启动 gRPC 服务器"""
    # 增加消息大小限制（支持最多 100MB）
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ]
    )
    message_pb2_grpc.add_SampleRemoteExecutorServicer_to_server(
        RemoteModelExecutorServicer(), server
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
