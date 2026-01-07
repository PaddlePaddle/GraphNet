import grpc
import message_pb2
import message_pb2_grpc
import argparse
import tarfile
from pathlib import Path
from io import BytesIO


def _compress_model(model_path: str):
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        model_dir = Path(model_path)
        for item in model_dir.rglob("*"):
            if item.is_file():
                arcname = item.relative_to(model_dir)
                tar.add(item, arcname=arcname)

    compressed_bytes = buffer.getvalue()

    return message_pb2.CompressedData(
        filename=f"{Path(model_path).name}.tar.gz",
        original_size=len(compressed_bytes),
        payload=compressed_bytes,
        compression_algo="gzip"
    )


def run(server_ip: str = "localhost", port: int = 50052, timeout: float = 10.0,
        rpc_cmd: str = "execute_model", output_file_name: str = "42",
        model_path: str = None):
    channel = grpc.insecure_channel(
        f"{server_ip}:{port}",
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ]
    )
    stub = message_pb2_grpc.SampleRemoteExecutorStub(channel)

    if model_path:
        # 发送压缩的模型数据
        compressed_data = _compress_model(model_path)
        request = message_pb2.ExecutionRequest(
            rpc_cmd=rpc_cmd,
            rpc_input=message_pb2.RpcData(compressed_data=compressed_data),
            output_file_name=output_file_name
        )
    else:
        # 发送简单的字符串数据（用于测试连通性）
        request = message_pb2.ExecutionRequest(
            rpc_cmd=rpc_cmd,
            rpc_input=message_pb2.RpcData(str_data="test data"),
            output_file_name=output_file_name
        )

    try:
        response = stub.Execute(request, timeout=timeout)
        print(f"ret_code: {response.ret_code}")
        # print(f"stdout: {response.stdout}")
        print(f"stderr: {response.stderr}")
        if response.stdout:
            print(f"rpc_output: {response.rpc_output}")
    except grpc.RpcError as e:
        print(f"gRPC 调用失败: {e.code()}: {e.details()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gRPC Client")
    parser.add_argument("--server-ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--rpc-cmd", type=str, default="execute_model")
    parser.add_argument("--output-file-name", type=str, default="42")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model directory (optional)")
    args = parser.parse_args()

    run(server_ip=args.server_ip, port=args.port, timeout=args.timeout,
        rpc_cmd=args.rpc_cmd, output_file_name=args.output_file_name,
        model_path=args.model_path)
