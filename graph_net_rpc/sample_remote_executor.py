import grpc
import tarfile
from pathlib import Path
from io import BytesIO
from typing import Optional, Dict

from graph_net_rpc.grpc import message_pb2
from graph_net_rpc.grpc import message_pb2_grpc


class SampleRemoteExecutor:
    """
    - compresses a local directory (model_path) into tar.gz payload
    - sends rpc_cmd and rpc_input (model_path) to the server
    - returns a dict of {relative_path_in_tar: bytes} extracted from the server's output tar.gz
    """

    def __init__(self, machine: str, port: int):
        self.machine = machine
        self.port = port
        self._channel: Optional[grpc.Channel] = None
        self._stub = None

    def _get_stub(self):
        if self._stub is None:
            # Default is 4MB (4194304), increase it to 32MB
            options = [
                ("grpc.max_send_message_length", 32 * 1024 * 1024),
                ("grpc.max_receive_message_length", 32 * 1024 * 1024),
            ]
            self._channel = grpc.insecure_channel(
                f"{self.machine}:{self.port}", options=options
            )
            self._stub = message_pb2_grpc.SampleRemoteExecutorStub(self._channel)
        return self._stub

    def execute(self, model_path: str, rpc_cmd: str) -> Dict[str, bytes]:
        compressed_data = self._compress_dir(model_path)

        request = message_pb2.ExecutionRequest(
            rpc_cmd=rpc_cmd,
            rpc_input=message_pb2.RpcData(compressed_data=compressed_data),
        )

        stub = self._get_stub()
        reply = stub.Execute(request)

        if reply.ret_code != 0:
            raise RuntimeError(
                "Remote execution failed:\n"
                f"ret_code={reply.ret_code}\n"
                f"stdout:\n{reply.stdout}\n"
                f"stderr:\n{reply.stderr}\n"
            )

        if reply.rpc_output.WhichOneof("rpc_data_type") != "compressed_data":
            raise RuntimeError(
                "Remote execution succeeded but rpc_output is not compressed_data"
            )

        return self._extract_tar_to_dict(reply.rpc_output.compressed_data)

    def _compress_dir(self, model_path: str) -> message_pb2.CompressedData:
        buffer = BytesIO()
        model_dir = Path(model_path)

        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for item in model_dir.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(model_dir)
                    tar.add(item, arcname=arcname)

        compressed_bytes = buffer.getvalue()

        return message_pb2.CompressedData(
            filename=f"{model_dir.name}.tar.gz",
            original_size=len(compressed_bytes),
            payload=compressed_bytes,
            compression_algo="gzip",
        )

    def _extract_tar_to_dict(
        self, compressed_data: message_pb2.CompressedData
    ) -> Dict[str, bytes]:
        buffer = BytesIO(compressed_data.payload)
        files_dict: Dict[str, bytes] = {}
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    files_dict[member.name] = f.read()
        return files_dict

    def close(self):
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
