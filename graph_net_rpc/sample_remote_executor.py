import grpc
import tarfile
from pathlib import Path
from io import BytesIO
from typing import Tuple, Optional

import torch

from graph_net_rpc.grpc import message_pb2
from graph_net_rpc.grpc import message_pb2_grpc


class SampleRemoteExecutor:

    def __init__(self, machine: str, port: int, rpc_cmd: str = "python3 -m graph_net.torch.test_reference_device"):
        self.machine = machine
        self.port = port
        self.rpc_cmd = rpc_cmd
        self._channel: Optional[grpc.Channel] = None
        self._stub = None

    def _get_stub(self):
        if self._stub is None:
            self._channel = grpc.insecure_channel(f"{self.machine}:{self.port}")
            self._stub = message_pb2_grpc.SampleRemoteExecutorStub(self._channel)
        return self._stub

    def __call__(self, model_path: str, random_seed: int) -> dict:

        compressed_data = self._compress_dir(model_path)

        output_file_name = f"outputs_seed_{random_seed}"
        rpc_cmd = f"{self.rpc_cmd} --seed {int(random_seed)}"
        request = message_pb2.ExecutionRequest(
            rpc_cmd=rpc_cmd,
            rpc_input=message_pb2.RpcData(compressed_data=compressed_data),
            output_file_name=output_file_name,
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
            raise RuntimeError("Remote execution succeeded but rpc_output is not compressed_data")

        # decompress returned tar.gz file
        return self._extract_tar_to_dict(reply.rpc_output.compressed_data)

    def _compress_dir(self, model_path: str):
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

    def _extract_tar_to_dict(self, compressed_data: message_pb2.CompressedData) -> dict:
        """Extract tar.gz to {filename: bytes} dict."""
        buffer = BytesIO(compressed_data.payload)
        files_dict = {}
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    file_content = tar.extractfile(member).read()
                    files_dict[member.name] = file_content
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