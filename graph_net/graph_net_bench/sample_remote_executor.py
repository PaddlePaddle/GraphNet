import grpc
import tarfile
from pathlib import Path
from io import BytesIO
from typing import Tuple, Optional

import torch

from graph_net.graph_net_bench.grpc import message_pb2
from graph_net.graph_net_bench.grpc import message_pb2_grpc


class SampleRemoteExecutor:

    def __init__(self, machine: str, port: int, rpc_cmd: str = "python3 /denghaodong/code/GraphNet/graph_net/graph_net_bench/sample_rpc_cmd.py"):
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

    def __call__(self, model_path: str, random_seed: int) -> Tuple[torch.Tensor, ...]:

        compressed_data = self._compress_dir(model_path)

        # 输出文件名必须包含扩展名（mentor 约定）
        output_file_name = f"outputs_seed_{random_seed}.npz"

        request = message_pb2.ExecutionRequest(
            rpc_cmd=self.rpc_cmd,
            rpc_input=message_pb2.RpcData(compressed_data=compressed_data),
            output_file_name=output_file_name,
            random_seed=int(random_seed),
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

        npz_bytes = reply.rpc_output.compressed_data.payload
        return self._npz_bytes_to_tensors(npz_bytes)

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

    def _npz_bytes_to_tensors(self, npz_bytes: bytes) -> Tuple[torch.Tensor, ...]:
        import numpy as np

        with np.load(BytesIO(npz_bytes), allow_pickle=False) as npz:
            # 只接受 output_{i} 格式，按 i 排序，保证返回 tuple 稳定
            keys = [k for k in npz.files if k.startswith("output_")]

            def key_index(k: str) -> int:
                # "output_0" -> 0
                return int(k.split("_", 1)[1])

            keys.sort(key=key_index)

            tensors = []
            for k in keys:
                arr = npz[k]
                tensors.append(torch.from_numpy(arr))  # 默认 CPU
            return tuple(tensors)

    def close(self):
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()