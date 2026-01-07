import grpc
import tarfile
import json
from pathlib import Path
from io import BytesIO
from typing import Tuple, Optional
from contextlib import contextmanager
import torch


class SampleRemoteExecutor:
    
    def __init__(self, machine: str, port: int):
        self.machine = machine
        self.port = port
        self._channel: Optional[grpc.Channel] = None
        self._stub = None
    
    def _get_stub(self):
        if self._stub is None:
            from .grpc import message_pb2, message_pb2_grpc
            self._channel = grpc.insecure_channel(
                f"{self.machine}:{self.port}",
                options=[
                    ("grpc.max_receive_message_length", 32 * 1024 * 1024),  # 32MB
                    ("grpc.max_send_message_length", 32 * 1024 * 1024),
                ],)
            self._stub = message_pb2_grpc.SampleRemoteExecutorStub(self._channel)
        return self._stub
    
    def __call__(self, model_path: str, random_seed: int) -> Tuple[torch.Tensor, ...]:
        """远程执行模型"""
        from .grpc import message_pb2
        
        compressed_data = self._compress_model(model_path)
        
        stub = self._get_stub()
        request = message_pb2.ExecutionRequest(
            rpc_cmd="execute_model",
            rpc_input=message_pb2.RpcData(compressed_data=compressed_data),
            output_file_name=str(random_seed)
        )
        
        reply = stub.Execute(request)
        
        if reply.ret_code != 0:
            raise RuntimeError(f"Remote execution failed: {reply.stderr}")
        
        # 从 rpc_output.npz_data 读取 .npz 文件
        if reply.rpc_output.HasField("npz_data"):
            return self._load_npz_from_bytes(reply.rpc_output.npz_data)
        else:
            raise RuntimeError(f"Invalid reply: expected npz_data in rpc_output")
    
    def _compress_model(self, model_path: str):
        from .grpc import message_pb2
        
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
    
    def _load_npz_from_bytes(self, npz_bytes: bytes) -> Tuple[torch.Tensor, ...]:
        """从 bytes 加载 .npz 文件并转换为张量元组"""
        import numpy as np
        from io import BytesIO
        
        # 将 bytes 写入临时内存文件
        with BytesIO(npz_bytes) as f:
            npz = np.load(f, allow_pickle=True)
            
            result = []
            # 按字母顺序读取所有数组（保持一致性）
            for key in sorted(npz.files):
                arr = npz[key]
                result.append(torch.from_numpy(arr))
            
            npz.close()
        
        return tuple(result)
    
    def close(self):
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
