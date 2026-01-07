import grpc
import tarfile
import json
from pathlib import Path
from io import BytesIO
from typing import Tuple, Optional
from contextlib import contextmanager
import torch


class SampleRemoteExecutor:
    """远程模型执行器"""
    
    def __init__(self, machine: str, port: int):
        self.machine = machine
        self.port = port
        self._channel: Optional[grpc.Channel] = None
        self._stub = None
    
    def _get_stub(self):
        if self._stub is None:
            from .grpc import message_pb2, message_pb2_grpc
            self._channel = grpc.insecure_channel(f"{self.machine}:{self.port}")
            self._stub = message_pb2_grpc.SampleRemoteExecutorStub(self._channel)
        return self._stub
    
    def __call__(self, model_path: str, random_seed: int) -> Tuple[torch.Tensor, ...]:
        """远程执行模型"""
        from .grpc import message_pb2
        
        compressed_data = self._compress_model(model_path)
        
        # 2. 构建请求
        stub = self._get_stub()
        request = message_pb2.ExecutionRequest(
            rpc_cmd="execute_model",
            rpc_input=message_pb2.RpcData(compressed_data=compressed_data),
            output_file_name=str(random_seed)
        )
        
        reply = stub.Execute(request)
        
        if reply.ret_code != 0:
            raise RuntimeError(f"Remote execution failed: {reply.stderr}")
        
        return self._deserialize_tensors(reply.stdout)
    
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
    
    def _deserialize_tensors(self, json_str: str) -> Tuple[torch.Tensor, ...]:
        import numpy as np
        
        data = json.loads(json_str)
        result = []
        
        for tensor_data in data:
            dtype = getattr(torch, tensor_data["dtype"])
            shape = tuple(tensor_data["shape"])
            
            if tensor_data["data"] is None:
                np_array = np.zeros(shape, dtype=dtype.__name__)
            else:
                np_array = np.frombuffer(
                    tensor_data["data"].encode("latin1"),
                    dtype=np.dtype(dtype.__name__.replace("torch.", ""))
                )
                np_array = np_array.reshape(shape)
            
            result.append(torch.from_numpy(np_array))
        
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
