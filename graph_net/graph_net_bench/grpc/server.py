import grpc
from concurrent import futures
import tempfile
import shutil
import tarfile
import json
import torch
import numpy as np
from pathlib import Path
from io import BytesIO

import message_pb2
import message_pb2_grpc


class RemoteModelExecutorServicer(message_pb2_grpc.SampleRemoteExecutorServicer):
    """远程模型执行服务"""

    def Execute(self, request, context):
        """执行模型推理"""
        temp_dir = None

        try:
            # 1. 验证命令
            if request.rpc_cmd != "execute_model":
                return message_pb2.ExecutionReply(
                    ret_code=-1,
                    stderr=f"Unknown rpc_cmd: {request.rpc_cmd}"
                )

            # 2. 获取 random_seed
            random_seed = int(request.output_file_name)

            # 3. 解压模型
            temp_dir = tempfile.mkdtemp(prefix="remote_model_")
            model_path = self._decompress_model(
                request.rpc_input.compressed_data,
                temp_dir
            )

            # 4. 加载模型
            model = self._load_model(model_path)

            # 5. 设置随机种子
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

            # 6. 执行推理
            model.eval()
            with torch.no_grad():
                outputs = model()

            # 7. 序列化输出
            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            json_output = self._serialize_tensors(outputs)

            return message_pb2.ExecutionReply(
                ret_code=0,
                stdout=json_output,
                stderr=""
            )

        except Exception as e:
            import traceback
            return message_pb2.ExecutionReply(
                ret_code=-1,
                stderr=f"{str(e)}\n{traceback.format_exc()}"
            )

        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _decompress_model(self, compressed_data, temp_dir):
        """解压模型目录"""
        buffer = BytesIO(compressed_data.payload)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(path=temp_dir)
        return temp_dir

    def _load_model(self, model_path):
        """加载模型"""
        import importlib.util

        model_file = Path(model_path) / "model.py"
        if not model_file.exists():
            raise FileNotFoundError(f"model.py not found in {model_path}")

        spec = importlib.util.spec_from_file_location(
            "remote_model_module",
            str(model_file)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'GraphModule'):
            raise ValueError("model.py must define 'GraphModule' class")

        # 创建权重
        weight_tensors = self._create_weight_tensors(model_path)
        model = module.GraphModule()

        # 加载权重
        for name, tensor in weight_tensors.items():
            param = getattr(model, name, None)
            if param is not None and isinstance(param, torch.Tensor):
                param.data.copy_(tensor)

        return model

    def _create_weight_tensors(self, model_path):
        """根据 weight_meta.py 创建权重张量"""
        import importlib.util

        weight_meta_file = Path(model_path) / "weight_meta.py"
        if not weight_meta_file.exists():
            return {}

        spec = importlib.util.spec_from_file_location(
            "weight_meta_module",
            str(weight_meta_file)
        )
        weight_meta_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(weight_meta_module)

        weight_tensors = {}
        tensor_classes = [
            getattr(weight_meta_module, name)
            for name in dir(weight_meta_module)
            if name.startswith("Program_weight_tensor_meta_")
        ]

        for tensor_cls in tensor_classes:
            name = tensor_cls.name
            shape = tensor_cls.shape
            dtype = getattr(torch, tensor_cls.dtype)
            device = getattr(tensor_cls, 'device', 'cpu')

            if tensor_cls.data is not None:
                np_array = np.array(tensor_cls.data, dtype=np.dtype(dtype.__name__))
                np_array = np_array.reshape(shape)
                weight_tensors[name] = torch.from_numpy(np_array).to(device)
            else:
                weight_tensors[name] = torch.randn(shape, dtype=dtype, device=device)

        return weight_tensors

    def _serialize_tensors(self, outputs):
        """序列化张量为 JSON"""
        tensor_list = []
        for tensor in outputs:
            tensor_data = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "data": tensor.cpu().numpy().tobytes().decode("latin1")
            }
            tensor_list.append(tensor_data)
        return json.dumps(tensor_list)


def serve(port=50052, max_workers=4):
    """启动 gRPC 服务器"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
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
