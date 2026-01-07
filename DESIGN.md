# GraphNet 远程模型测试框架 - 设计文档 v4

## 一、设计原则

**核心思想**：服务端是**通用的模型执行引擎**，不依赖 GraphNet 代码。客户端负责调用 GraphNet 测试框架，通过 RPC 远程执行模型。

**重要约束**：严格基于现有的 `message.proto`，不修改协议定义。

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Client                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              GraphNet Test Framework                  │  │
│  │  ┌────────────────────────────────────────────┐    │ │ 
│  │  │  test_compiler.py                        │    │ │
│  │  │  - 性能测试 (warmup + trials)            │    │ │
│  │  │  - 正确性验证                            │    │ │
│  │  └────────────────────────────────────────────┘    │ │
│  └───────────────────┬──────────────────────────────────┘ │
│                      │                                   │
│  ┌───────────────────▼──────────────────────────────────┐ │
│  │         SampleRemoteExecutor (graph_net_bench)        │ │
│  │  - 打包模型目录 (tar.gz)                              │ │
│  │  - 通过 RPC 执行模型                                  │ │
│  │  - 返回 tuple[Tensor] 结果                           │ │
│  └───────────────────┬──────────────────────────────────┘│
│                      │ gRPC (message.proto)              │
└──────────────────────┼───────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Server Machine (无 GraphNet 依赖)          │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                gRPC Server (基于现有 proto)          │ │
│  │  - 接收 ExecutionRequest                             │ │
│  │  - 解压并执行模型                                    │ │
│  │  - 返回 ExecutionReply                              │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、现有协议 (message.proto)

```protobuf
syntax = "proto3";

package sample_remote_executor;

message CompressedData {
    string filename = 1;
    uint32 original_size = 2;
    bytes payload = 3;
    string compression_algo = 4;
}

message RpcData {
    oneof rpc_data_type {
        CompressedData compressed_data = 1;
        string str_data = 2;
    }
}

message ExecutionRequest {
    string rpc_cmd = 1;
    RpcData rpc_input = 2;
    optional string output_file_name = 3;
}

message ExecutionReply {
    int64 ret_code = 1;
    string stdout = 2;
    string stderr = 3;
    RpcData rpc_output = 4;
}

service SampleRemoteExecutor {
  rpc Execute (ExecutionRequest) returns (ExecutionReply);
}
```

### 协议使用约定

| 字段 | 用途 |
|------|------|
| `rpc_cmd` | 命令标识："execute_model" |
| `rpc_input.compressed_data` | 压缩的模型目录 (tar.gz) |
| `output_file_name` | random_seed (字符串形式) |
| `stdout` | 序列化的输出张量列表 (JSON) |
| `ret_code` | 0=成功, 非0=失败 |

---

## 四、客户端实现

### 4.1 SampleRemoteExecutor 类

**文件**: `graph_net/graph_net_bench/sample_remote_executor.py`

**状态**: ✅ 已实现

```python
"""
SampleRemoteExecutor: 远程模型执行器

使用方式:
    import graph_net_bench as gnb

    # 创建执行器
    sample_remote_executor = gnb.SampleRemoteExecutor(machine="192.168.1.100", port=50052)

    # 直接调用，返回 tuple[Tensor, ...]
    ret: tuple[torch.Tensor, ...] = sample_remote_executor(sample_model_path, random_seed=42)

    # 支持上下文管理器
    with sample_remote_executor:
        outputs = sample_remote_executor(model_path, random_seed=1024)
"""

import grpc
import tarfile
import json
from pathlib import Path
from io import BytesIO
from typing import Tuple, Optional
from contextlib import contextmanager

import torch


class SampleRemoteExecutor:
    """远程模型执行器

    通过 gRPC 在远程服务器上执行模型推理。

    Attributes:
        machine: 服务器 IP 地址
        port: 服务器端口
        channel: gRPC 通道
        stub: gRPC 存根
    """

    def __init__(self, machine: str, port: int):
        """
        Args:
            machine: 服务器 IP 地址
            port: 服务器端口
        """
        self.machine = machine
        self.port = port
        self._channel: Optional[grpc.Channel] = None
        self._stub = None

    def _get_stub(self):
        """获取 gRPC 存根（延迟初始化）"""
        if self._stub is None:
            from .grpc import message_pb2, message_pb2_grpc
            self._channel = grpc.insecure_channel(f"{self.machine}:{self.port}")
            self._stub = message_pb2_grpc.SampleRemoteExecutorStub(self._channel)
        return self._stub

    @contextmanager
    def __call__(self, model_path: str, random_seed: int) -> Tuple[torch.Tensor, ...]:
        """
        远程执行模型

        Args:
            model_path: 模型目录路径
            random_seed: 随机种子，用于生成可复现的输入

        Returns:
            tuple[Tensor, ...]: 模型输出张量

        Raises:
            RuntimeError: 远程执行失败
        """
        # 1. 压缩模型目录
        compressed_data = self._compress_model(model_path)

        # 2. 构造 RPC 请求
        from .grpc import message_pb2

        stub = self._get_stub()
        request = message_pb2.ExecutionRequest(
            rpc_cmd="execute_model",
            rpc_input=message_pb2.RpcData(
                compressed_data=compressed_data
            ),
            # 使用 output_file_name 字段传递 random_seed
            output_file_name=str(random_seed)
        )

        # 3. 发送请求
        reply = stub.Execute(request)

        # 4. 解析结果
        if reply.ret_code != 0:
            raise RuntimeError(f"Remote execution failed: {reply.stderr}")

        # stdout 包含序列化的张量数据 (JSON 格式)
        return self._deserialize_tensors(reply.stdout)

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """清理资源"""
        self.close()

    def close(self):
        """关闭 gRPC 通道"""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def _compress_model(self, model_path: str):
        """压缩模型目录为 CompressedData

        Args:
            model_path: 模型目录路径

        Returns:
            CompressedData: 压缩的模型数据
        """
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
        """从 JSON 反序列化张量列表

        Args:
            json_str: JSON 格式的张量数据

        Returns:
            tuple[Tensor, ...]: 张量元组
        """
        import numpy as np

        data = json.loads(json_str)
        result = []

        for tensor_data in data:
            dtype = getattr(torch, tensor_data["dtype"])
            shape = tuple(tensor_data["shape"])
            # 处理不同数据类型的序列化
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
```

### 4.2 使用示例

```python
# 方式一：直接调用
import graph_net_bench as gnb

# 创建执行器
executor = gnb.SampleRemoteExecutor(machine="192.168.1.100", port=50052)

# 直接调用，返回 tuple[Tensor, ...]
ret: tuple[torch.Tensor, ...] = executor("/path/to/model", random_seed=42)

# 方式二：使用上下文管理器（推荐，自动关闭连接）
with gnb.SampleRemoteExecutor(machine="192.168.1.100", port=50052) as executor:
    outputs = executor("/path/to/model", random_seed=1024)

# 方式三：在 GraphNet 测试框架中使用
def test_single_model_remote(args):
    import graph_net_bench as gnb

    executor = gnb.SampleRemoteExecutor(
        machine=args.remote_machine,
        port=args.remote_port
    )

    with executor:
        # 远程执行模型
        eager_out = executor(args.model_path, random_seed=1024)

    # 本地进行正确性对比
    compare_correctness(eager_out, compiled_out, args)
```

---

## 五、服务端实现 (TODO)

### 5.1 remote_model_server.py

**文件**: `graph_net/graph_net_bench/server/remote_model_server.py`

**状态**: ⏳ 待实现

```python
import grpc
from concurrent import futures
import tempfile
import shutil
import tarfile
import json
import torch
from pathlib import Path
from io import BytesIO

import message_pb2
import message_pb2_grpc


class RemoteModelExecutorServicer(message_pb2_grpc.SampleRemoteExecutorServicer):
    """远程模型执行服务"""

    def Execute(self, request, context):
        """
        执行模型推理

        Args:
            request: ExecutionRequest
                - rpc_cmd: "execute_model"
                - rpc_input.compressed_data: 压缩的模型
                - output_file_name: random_seed (字符串)

        Returns:
            ExecutionReply
                - ret_code: 0=成功
                - stdout: 序列化的输出张量 (JSON)
                - stderr: 错误信息
        """
        temp_dir = None
        try:
            # 1. 解析参数
            if request.rpc_cmd != "execute_model":
                return message_pb2.ExecutionReply(
                    ret_code=-1,
                    stderr=f"Unknown rpc_cmd: {request.rpc_cmd}"
                )

            random_seed = int(request.output_file_name)

            # 2. 解压模型到临时目录
            temp_dir = tempfile.mkdtemp(prefix="remote_model_")
            model_path = self._decompress_model(
                request.rpc_input.compressed_data,
                temp_dir
            )

            # 3. 加载模型和权重
            model = self._load_model(model_path)

            # 4. 设置随机种子
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

            # 5. 执行推理
            model.eval()
            with torch.no_grad():
                outputs = model()

            # 6. 序列化输出
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
        """加载模型

        模型目录结构:
        - model.py: 定义 GraphModule 类
        - weight_meta.py: 权重元数据 (包含权重数据或生成参数)
        """
        import sys
        import importlib.util

        model_file = Path(model_path) / "model.py"
        spec = importlib.util.spec_from_file_location(
            "remote_model",
            str(model_file)
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["remote_model"] = module
        spec.loader.exec_module(module)

        # 动态创建权重并加载到模型
        weight_tensors = self._create_weight_tensors(model_path, module.GraphModule)

        model = module.GraphModule()

        # 加载权重
        # GraphModule 的参数名与 weight_meta 中的 name 匹配
        for name, tensor in weight_tensors.items():
            param = getattr(model, name, None)
            if param is not None:
                param.data.copy_(tensor)

        return model

    def _create_weight_tensors(self, model_path, graph_module_class):
        """根据 weight_meta.py 创建权重张量

        从 weight_meta.py 中读取元数据，动态生成张量数据。
        如果元数据中有 data，则使用实际数据；否则生成随机张量。
        """
        import importlib.util
        import numpy as np

        weight_meta_file = Path(model_path) / "weight_meta.py"
        if not weight_meta_file.exists():
            return {}

        # 导入 weight_meta 模块
        spec = importlib.util.spec_from_file_location(
            "weight_meta",
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
            device = tensor_cls.device if hasattr(tensor_cls, 'device') else 'cpu'

            if tensor_cls.data is not None:
                # 使用实际数据
                np_array = np.array(tensor_cls.data, dtype=np.dtype(dtype.__name__))
                np_array = np_array.reshape(shape)
                weight_tensors[name] = torch.from_numpy(np_array).to(device)
            else:
                # 生成随机张量
                weight_tensors[name] = torch.randn(shape, dtype=dtype, device=device)

        return weight_tensors

    def _serialize_tensors(self, outputs):
        """序列化张量列表为 JSON"""
        tensor_list = []

        for tensor in outputs:
            tensor_data = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "data": tensor.numpy().tobytes().decode("latin1")
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
    serve()
```

---

## 六、目录结构

```
graph_net/graph_net_bench/
├── grpc/
│   ├── message.proto               # 保持不变 (协议定义)
│   ├── message_pb2.py              # protobuf 自动生成
│   ├── message_pb2_grpc.py         # gRPC 存根自动生成
│   ├── server.py                   # 保留 (简单 echo 测试)
│   └── client.py                   # 保留 (简单测试)
├── server/
│   ├── __init__.py                 # 新增: 包初始化
│   └── remote_model_server.py      # ⏳ 远程模型服务 (待实现)
├── sample_remote_executor.py       # ✅ 客户端实现 (已完成)
├── __init__.py                     # 新增: 包初始化，导出 SampleRemoteExecutor
└── DESIGN.md                       # 本设计文档
```

---

## 七、关键实现细节

### 7.1 数据序列化方案

| 数据 | 序列化方式 | 协议字段 |
|------|------------|----------|
| 模型目录 | tar.gz → bytes | `rpc_input.compressed_data` |
| random_seed | 字符串 → `str()` | `output_file_name` |
| 输出张量 | JSON → stdout | `stdout` |

### 7.2 模型目录结构

GraphNet 的 sample 目录结构:
```
model_path/
├── model.py              # 定义 GraphModule 类
├── weight_meta.py        # 权重元数据 (包含形状、数据)
└── input_tensor_constraints.py  # 可选: 输入约束
```

**weight_meta.py 示例**:
```python
class Program_weight_tensor_meta_L_self_modules_classifier_parameters_bias_:
    name = "L_self_modules_classifier_parameters_bias_"
    shape = [2]
    dtype = "torch.float32"
    device = "cuda:0"
    data = [0.0, 0.0]  # 可为 None (表示随机)
```

### 7.3 限制与约束

| 约束 | 说明 |
|------|------|
| 模型大小 | 受 gRPC 消息大小限制 (默认 4MB) |
| 张量大小 | JSON 序列化效率较低，大张量需优化 |
| 并发 | 服务端 max_workers 控制并发数 |
| 依赖 | 服务端需要 torch, numpy, protobuf |

### 7.4 未来优化

| 优化项 | 方案 |
|--------|------|
| 大模型传输 | 分块传输，使用 RpcData 流式处理 |
| 张量序列化 | 使用 protobuf bytes 替代 JSON |
| 结果缓存 | 缓存模型加载结果 |
| GPU 分配 | 支持指定设备 (cuda:0, cpu 等) |

---

## 八、命令行使用

### 服务端 (远程机器)

```bash
# 安装依赖
pip install torch numpy grpcio grpcio-tools

# 启动远程模型服务器
cd /denghaodong/code/GraphNet/graph_net/graph_net_bench/server
python remote_model_server.py --port 50052
```

### 客户端 (本地机器)

```bash
# 安装依赖
pip install torch numpy grpcio

# 使用示例
python -c "
import graph_net_bench as gnb

# 创建执行器
executor = gnb.SampleRemoteExecutor(machine='192.168.1.100', port=50052)

# 远程执行模型
outputs = executor('/path/to/sample/model', random_seed=42)
print(f'Received {len(outputs)} output tensors')

# 关闭连接
executor.close()
"
```

---

## 九、与 GraphNet 集成示例

### 9.1 独立使用

```python
import graph_net_bench as gnb
from pathlib import Path

# 配置
SERVER_IP = "192.168.1.100"
SERVER_PORT = 50052
MODEL_PATH = "/path/to/transformers-auto-model/model_name"

# 创建执行器
executor = gnb.SampleRemoteExecutor(machine=SERVER_IP, port=SERVER_PORT)

try:
    # 执行远程推理
    outputs = executor(MODEL_PATH, random_seed=42)
    print(f"Success: {len(outputs)} outputs")
finally:
    executor.close()
```

### 9.2 在 test_compiler.py 中集成

```python
# graph_net/torch/test_compiler.py

def test_single_model_remote(args):
    """使用远程服务器测试单个模型"""
    import graph_net_bench as gnb

    executor = gnb.SampleRemoteExecutor(
        machine=args.remote_machine,
        port=args.remote_port
    )

    with executor:
        # 1. 远程执行 eager 模式 (基线)
        eager_out = executor(args.model_path, random_seed=args.seed)

        # 2. 编译模型 (本地或远程)
        compiled_out = compile_and_execute(args)

        # 3. 本地进行正确性对比
        compare_correctness(eager_out, compiled_out, args)

    return True
```

---

## 十、故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 连接超时 | 网络不通/端口未开放 | 检查防火墙和服务器状态 |
| 内存不足 | 模型过大 | 使用更小的 batch size |
| 序列化失败 | 张量包含复杂类型 | 简化输出或使用分块传输 |
| 权重加载失败 | 参数名不匹配 | 检查 model.py 和 weight_meta.py |

---

## 十一、参考资源

- [gRPC Python 文档](https://grpc.io/docs/languages/python/)
- [GraphNet 项目](https://github.com/PaddlePaddle/GraphNet)
- [PyTorch TorchScript](https://pytorch.org/docs/stable/jit.html)