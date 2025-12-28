# 为 GraphNet 做贡献
为确保数据集整体质量、可复现性及跨编译器兼容性，我们定义了以下构建**约束条件**：

1. 可运行：计算图必须在命令式（即时）模式下可执行。
2. 可序列化：计算图及其对应的 Python 代码必须支持序列化与反序列化。
3. 可分解：完整计算图应能分解为两个不相交的子图。
4. 可静态分析：每个计算图内的算子名称必须可静态解析。
5. 自定义算子可访问：若使用了自定义算子，其实现代码必须完全可访问。

## 图提取与验证
GraphNet 提供了用于图提取和验证的自动化工具。

<div align="center">
<img src="/pics/graphnet_overview.jpg" alt="GraphNet Architecture Overview" width="65%">
</div>

**示例：提取并验证 ResNet‑18**
```bash
git clone https://github.com/PaddlePaddle/GraphNet.git
cd GraphNet

# 设置您的工作空间目录
export GRAPH_NET_EXTRACT_WORKSPACE=/home/yourname/graphnet_workspace/

# 提取 ResNet-18 计算图
python graph_net/test/vision_model_test.py

# 验证提取的图（例如 /home/yourname/graphnet_workspace/resnet18/）
python -m graph_net.torch.validate \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/resnet18/
```

**工作流程图示说明**

<div align="center">
<img src="/pics/dataset_composition.png" alt="GraphNet Extract Sample" width="65%">
</div>

> 注：**仅当**模块中使用了相应的自定义算子时，才需要提供 custom_op 的源代码，且**对其格式无特定要求**。

**步骤 1：`graph_net.torch.extract`**

使用提取器包装您的模型——这就是您需要做的全部：

```bash
import graph_net

# 实例化模型（例如一个 torchvision 模型）
model = ...  

# 提取您自己的模型
model = graph_net.torch.extract(name="model_name", dynamic=True)(model)
```

运行后，提取的计算图将保存至：`$GRAPH_NET_EXTRACT_WORKSPACE/model_name/`。

更多详细信息，请参阅 `graph_net/torch/extractor.py` 中定义的 `graph_net.torch.extract` 的文档字符串。

**步骤 2：`graph_net.torch.validate`**

为验证提取的模型是否符合要求，我们在 CI 工具中使用 `graph_net.torch.validate`，同时也要求贡献者提前进行自检：

```bash
python -m graph_net.torch.validate \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

所有**构建约束条件**都将被自动检查。通过验证后，将生成一个唯一的 `graph_hash.txt` 文件，后续在 CI 流程中会检查此文件以避免重复。

## 📁 代码仓库结构
本仓库结构如下：

| 目录 | 描述 |
|------------|--------------|
| **graph_net/** | 图提取、验证与基准测试的核心模块 |
| **paddle_samples/** | 从 PaddlePaddle 提取的计算图样本 |
| **samples/** | 从 PyTorch 提取的计算图样本 |
| **docs/** | 技术文档与贡献者指南|

以下是 **graph_net/** 目录的结构：
```text
graph_net/
 ├─ config/    # 配置文件、参数
 ├─ paddle/    # PaddlePaddle 图提取与验证
 ├─ torch/     # PyTorch 图提取与验证
 ├─ test/      # 单元测试与示例脚本
 └─ *.py       # 基准测试与分析脚本
