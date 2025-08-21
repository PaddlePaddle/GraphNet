# GraphNet  ![](https://img.shields.io/badge/version-v0.1-brightgreen) ![](https://img.shields.io/github/issues/PaddlePaddle/GraphNet?label=open%20issues)    [![](https://img.shields.io/badge/Contribute%20to%20GraphNet-blue)](https://github.com/PaddlePaddle/GraphNet/issues/98)


**GraphNet** 是一个面向AI 编译器开发的大规模**计算图数据集**，旨在为研究者提供统一、开放的实验平台。它收录了大量深度学习模型的计算图，方便评估不同编译器 Pass 的优化效果，支持跨框架、跨平台的性能比较。

**GraphNet** is a large-scale dataset of deep learning **computation graphs**, designed to serve as a standard benchmark and training corpus for **AI-driven tensor compiler optimization**. It contains diverse graphs extracted from state-of-the-art models, enabling effective evaluation of compiler pass optimizations across frameworks and hardware platforms.


通过 GraphNet，用户可以：

1. 快速测试不同编译器策略的通用优化效果
2. 方便已有编译器做回归测试
3. 训练AI-for-system模型以自动生成编译器优化Pass

With GraphNet, users can:
1. Quickly benchmark the optimization performance of various compiler strategies.
2. Easily conduct regression tests on existing compilers.
3. Train AI‑for‑Systems models to automatically generate compiler optimization passes.

**目标**：我们致力于实现编译优化策略在不同硬件间的可移植性，使大模型能够学习并迁移这些策略，从而大幅降低高效算子开发的成本。

**Vision**: We aim to achieve cross-hardware portability of compiler optimizations by allowing models to learn and transfer optimization strategies. It will significantly  reduce the manual effort required to develop efficient operator implementations.


### 数据集构建约束 / Dataset Construction Constraints

<table style="width:100%; border-collapse: collapse;">
  <tbody>
    <tr>
      <td style="padding: 8px; vertical-align: top;">
        <ol>
          <li>动态图能正常运行</li>
          <li>每份计算图有通用方法测定性能指标</li>
          <li>计算图与python代码之间序列化与反序列化</li>
          <li>整图可分解为不相交的两个子图</li>
          <li>可配置pass或编译器行为</li>
          <li>每份计算图中的op names可以被静态解析出来</li>
          <li>若存在自定义算子，则自定义算子的代码必须能被完整访问</li>
          <li>可通过统一方式配置计算图在不同芯片上运行</li>
        </ol>
      </td>
      <td style="padding: 8px; vertical-align: top;">
        <ol>
          <li>Dynamic graphs must execute correctly.</li>
          <li>Each computation graph should include a standardized method for measuring performance.</li>
          <li>Graphs and their corresponding Python code must support serialization and deserialization.</li>
          <li>The full graph can be decomposed into two disjoint subgraphs.</li>
          <li>Compiler passes or behaviors must be configurable.</li>
          <li>Operator names within each computation graph must be statically parseable.</li>
          <li>If custom operators are used, their implementation code must be fully accessible.</li>
          <li>Graph execution on different hardware backends must be configurable via a unified interface.</li>
        </ol>
      </td>
    </tr>
  </tbody>
</table>


## ⚡ 快速开始 / Quick Start
详细的实现细节请参见 [共创者指引](https://github.com/PaddlePaddle/GraphNet/blob/develop/CONTRIBUTE_TUTORIAL_cn.md#共创者指引)。

For full implementation details, please refer to the [Co-Creation Tutorial](https://github.com/PaddlePaddle/GraphNet/blob/develop/CONTRIBUTE_TUTORIAL.md#co-creation-tutorial).
### 测试编译器性能 / Benchmark your compiler on the model:

**graph_net.torch.test_compiler** 
```
python3 -m graph_net.torch.test_compiler \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name/ \
  --compiler /path/to/custom/compiler 
# Note: if --compiler is omitted, PyTorch’s built-in compiler is used by default
```

### 向 GraphNet 提交计算图 / Contribute computation graphs to GraphNet:

**示例：对ResNet‑18进行计算图捕获和验证 / Demo: Extract & Validate ResNet‑18**
```
git clone https://github.com/PaddlePaddle/GraphNet.git
cd GraphNet

# Set your workspace directory
export GRAPH_NET_EXTRACT_WORKSPACE=/home/yourname/graphnet_workspace

# Extract the ResNet‑18 computation graph
python graph_net/test/vision_model_test.py

# Validate the extracted graph (e.g. /home/yourname/graphnet_workspace/resnet18)
python -m graph_net.torch.validate \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/resnet18
```

**graph_net.torch.extract**

```python
import graph_net

# Instantiate the model (e.g. a torchvision model)
model = ...  

# Extract your own model
model = graph_net.torch.extract(name="model_name")(model)

# After running, the extracted graph will be saved to:
#   $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

**graph_net.torch.validate**
```
# Verify that the extracted model meets requirements
python -m graph_net.torch.validate \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

**graph_net.pack**
```
# Create a ZIP archive of $GRAPH_NET_EXTRACT_WORKSPACE.
# The --clear-after-pack flag (True|False) determines whether to delete the workspace after packing.
python -m graph_net.pack \
  --output /path/to/output.zip \
  --clear-after-pack True
```

注意： 要为 GraphNet 配置您的用户信息（用户名和电子邮件），请运行：

Note: To configure your user details (username and email) for GraphNet, run:
```
python -m graph_net.config --global \
  --username "your-name" \
  --email "your-email"
```
打包完这些计算图后，请通过以下群聊提交给 GraphNet 社区：

Once you have packaged these extracted computation graphs, submit them to the GraphNet community via the following group chats:


<div align="center">
<table>
<tr>
<td align="center">
    <img width="200" src="https://github.com/user-attachments/assets/f097977b-2101-4ddf-8be2-4cf08cb40ba5" />
</td>
<td align="center">
    <img width="150" src="https://cdn.prod.website-files.com/6257adef93867e50d84d30e2/67d00cf7266d2c75571aebde_Example.svg" />
    <p><a href="https://discord.gg/FCZQVCkC">Channel</a> is also available.</p>
</td>
</tr>
</table>
</div>


##  License
This project is released under the [MIT License](LICENSE).

