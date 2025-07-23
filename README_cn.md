# GraphNet

## 📌 项目简介
GraphNet —— 一个面向编译器开发的大规模数据集，旨在为研究者提供一个统一、开放的实验平台。其中包含大量来自真实模型的计算图，方便评估不同编译器Pass的优化效果。

通过 GraphNet，用户可以：

1. 快速测试不同编译器策略的通用优化效果
2. 方便已有编译器做回归测试
3. 训练AI-for-system模型以自动生成编译器优化Pass

## 快速开始

示例：对ResNet‑18进行计算图捕获和验证
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

### graph_net.torch.extract 使用方式

```python
import graph_net

# Instantiate the model (e.g. a torchvision model)
model = ...  

# Extract your own model
model = graph_net.torch.extract(name="model_name")(model)

# After running, the extracted graph will be saved to:
#   $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

### graph_net.torch.validate 使用方式
```
# Verify that the extracted model meets requirements
python -m graph_net.torch.validate \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

##  数据集约束

GraphNet数据集遵循以下约束规范：

1. 动态图能正常运行
2. 每份计算图有通用方法测定性能指标
3. 计算图与python代码之间序列化与反序列化
4. 整图可分解为不相交的两个子图
5. 可配置pass或编译器行为
6. 每份计算图中的op names可以被静态解析出来
7. 若存在自定义算子，则自定义算子的代码必须能被完整访问
8. 可通过统一方式配置计算图在不同芯片上运行

## 社区交流

* 扫描微信二维码或QQ二维码，即可加入交流群与众多社区开发者以及官方团队深度交流.

<div align="center">
<table>
<tr>
<td align="center">
    <img width="190" height="220" src="https://github.com/user-attachments/assets/12a4c2a1-0d3c-468f-9e6b-e141600fa6ff" />
</td>
<td align="center">
    <img width="190" height="220" src="https://github.com/user-attachments/assets/140fa03e-36ef-44bf-8d9a-ca65c83b0139" />
</td>
</tr>
</table>
</div>
