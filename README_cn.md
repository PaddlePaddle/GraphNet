# GraphNet

GraphNet —— 一个面向编译器开发的大规模数据集，旨在为研究者提供一个统一、开放的实验平台。其中包含大量来自真实模型的计算图，方便评估不同编译器Pass的优化效果。

通过 GraphNet，用户可以：

1. 快速测试不同编译器策略的通用优化效果
2. 方便已有编译器做回归测试
3. 训练AI-for-system模型以自动生成编译器优化Pass

数据集构建约束：

1. 动态图能正常运行
2. 每份计算图有通用方法测定性能指标
3. 计算图与python代码之间序列化与反序列化
4. 整图可分解为不相交的两个子图
5. 可配置pass或编译器行为
6. 每份计算图中的op names可以被静态解析出来
7. 若存在自定义算子，则自定义算子的代码必须能被完整访问
8. 可通过统一方式配置计算图在不同芯片上运行

## 快速开始
### 测试编译器性能
**graph_net.torch.test_compiler** 
```
python3 -m graph_net.torch.test_compiler \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name/ \
  --compiler /path/to/custom/compiler 
# Note: if --compiler is omitted, PyTorch’s built-in compiler is used by default
```

### 向 GraphNet 提交计算图
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

**graph_net.torch.extract 使用方式**

```python
import graph_net

# Instantiate the model (e.g. a torchvision model)
model = ...  

# Extract your own model
model = graph_net.torch.extract(name="model_name")(model)

# After running, the extracted graph will be saved to:
#   $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

**graph_net.torch.validate 使用方式**
```
# Verify that the extracted model meets requirements
python -m graph_net.torch.validate \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

**graph_net.pack 使用方式**
```
# Create a ZIP archive of $GRAPH_NET_EXTRACT_WORKSPACE.
# The --clear-after-pack flag (True|False) determines whether to delete the workspace after packing.
python -m graph_net.pack \
  --output /path/to/output.zip \
  --clear-after-pack True
```

注意： 要为 GraphNet 配置您的用户信息（用户名和电子邮件），请运行：
```
python -m graph_net.config --global\
  --username "your-name" \
  --email "your-email"
```

打包完这些计算图后，请通过以下群聊提交给 GraphNet 社区

<div align="center">
<table>
<tr>
<td align="center">
    <img width="190" height="220" src="https://github.com/user-attachments/assets/31b4f0ba-417e-48b6-a860-124d74bd6643" />
</td>
<td align="center">
    <img width="190" height="220" src="https://github.com/user-attachments/assets/140fa03e-36ef-44bf-8d9a-ca65c83b0139" />
</td>
</tr>
</table>
</div>

## 开源协议
[MIT License](LICENSE)