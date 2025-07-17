# GraphNet

## 📌 项目简介
GraphNet —— 一个面向编译器开发的大规模数据集，旨在为研究者提供一个统一、开放的实验平台。其中包含大量来自真实模型的计算图，方便评估不同编译器Pass的优化效果。

通过 GraphNet，用户可以：

1. 快速测试不同编译器策略的通用优化效果
2. 训练AI-for-system模型以自动生成编译器优化Pass


## 计算图抽取Demo
### torch 
```
export PYTHONPATH=$PYTHONPATH:/path/to/your/GraphNet/repo
python3 -m graph_net.torch.extractor.vision_model_extractor --key resnet18  --model-path  /path/to/your/extracted/graph_net/sample
```

## 计算图运行Demo
### torch
```
export PYTHONPATH=$PYTHONPATH:/path/to/your/GraphNet/repo
python3 -m graph_net.torch.runner.single_device_runner --model-path /path/to/your/extracted/graph_net/sample
```

