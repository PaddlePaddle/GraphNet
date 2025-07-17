# GraphNet

## 🧠 GraphNet：高性能编译器优化的基准数据集
我们提出了 GraphNet —— 一个面向高性能编译器优化的大规模基准数据集，旨在为研究者和开发者提供一个统一、开放、可扩展的实验平台。

## 📌 项目简介
GraphNet 包含大量来自真实高性能计算任务的图结构表示，可用于评估编译器Pass的优化效果、高性能优化方案。

通过 GraphNet，用户可以：

快速测试不同优化策略的效果；
训练模型以自动生成编译器优化Pass；
降低高性能算法测评的门槛。


## 生成单侧

```
python vision_model_generator.py --key "model-name like resnet18"  --model-path  "path to be restored at"
```

## 运行单侧
```
python runner.py  --model-path ../../sample/torch/resnet18
```

