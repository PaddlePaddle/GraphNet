
<h1 align="center">GraphNet：面向张量编译器研究的大规模计算图数据集</h1>

<div align="center">

![](https://img.shields.io/github/issues/PaddlePaddle/GraphNet?label=open%20issues)
[![arXiv](https://img.shields.io/badge/arXiv-2510.24035-b31b1b.svg)](https://arxiv.org/abs/2510.24035)
<a href="https://github.com/user-attachments/assets/125e3494-25c9-4494-9acd-8ad65ca85d03"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

**GraphNet** 是一个大规模深度学习**计算图数据集**，旨在为**张量编译器**优化提供一个标准的基准测试平台。它包含了从覆盖多种任务和机器学习框架的先进深度学习模型中提取的超过 2700个 计算图。凭借其标准化的格式和丰富的元数据，GraphNet 能够对张量编译器的通用优化能力进行公平比较和可复现的评估，从而支持诸如面向编译器的“AI for System”等前沿研究。

## 📣 最新动态
- [2025-11-19] ✨ 在 GTOC Forum 2025 上的主题演讲：[GraphNet 助力 AI 软件栈催熟](https://b23.tv/PFzSKK1)
- [2025-10-14] ✨ 我们的技术报告已发布：这是一份关于数据集构建和编译器基准测试的详细研究，并引入了新颖的性能指标——加速分数 S(t) 和感知错误的加速分数 ES(t)。[📘 GraphNet: A Large-Scale Computational Graph Dataset for Tensor Compiler Research](https://arxiv.org/abs/2510.24035)
- [2025-8-20] 🚀 第二轮 [开源贡献任务](https://github.com/PaddlePaddle/Paddle/issues/74773) 已发布。（已完成 ✅）
- [2025-7-30] 🚀 第一轮 [开源贡献任务](https://github.com/PaddlePaddle/GraphNet/issues/44) 已发布。（已完成 ✅）
## 📊 基准测试结果
我们在 GraphNet 的 NLP 和 CV 子集上评估了两个代表性的张量编译器后端：CINN (PaddlePaddle) 和 TorchInductor (PyTorch)。评估采用了[技术报告](https://arxiv.org/abs/2510.24035)中提出的两个量化指标：
- **加速分数** S(t) — 评估编译器在不同数值容忍度下的性能。
<div align="center">
  <img src="/pics/St-result.jpg" alt="Speedup Score S_t Results" width="80%">
</div>

- **感知错误的加速分数** ES(t) — 进一步考量运行时和编译错误。
<div align="center">
  <img src="/pics/ESt-result.jpg" alt="Error-aware Speedup Score ES_t Results" width="80%">

</div>

## ⚡ 快速开始
本节面向编译器用户/开发者展示如何评估张量编译器并复现基准测试结果，以及面向 GraphNet 贡献者展示如何贡献新的计算图。

### ⚖️ 编译器评估

**步骤 1：基准测试**

使用 `graph_net.torch.test_compiler` 对 GraphNet 样本进行基准测试，可指定批次和日志配置：

```bash
# 设置你的基准测试目录
export GRAPH_NET_BENCHMARK_PATH=/home/yourname/graphnet_benchmark/

# 运行基准测试
python -m graph_net.torch.test_compiler \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name/ \
  --compiler /custom/or/builtin/compiler/ \
  --device /device/to/execute/ \
  --warmup /times/to/warmup/ \
  --trials /times/to/test/ \
  > $GRAPH_NET_BENCHMARK_PATH/log.log 2>&1

# 注意：如果省略 --compiler 参数，默认使用 PyTorch 的内置编译器。
```

执行后，`graph_net.torch.test_compiler` 将：
1. 以即时执行模式运行原始模型，记录基线性能。
2. 使用指定的后端（例如 CINN, TVM, Inductor, TensorRT, XLA, BladeDISC）编译模型。
3. 执行编译后的模型，收集其运行时间和输出。
4. 若无执行失败，则将编译结果与基线对比，计算加速比。

**步骤 2：分析**

使用 `graph_net.plot_St`、`graph_net.plot_ESt` 和 `graph_net.plot_violin` 这三个脚本，根据基准测试日志中的加速比、正确性和运行时信息，生成 St 图、ESt 图和 [小提琴图](https://en.m.wikipedia.org/wiki/Violin_plot)。

```bash
python -m graph_net.plot_St \
  --benchmark-path $GRAPH_NET_BENCHMARK_PATH/log.log \
  --output-dir $GRAPH_NET_BENCHMARK_PATH \
  --negative-speedup-penalty penalty/power/for/negative/speedup \
  --fpdb base/penalty/for/severe/errors

python -m graph_net.plot_ESt \
  --benchmark-path $GRAPH_NET_BENCHMARK_PATH/log.log \
  --output-dir $GRAPH_NET_BENCHMARK_PATH \
  --negative-speedup-penalty penalty/power/for/negative/speedup \
  --fpdb base/penalty/for/severe/errors

# 注意：如果省略 --negative-speedup-penalty 参数，默认使用 p=0。
# 如果省略 --fpdb 参数，默认使用 b=0.1。

python -m graph_net.plot_violin \
  --benchmark-path $GRAPH_NET_BENCHMARK_PATH/JSON_results/ \
  --output-dir $GRAPH_NET_BENCHMARK_PATH
```

这些脚本设计用于处理 `/benchmark_path/category_name/` 这样的文件结构，x 轴上的项目由子目录名称标识。执行后，按类别（模型任务、库等）划分的结果汇总图表将被导出到 `$GRAPH_NET_BENCHMARK_PATH`。

### 硬件回归测试
我们还提供了一个两步工作流，用于根据“黄金标准”参考验证编译器的正确性和性能，这对于硬件专用测试和回归跟踪至关重要。详情可参阅 [指南](./docs/hardware_test_cn.md)。

### 🧱 构建与贡献指南
想了解 GraphNet 如何构建或贡献新样本？查看 [构建指南](./docs/README_contribute_cn.md) 以获取有关提取和验证工作流的详细信息。

## 🚀 未来路线图

1. 将 GraphNet 扩展至 10,000+ 计算图。
2. 为 GraphNet 样本添加更精细的子类别注释。
3. 从多 GPU 场景中提取样本，以支持大规模分布式计算的基准测试和优化。
4. 支持将完整计算图拆分为可独立优化的子图和算子序列。

**愿景**: GraphNet 旨在通过对张量编译器优化进行**大规模、系统性**的评估，并**为模型学习和迁移优化策略提供数据集**，从而为“面向编译器的 AI (AI for Compiler)”奠定基础。

## GraphNet 社区

您可以通过扫描下方群聊二维码加入我们的社区。欢迎提出任何关于使用和构建 GraphNet 的问题。

<div align="center">
<table>
<tr>
<td align="center">
    <img width="200" src="https://github.com/user-attachments/assets/125e3494-25c9-4494-9acd-8ad65ca85d03" />
</td>
<td align="center">
    <img width="150" src="https://cdn.prod.website-files.com/6257adef93867e50d84d30e2/67d00cf7266d2c75571aebde_Example.svg" />
    <p><a href="https://discord.gg/vyeAydwh">Channel</a> is also available.</p>
</td>
</tr>
</table>
</div>

## 许可证与致谢

GraphNet 基于 [MIT 许可证](./LICENSE) 开源发布。

如果您觉得本项目对您的研究或工作有帮助，请引用：

```bibtex
@misc{li2025graphnetlargescalecomputationalgraph,
      title={GraphNet: A Large-Scale Computational Graph Dataset for Tensor Compiler Research}, 
      author={Xinqi Li and Yiqun Liu and Shan Jiang and Enrong Zheng and Huaijin Zheng and Wenhao Dai and Haodong Deng and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2510.24035},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.24035}, 
}
```
