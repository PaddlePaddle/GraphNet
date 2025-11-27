
<h1 align="center">GraphNet: A Large-Scale Computational Graph Dataset for Tensor Compiler Research</h1>

<div align="center">

![](https://img.shields.io/github/issues/PaddlePaddle/GraphNet?label=open%20issues)
[![arXiv](https://img.shields.io/badge/arXiv-2510.24035-b31b1b.svg)](https://arxiv.org/abs/2510.24035)
<a href="https://github.com/user-attachments/assets/125e3494-25c9-4494-9acd-8ad65ca85d03"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

**GraphNet** is a large-scale dataset of deep learning **computation graphs**, built as a standard benchmark for **tensor compiler** optimization. It provides over 2.7K computation graphs extracted from state-of-the-art deep learning models spanning diverse tasks and ML frameworks. With standardized formats and rich metadata, GraphNet enables fair comparison and reproducible evaluation of the general optimization capabilities of tensor compilers, thereby supporting advanced research such as AI for System on compilers.

## 📣 News
- [2025-10-14] ✨ Our technical report is out: a detailed study of dataset construction and compiler benchmarking, introducing the novel performance metrics Speedup Score S(t) and Error-aware Speedup Score ES(t). [📘 GraphNet: A Large-Scale Computational Graph Dataset for Tensor Compiler Research](https://arxiv.org/abs/2510.24035)
- [2025-8-20] 🚀 The second round of [open contribution tasks](https://github.com/PaddlePaddle/Paddle/issues/74773) was released. (completed ✅)
- [2025-7-30] 🚀 The first round of [open contribution tasks](https://github.com/PaddlePaddle/GraphNet/issues/44) was released.  (completed ✅)
## 📊 Benchmark Results
We evaluate two representative tensor compiler backends, CINN (PaddlePaddle) and TorchInductor (PyTorch), on GraphNet's NLP and CV subsets. The evaluation adopts two quantitative metrics proposed in the [Technical Report](https://arxiv.org/abs/2510.24035):
- **Speedup Score** S(t) — evaluates compiler performance under varying numerical tolerance levels.
<div align="center">
  <img src="/pics/St-result.jpg" alt="Speedup Score S_t Results" width="80%">
</div>

- **Error-aware Speedup Score** ES(t) — further accounts for runtime and compilation errors.
<div align="center">
  <img src="/pics/ESt-result.jpg" alt="Error-aware Speedup Score ES_t Results" width="80%">

</div>

## ⚡ Quick Start
This section shows how to evaluate tensor compilers and reproduce benchmark results (for compiler users and developers),
as well as how to contribute new computation graphs (for GraphNet contributors).

### ⚖️ Compiler Evaluation

**Step 1: Benchmark**

Use graph_net.torch.test_compiler to benchmark GraphNet samples with specific batch and logging configurations:

```bash
# Set your benchmark directory
export GRAPH_NET_BENCHMARK_PATH=/home/yourname/graphnet_benchmark/

# Run benchmark
python -m graph_net.torch.test_compiler \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name/ \
  --compiler /custom/or/builtin/compiler/ \
  --device /device/to/execute/ \
  --warmup /times/to/warmup/ \
  --trials /times/to/test/ \
  > $GRAPH_NET_BENCHMARK_PATH/log.log 2>&1

# Note: If --compiler is omitted, PyTorch’s built-in compiler is used by default.
```

After executing, `graph_net.torch.test_compiler` will:
1. Running the original model in eager mode to record a baseline.
2. Compiling the model with the specified backend (e.g., CINN, TVM, Inductor, TensorRT, XLA, BladeDISC).
3. Executing the compiled model and collecting its runtime and outputs.
4. Conduct speedup by comparing the compiled results against the baseline (if no execution failure occurs).

**Step 2: Analysis**

Use the three scripts `graph_net.plot_St`, `graph_net.plot_ESt` and `graph_net.plot_violin` to generate St plot, ESt plot, and [violin plot](https://en.m.wikipedia.org/wiki/Violin_plot) based on speedup, correctness and runtime information from benchmark logs.

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

# Note: If --negative-speedup-penalty is omitted, p=0 is used by default.
# If --fpdb, b=0.1 is used by default.

python -m graph_net.plot_violin \
  --benchmark-path $GRAPH_NET_BENCHMARK_PATH/JSON_results/ \
  --output-dir $GRAPH_NET_BENCHMARK_PATH
```

The scripts are designed to process a file structure as `/benchmark_path/category_name/`, and items on x-axis are identified by name of the sub-directories. After executing, several summary plots of result in categories (model tasks, libraries...) will be exported to `$GRAPH_NET_BENCHMARK_PATH`.

### Hardware Regression Testing
We also provide a two-step workflow that validates compiler correctness and performance against a "golden" reference, which is crucial for hardware-specific testing and regression tracking. Details can be found in this [guide](./docs/hardware_test.md).

### 🧱 Construction & Contribution Guide
Want to understand how GraphNet is built or contribute new samples?
Check out the [Construction Guide](./docs/README_contribute.md) for details on the extraction and validation workflow.


## 🚀 Future Roadmap

1. Scale GraphNet to 10K+ graphs.
2. Further annotate GraphNet samples into more granular sub-categories
3. Extract samples from multi-GPU scenarios to support benchmarking and optimization for large-scale, distributed computing.
4. Enable splitting full graphs into independently optimized subgraphs and operator sequences.

**Vision**: GraphNet aims to lay the foundation for AI for Compiler by enabling **large-scale, systematic evaluation** of tensor compiler optimizations, and providing a **dataset for models to learn** and transfer optimization strategies.

## GraphNet Community

You can join our community via following group chats. Welcome to ask any questions about using and building GraphNet.

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

## License and Acknowledgement

GraphNet is released under the [MIT License](./LICENSE).

If you find this project helpful, please cite:

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
