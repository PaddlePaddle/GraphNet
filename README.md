# GraphNet  ![](https://img.shields.io/badge/version-v0.1-brightgreen) ![](https://img.shields.io/github/issues/PaddlePaddle/GraphNet?label=open%20issues)    [![](https://img.shields.io/badge/Contribute%20to%20GraphNet-blue)](https://github.com/PaddlePaddle/GraphNet/issues/98)

**GraphNet** is a large-scale dataset of deep learning **computation graphs**, built as a standard benchmark for **tensor compiler** optimization. It provides 2.7K computation graphs extracted from state-of-the-art deep learning models spanning diverse tasks and ML frameworks. With standardized formats and rich metadata, GraphNet enables fair comparison and reproducible evaluation of the general optimization capabilities of tensor compilers, thereby supporting advanced research such as AI for System on compilers (AI for Compiler).

<br>
<div align="center">
<img src="/pics/Eval_result.png" alt="Violin plots of speedup distributions" width="65%">
</div>

Compiler developers can use GraphNet samples to evaluate tensor compilers (e.g., CINN, TorchInductor, TVM) on target tasks. The figure above shows the speedup of two compilers (CINN and TorchInductor) across two tasks (CV and NLP).

## 🧱 Dataset Construction

To guarantee the dataset’s overall quality, reproducibility, and cross-compiler compatibility, we define the following construction **constraints**:

1. Computation graphs must be executable in imperative (eager) mode.
2. Computation graphs and their corresponding Python code must support serialization and deserialization.
3. The full graph can be decomposed into two disjoint subgraphs.
4. Operator names within each computation graph must be statically parseable.
5. If custom operators are used, their implementation code must be fully accessible.

### Graph Extraction & Validation

We provide automated extraction and validation tools for constructing this dataset.

<div align="center">
<img src="/pics/graphnet_overview.jpg" alt="GraphNet Architecture Overview" width="65%">
</div>

**Demo: Extract & Validate ResNet‑18**
```bash
git clone https://github.com/PaddlePaddle/GraphNet.git
cd GraphNet

# Set your workspace directory
export GRAPH_NET_EXTRACT_WORKSPACE=/home/yourname/graphnet_workspace/

# Extract the ResNet‑18 computation graph
python graph_net/test/vision_model_test.py

# Validate the extracted graph (e.g. /home/yourname/graphnet_workspace/resnet18/)
python -m graph_net.torch.validate \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/resnet18/
```

**Illustration: How does GraphNet extract and construct a computation graph sample on PyTorch?**

<div align="center">
<img src="/pics/graphnet_sample.png" alt="GraphNet Extract Sample" width="65%">
</div>

* Source code of custom_op is required **only when** corresponding operator is used in the module, and **no specific format** is required.

**Step 1: graph_net.torch.extract**

Import and wrap the model with `graph_net.torch.extract(name=model_name, dynamic=dynamic_mode)()` is all you need:

```bash
import graph_net

# Instantiate the model (e.g. a torchvision model)
model = ...  

# Extract your own model
model = graph_net.torch.extract(name="model_name", dynamic="True")(model)
```

After running, the extracted graph will be saved to: `$GRAPH_NET_EXTRACT_WORKSPACE/model_name/`.

For more details, see docstring of `graph_net.torch.extract` defined in `graph_net/torch/extractor.py`.

**Step 2: graph_net.torch.validate**

To verify that the extracted model meets requirements, we use `graph_net.torch.validate` in CI tool and also ask contributors to self-check in advance:

```bash
python -m graph_net.torch.validate \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

All the **construction constraints** will be examined automatically. After passing validation, a unique `graph_hash.txt` will be generated and later checked in CI procedure to avoid redundant.

## ⚖️ Compiler Evaluation

**Step 1: Benchmark**

We use `graph_net.torch.test_compiler` to benchmark GraphNet samples with specific batch and log configurations:

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

**Step 2: Generate JSON Record**

This step is to extract information (including failure) from logs in benchmark.
All the information will be saved to multiple `model_compiler.json` files via:

```bash
python -m graph_net.log2json \
  --log-file $GRAPH_NET_BENCHMARK_PATH/log.log \
  --output-dir $GRAPH_NET_BENCHMARK_PATH/JSON_results/
```

**Step 3: Analysis**

After processing, we provide `graph_net.violin_analysis` to generate [violin plot](https://en.m.wikipedia.org/wiki/Violin_plot) and `graph_net.S_analysis` to generate S and ES plot based on the JSON results.

```bash
python -m graph_net.violin_analysis \
  --benchmark-path $GRAPH_NET_BENCHMARK_PATH/JSON_results/ \
  --output-dir $GRAPH_NET_BENCHMARK_PATH

python -m graph_net.S_analysis \
  --benchmark-path $GRAPH_NET_BENCHMARK_PATH/JSON_results/ \
  --output-dir $GRAPH_NET_BENCHMARK_PATH \
  --negative-speedup-penalty penalty/power/for/negative/speedup \
  --fpdb base/penalty/for/severe/errors

# Note: If --negative-speedup-penalty is omitted, p=0 is used by default.
# If --fpdb, b=0.1 is used by default.
```

The scripts are designed to process a file structure as `/benchmark_path/category_name/`, and items on x-axis are identified by name of the sub-directories. After executing, several summary plots of result in categories (model tasks, libraries...) will be exported to `$GRAPH_NET_BENCHMARK_PATH`.

## 📌 Roadmap

1. Scale GraphNet to 10K+ graphs.
2. Further annotate GraphNet samples into more granular sub-categories
3. Extract samples from multi-GPU scenarios to support benchmarking and optimization for large-scale, distributed computing.
4. Enable splitting full graphs into independently optimized subgraphs and operator sequences.

**Vision**: GraphNet aims to lay the foundation for AI for Compiler by enabling **large-scale, systematic evaluation** of tensor compiler optimizations, and providing a **dataset for models to learn** and transfer optimization strategies.

## 💬 GraphNet Community

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

## 🪪 License
This project is released under the [MIT License](LICENSE).
