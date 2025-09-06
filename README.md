# GraphNet  ![](https://img.shields.io/badge/version-v0.1-brightgreen) ![](https://img.shields.io/github/issues/PaddlePaddle/GraphNet?label=open%20issues)    [![](https://img.shields.io/badge/Contribute%20to%20GraphNet-blue)](https://github.com/PaddlePaddle/GraphNet/issues/98)


**GraphNet** is a large-scale dataset of deep learning **computation graphs**, built as a standard benchmark for **tensor compiler** optimization. It provides 2.7K computation graphs extracted from state-of-the-art deep learning models spanning diverse tasks and ML frameworks. With standardized formats and rich metadata, GraphNet enables fair comparison, reproducible evaluation, and deeper research into the general optimization capabilities of tensor compilers.
<br>
<div align="center">
<img src="/pics/graphnet_overview.jpg" alt="GraphNet Architecture Overview" width="65%">
</div>

With GraphNet, users can:
1. **Contribute new computation graphs** through the built-in automated extraction and validation pipeline.
2. **Evaluate tensor compilers** on existing graphs with the integrated compiler evaluation tool, supporting multiple compiler backends.
3. **Advance research** in tensor compiler optimization using the test data and statistics provided by GraphNet.




**Vision**: We aim to achieve cross-hardware portability of compiler optimizations by allowing models to learn and transfer optimization strategies. It will significantly  reduce the manual effort required to develop efficient operator implementations.


## Dataset Construction

To guarantee the dataset’s overall quality, reproducibility, and cross-compiler compatibility, we define the following construction **constraints**:

1. Dynamic graphs must execute correctly.
2. Graphs and their corresponding Python code must support serialization and deserialization.
3. The full graph can be decomposed into two disjoint subgraphs.
4. Operator names within each computation graph must be statically parseable.
5. If custom operators are used, their implementation code must be fully accessible.


### Graph Extraction & Validation
For full implementation details, please refer to the [Co-Creation Tutorial](https://github.com/PaddlePaddle/GraphNet/blob/develop/CONTRIBUTE_TUTORIAL.md#co-creation-tutorial).

**Demo: Extract & Validate ResNet‑18**
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


## Compiler Evaluation

The compiler evaluation process takes a GraphNet sample as input and involves:
1. Running the original model in eager mode to record a baseline.
2. Compiling the model with the specified backend (e.g., CINN, TorchInductor, TVM).
3. Executing the compiled model and collecting its runtime and outputs.
4. Analyzing performance by comparing the compiled results against the baseline.

### Evaluation Metrics

We define two key metrics here: **rectified speedup** and **GraphNet Score**. Rectified speedup measures runtime performance while incorporating compilation success, time cost, and correctness. GraphNet Score aggregates the rectified speedup of a compiler on specified tasks, providing a measure of its general optimization capability. 

**Demo: How to benchmark your compiler on the model:**

```
python3 -m graph_net.torch.test_compiler \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name/ \
  --compiler /path/to/custom/compiler 
# Note: if --compiler is omitted, PyTorch’s built-in compiler is used by default
```

### Evaluation Results Example

<div align="center">
<img src="/pics/Eval_result.jpg" alt="Violin plots of rectified speedup distributions" width="65%">
</div>


## Roadmap

1. Scale GraphNet to 10K+ graphs.
2. Further annotate GraphNet samples into more granular sub-categories
3. Extract samples from multi-GPU scenarios to support benchmarking and optimization for large-scale, distributed computing.
4. Enable splitting full graphs into independently optimized subgraphs and operator sequences.

## GraphNet Community:


You can join GraphNet community via the following group chats.


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



##  License
This project is released under the [MIT License](LICENSE).

