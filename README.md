# GraphNet  ![](https://img.shields.io/badge/version-v0.1-brightgreen) ![](https://img.shields.io/github/issues/PaddlePaddle/GraphNet?label=open%20issues)    [![](https://img.shields.io/badge/Contribute%20to%20GraphNet-blue)](https://github.com/PaddlePaddle/GraphNet/issues/98)


**GraphNet** is a large-scale dataset of deep learning **computation graphs**, designed to serve as a standard benchmark and training corpus for **AI-driven tensor compiler optimization**. It contains diverse graphs extracted from state-of-the-art models, enabling effective evaluation of compiler pass optimizations across frameworks and hardware platforms.


With GraphNet, users can:
1. Quickly benchmark the optimization performance of various compiler strategies.
2. Easily conduct regression tests on existing compilers.
3. Train AI‑for‑Systems models to automatically generate compiler optimization passes.

**Vision**: We aim to achieve cross-hardware portability of compiler optimizations by allowing models to learn and transfer optimization strategies. It will significantly  reduce the manual effort required to develop efficient operator implementations.


### Dataset Construction Constraints：
1. Dynamic graphs must execute correctly.
2. Graphs and their corresponding Python code must support serialization and deserialization.
3. The full graph can be decomposed into two disjoint subgraphs.
4. Operator names within each computation graph must be statically parseable.
5. If custom operators are used, their implementation code must be fully accessible.


## ⚡ Quick Start
For full implementation details, please refer to the [Co-Creation Tutorial](https://github.com/PaddlePaddle/GraphNet/blob/develop/CONTRIBUTE_TUTORIAL.md#co-creation-tutorial).
### Benchmark your compiler on the model:

**graph_net.torch.test_compiler** 
```
python3 -m graph_net.torch.test_compiler \
  --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name/ \
  --compiler /path/to/custom/compiler 
# Note: if --compiler is omitted, PyTorch’s built-in compiler is used by default
```

### Contribute computation graphs to GraphNet:
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

**graph_net.pack**
```
# Create a ZIP archive of $GRAPH_NET_EXTRACT_WORKSPACE.
# The --clear-after-pack flag (True|False) determines whether to delete the workspace after packing.
python -m graph_net.pack \
  --output /path/to/output.zip \
  --clear-after-pack True
```

Note: To configure your user details (username and email) for GraphNet, run:
```
python -m graph_net.config --global \
  --username "your-name" \
  --email "your-email"
```
Once you have packaged these extracted computation graphs, submit them to the GraphNet community via the following group chats.


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

