# GraphNet  ![](https://img.shields.io/badge/version-v0.0-brightgreen)

GraphNet is a large‑scale dataset for compiler development, providing researchers with a standardized, open‑access experimental environment. It includes numerous computation graphs extracted from deep learning models, making it easy to compare the optimization effectiveness of different compiler passes.

With GraphNet, users can:
1. Quickly benchmark the optimization performance of various compiler strategies.
2. Easily conduct regression tests on existing compilers.
3. Train AI‑for‑Systems models to automatically generate compiler optimization passes.

## ⚡ Quick Start
### Extract a computation graph
```
git clone https://github.com/PaddlePaddle/GraphNet.git
cd GraphNet

# Set your workspace directory (e.g. /home/yourname/graphnet_workspace)
export GRAPH_NET_EXTRACT_WORKSPACE=/home/yourname/graphnet_workspace

# Extract the ResNet‑18 computation graph
python graph_net/test/vision_model_test.py
```
### Validation
```
# Validate the extracted graph (e.g. /home/yourname/graphnet_workspace/resnet18)
python -m graph_net.torch.validate \
  --model-path /home/yourname/graphnet_workspace/resnet18
```

## Dataset  Construction  Constraints
GraphNet enforces the following constraints during dataset construction:

1. Dynamic graphs must execute correctly.
2. Each computation graph should include a standardized method for measuring performance.
3. Graphs and their corresponding Python code must support serialization and deserialization.
4. The full graph can be decomposed into two disjoint subgraphs.
5. Compiler passes or behaviors must be configurable.
6. Operator names within each computation graph must be statically parseable.
7. If custom operators are used, their implementation code must be fully accessible.
8. Graph execution on different hardware backends must be configurable via a unified interface.

## Community

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

##  License
This project is released under the MIT License

