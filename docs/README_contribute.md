# Contributing to GraphNet
To guarantee the dataset’s overall quality, reproducibility, and cross-compiler compatibility, we define the following construction **constraints**:

1. Computation graphs must be executable in imperative (eager) mode.
2. Computation graphs and their corresponding Python code must support serialization and deserialization.
3. The full graph can be decomposed into two disjoint subgraphs.
4. Operator names within each computation graph must be statically parseable.
5. If custom operators are used, their implementation code must be fully accessible.

## Graph Extraction & Validation
GraphNet provides automated tools for graph extraction and validation.

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

**Illustration – Extraction Workflow**

<div align="center">
<img src="/pics/dataset_composition.png" alt="GraphNet Extract Sample" width="65%">
</div>

* Source code of custom_op is required **only when** corresponding operator is used in the module, and **no specific format** is required.

**Step 1: graph_net.torch.extract**

Wrap the model with the extractor — that’s all you need:

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

## 📁 Repository Structure
This repository is organized as follows:

| Directory | Description |
|------------|--------------|
| **graph_net/** | Core module for graph extraction, validation, and benchmarking |
| **paddle_samples/** | Computation graph samples extracted from PaddlePaddle |
| **samples/** | Computation graph samples extracted from PyTorch |
| **docs/** | Technical documents and contributor guides|

Below is the structure of the **graph_net/**:
```text
graph_net/
 ├─ config/    # Config files, params
 ├─ paddle/    # PaddlePaddle graph extraction & validation
 ├─ torch/     # PyTorch graph extraction & validation
 ├─ test/      # Unit tests and example scripts
 └─ *.py       # Benchmark & analysis scripts 
