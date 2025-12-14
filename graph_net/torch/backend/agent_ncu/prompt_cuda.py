PROMPT_SYSTEM = """You are a coding assistant that writes CUDA code."""

PROMPT_GENERATE = """
You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.
Here an example to show you the syntax of inline embedding custom CUDA operators in torch: 

The example given architecture is:
``` 
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
``` 

The example new arch with custom CUDA kernels looks like this: 
```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}


torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
``` 

You are given the following architecture: 
"""


PROMPT_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators! (Never optimize Conv operators, keep them in torch API).
Name your optimized output architecture ModelNew. **NOTE: There are no args in the __init__ function of Model.**
Output the new code in codeblocks. (All inputs are on GPU with float32, so no need to check is_cuda or dtype, don't include testing code `if xx.is_cuda()` and `if xx.dtype==float32`.)
Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. 
Just output the new model code, no other text, and NO testing code! 
"""


def generate_default_cuda_prompt(torch_model_code: str) -> str:
    prompt = PROMPT_GENERATE + "'''\n" + torch_model_code + "'''\n" + PROMPT_INSTRUCTION
    return prompt


def judge_optimize_prompt(pytorch_code, cuda_code, ncu_matrics):
    device = "A800"
    Architecture = "Ampere"
    prompt = f"""
You are a senior CUDA performance engineer. Read the target GPU spec, the PyTorch reference code, the current CUDA candidate, and the Nsight Compute metrics. Then identify **exactly one** highest-impact speed bottleneck by 3-4 most important metrics, propose ** exactly one** optimisation method and propose a modification plan. Be surgical and metrics-driven.
Rules: 
    Return **one and only one** optimisation method the largest expected speedup.
    Prefer changes that directly address measured bottlenecks (occupancy limits, memory coalescing, smem bank conflicts, register pressure, long/short scoreboard stalls, tensor−core underutilisation, etc.).
    Keep fields brief; avoid lists of alternatives, disclaimers, or generic advice.
Output format (JSON): 
 ```json 
 {{  
      "bottleneck": "<max 40 words>",
      "optimisation method": "<max 40 words>", 
      "modification plan": "<max 40 words>"  
 }}
 ```

 # Target GPU
  GPU Name:{ device } 
  Architecture: { Architecture }
  # Pytorch Reference
  {{
  ```python
  {pytorch_code}
  ```
  }}
  # CUDA candidate 
  {{
  ```python
    {cuda_code}
  ```
  }}


    # Nsight Compute metrics (verbatim)
    {{ 
    {ncu_matrics}
    }}
    Read everything and follow the Rules exactly. Return the JSON in the specified format.
  
"""
    return prompt


def coder_optimize_prompt(cuda_code, strategy):
    device = "A800"
    Architecture = "Ampere"
    prompt = f"""
# Target GPU 
GPU Name: {device} 
Architecture: {Architecture} 

You are a CUDA-kernel optimization specialist.

Analyze the provided architecture and **strictly apply the following STRATEGY** to produce an improved CUDA kernel.
CUDA_CODE:
{{
```python
{cuda_code}
```
}}
STRATEGY:
{{
{strategy}
}}

GOAL 
- Maximize performance gain on the target GPU, improving both latency and throughput as the primary objective.

- Maintain numerical correctness as much as possible, preferably within atol ≤ 1e-5 and rtol ≤ 1e-8, 
  unless further relaxation is explicitly justified by performance.

- Preserve the public Python API (same inputs/outputs, shapes, dtypes).

- Never optimize Conv operators, keep them in torch API.

OUTPUT RULES (STRICT)
1. Inside the block, follow **exactly** this order: 
    1. Imports 'torch', 'torch.nn', 'load inline'.
    2. 'source' triplequoted CUDA string(s) (kernel + host wrapper).
    3. 'cpp src' prototypes for *all* kernels you expose.
    4. **One** 'load inline' call per kernel group.
    5. 'class ModelNew(nn.Module)' mirrors original inputs/outputs but calls 32 your CUDA kernels.

2. **Do NOT include** testing code, 'if name == ” main ”', or extra prose.

```python
# <your optimized code> 
```
"""
    return prompt


def judge_correct_prompt(error_log, pytorch_code, cuda_code):
    prompt = f"""
You are a senior CUDA + PyTorch correctness auditor. Your job is to read a PyTorch reference and a CUDA candidate and report exactly one most critical correctness issue in the CUDA code that would cause a behavioral mismatch vs. the PyTorch reference. Be terse and precise.

Rules:

Return one and only one issue the single highest-impact problem.

Prefer semantic/correctness issues over micro-optimizations or style.

If multiple issues exist, pick the one that most changes outputs or gradients.

If nothing clearly wrong is found, say it explicitly.

Keep each field brief; avoid extra commentary, lists, or alternatives.

Output format (JSON): 
```json  
{{ 
    "critical issue": "<max 30 words>",
  "why it matters": "<max 35 words>", 
  "minimal fix hint": "<max 20 words>" 
}} 
```
You are given:
ERROR LOG: {error_log}

PyTorch reference (ground truth): 
{{
```python
{pytorch_code}
```
}} 

CUDA candidate (to audit):
{{
```python
{cuda_code}
```
}}
Follow the Rules and produce the JSON exactly in the specified format.
"""
    return prompt


def coder_correct_prompt(error_log, cuda_code, modify_text):
    prompt = f"""
You are a senior CUDA-extension developer.
Your job is to **FIX** the compilation or runtime errors in the Python script shown below.
OUTPUT RULES (STRICT)
1. Inside the block, follow **exactly** this order: 
    1. Imports 'torch', 'torch.nn', 'load inline'.
    2. 'source' triplequoted CUDA string(s) (kernel + host wrapper).
    3. 'cpp src' prototypes for *all* kernels you expose.
    4. **One** 'load inline' call per kernel group.
    5. 'class ModelNew(nn.Module)' mirrors original inputs/outputs but calls 12 your CUDA kernels.

2. **Do NOT include** testing code, 'if name == ” main ”', or extra prose.

ERROR LOG:{{
{error_log}
}} 

OLD CODE (read-only) 
{{
```python
{cuda_code}
```
}}
Main Critical Problem 
{{
    {modify_text}
}}
 ```python 
 # <your corrected code> 
 ```
"""
    return prompt
