# Co-Creation Tutorial
> Move fast, break things.

### 1. Environment Setup

1. **Fork the Repository**:

    a. Visit [https://github.com/PaddlePaddle/GraphNet](https://github.com/PaddlePaddle/GraphNet)  
    b. Click the **Fork** button at the top-right to copy the repository to your own GitHub account.

2. **Clone the Repository Locally**:

```bash
git clone https://github.com/your-username/GraphNet.git
cd GraphNet
````

3. **Add Upstream Repository**:

```bash
git remote add upstream https://github.com/PaddlePaddle/GraphNet.git
```

4. **Install Dependencies**:

```bash
pip install torch, torchvision
```

5. **Set Extraction Workspace Path**:

```bash
export GRAPH_NET_EXTRACT_WORKSPACE=/home/yourname/graphnet_workspace
```

### 2. Editing Scripts

#### 2.1 Write Auto-Extraction Script

> We encourage utilizing LLM Agents by combining examples in this document, model examples in the repository, the extractor decorator, and prompts to develop new scripts for graph extraction tailored to desired models.

Example workflow:

```python
import torch
from torchvision.models import get_model, get_model_weights, list_models  # or other model libraries
from graph_net.torch.extractor import extract

def run_model(name: str, device_str: str) -> None:
    """
    Run computational graph extraction for the specified model.

    Args:
        name (str): Model name (e.g., 'resnet50', 'vit_b_16', 'bert-base-uncased').
        device_str (str): Device identifier (e.g., 'cpu' or 'cuda:0').
    """
    device = torch.device(device_str)
    print(f"\nTesting model: {name} on {device_str}")

    # 1. Load model weights
    weights = None
    try:
        w = get_model_weights(name)
        weights = w.DEFAULT
    except Exception:
        pass  # If weights cannot be loaded automatically, leave empty or specify manually

    # 2. Instantiate model
    try:
        model = get_model(name, weights=weights)
    except Exception as e:
        print(f"[FAIL] {name}: instantiate model error - {e}")
        return

    # 3. Construct input tensor (default for image classification models)
    cfg = getattr(model, "default_cfg", {})
    C, H, W = cfg.get("input_size", (3, 224, 224))
    input_data = torch.rand(1, C, H, W, device=device)

    # 4. Wrap and extract graph
    model = model.to(device).eval()
    wrapped = extract(name=name)(model).eval()
    try:
        with torch.no_grad():
            wrapped(input_data)
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: extract error - {e}")
```

a. **Prepare the model**: Load model definition and weights.

b. **Construct Input**

> GraphNet examples are limited to specific models. You may need to adapt the input data to match other models' requirements.

Different models require different input formats. Construct `input_data` accordingly:

<table>
    <thead>
        <tr>
            <th>Model Type</th>
            <th>Input Construction Example</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Image Classification / CV</td>
            <td><code>torch.rand(1, C, H, W)</code>; e.g., ResNet, ViT with default <code>(3,224,224)</code></td>
        </tr>
        <tr>
            <td>Video Models</td>
            <td><code>torch.rand(1, C, T, H, W)</code>; e.g., R3D, MViT where <code>T=num_frames</code></td>
        </tr>
        <tr>
            <td>NLP Text Models</td>
            <td>
                <pre><code class="language-python">tokenizer = AutoTokenizer.from_pretrained(model_name)
                  inputs = tokenizer(
                      text,
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=512,
                  )
                  input_data = {key: val.to(device) for key, val in inputs.items()}</code></pre>
            </td>
        </tr>
        <tr>
            <td>Multi-Input / Complex</td>
            <td>Construct based on the model's <code>forward</code> signature. When multiple inputs are required (e.g., image features and positional encodings), pack all tensors into a <code>tuple</code> or <code>dict</code> accordingly.</td>
        </tr>
    </tbody>
</table>


> **Tip**: If you're unsure what arguments `model.forward()` requires, print the signature:

```python
import inspect
print(inspect.signature(model.forward))
````

c. **Use the Extractor Decorator**

i. Prefer the concise chained style as in `wrapped = extract(name=name)(model).eval()`

ii. Alternatively, use the `@extract` decorator explicitly on the `nn.Module` subclass.

#### 2.2 \[Optional] Modify the Extractor

In rare cases, you may need to modify the `extract` decorator to support specific models. If so, please raise an issue and **clearly indicate** that you propose a new extractor feature.

Note that this requires a higher level of expertise, and we will manually review all new extractor code for quality and security.

Completing extractor improvements means your method can generalize across a family of models—eligible for bonus rewards.
If your method only supports a few models, ROI is low—thus the "Eat the frog first" principle applies.

#### 2.3 TroubleShooting: Let LLM Generate Input Automatically

> Sample code only covers some common models. You may need to tailor `input_data` for others.
> If errors occur during extraction, simply provide the error log and code to an LLM Agent.
> Below is an example of debugging and completing input logic.

On first extraction, an error may occur due to input shape mismatch. We consult LLM to resolve based on logs.

Example log:

```bash
[FAIL] r2plus1d_18: Dynamo failed to run FX node with fake tensors:
 call_function <built-in method conv3d of type object at 0x7f14dbbd1f40>(
   *(FakeTensor(..., device='cuda:0', size=(1, s0, s1, s1)),
     Parameter(FakeTensor(..., device='cuda:0', size=(45, 3, 1, 7, 7), requires_grad=True)),
     None, (1, 2, 2), (0, 3, 3), (1, 1, 1), 1),
   **{}
 ): got RuntimeError(
   'Given groups=1, weight of size [45, 3, 1, 7, 7], expected input[1, 1, s0, s1, s1]
    to have 3 channels, but got 1 channels instead'
 )
```

This indicates the model expects 3-channel input but received 1-channel. Fix with an additional `T` dimension:

```python
if any(tok in name for tok in ("r2plus1d")):
    # Temporal models require input of shape (B, C, T, H, W)
    T = cfg.get("num_frames", 16)
    input_data = torch.rand(1, C, T, H, W, device=device)
```

Re-run `extract(name)(model)(input_data)` to complete extraction.

#### 2.4 Common Issues

* **Model name not in `list_models()`**:

  * Load model from a third-party library or implement `get_model` manually.
  * Ensure `name` in `extract(name=...)` matches the filename to be exported.

* **Input dimension mismatch**:

  * Catch exceptions and inspect model `cfg` or `forward` signature.
  * For dynamic-length models (e.g., variable sequence length), test with a temporary max length.

* **Out of memory / slow performance**:

  * Try testing small batches on CPU.
  * For large models, extract in stages or reduce `input_data` size.

#### 2.5 Advanced Usages

* **Multi-input / multi-output models**:

  * Pass `input_data` as a `tuple` or `dict` into `wrapped(...)`.

* **Distributed & parallel extraction**:

  * Use `multiprocessing` or `torch.multiprocessing` to run extraction in parallel.

* **Custom hooks**:

  * Insert hooks at specific layers by passing them into `wrapped = extract(...)(model, hooks=...)`.


### 3. Extracting the Computational Graph

1. **Run extract**

Execute scripts with `@graph_net.torch.extract` or `@graph_net.paddle.extract`. For example:

```bash
# Extract the ResNet‑18 computational graph
python -m graph_net.test.vision_model_test
```

Expected output will be saved under `$GRAPH_NET_EXTRACT_WORKSPACE`.

2. **Run validate**

```bash
python -m graph_net.torch.validate --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```

`validate` checks if the extracted graph meets the Dataset Construction Constraints. If success, you’re ready to continue.


### 4. Submitting the Extracted Graph

1. **Set contributor username and email**

```bash
python -m graph_net.config \
    --global \
    --username "john_doe" \
    --email "john@example.com"
```

2. **Commit the changes**

Move the new sample to **samples** directory and commit.
```bash
git add <the new sample>
git commit -m "Description"
```
Note: If third-party ops are used, contributors must include them manually in the package.

3. **Push the branch to your fork**

```bash
git push origin feature/your-branch-name
```

4. **Submit a Pull Request**

> **Note**: For clarity and maintainability, each PR should follow the Single Responsibility Principle (SRP). Submit only a single graph or a focused feature improvement per PR. For example, if you both update extraction logic and collect multiple models, each graph and each method update should be a separate PR.

a. Visit your fork at `https://github.com/your-username/GraphNet`

b. Click **Compare & Pull Request** when prompted

c. Fill in the PR using this template:

**// ------- PR Title --------**

`[Type (New Sample | Feature Enhancement | Bug Fix | Other)] Brief Description`

e.g. `[Feature Enhancement] Support Bert Model Family on Pytorch`

e.g. `[New Sample] Add "distilbert-base-uncased" Model Computational Graph`

**// ------- PR Content --------**

```markdown
Model:  e.g., distilbert-base-uncased  
Framework: e.g., Pytorch / Paddle  
Dependency: e.g., torchvision, transformers  
Content: Description
```

d. Click **Create Pull Request**

e. The GraphNet team will review and merge it with automated assistance.

> For more information and unlisted policies, refer to the [Paddle contribution guide](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/code_contributing_path_cn.html)
