# 共创者指引
> Move fast, break things.

### 一、环境准备
1. **Fork 仓库**：
   
    a. 访问 [https://github.com/PaddlePaddle/GraphNet](https://github.com/PaddlePaddle/GraphNet)

    b. 点击右上角的 **Fork** 按钮，将仓库复制到你的账户下。

2. **克隆仓库到本地**：

```bash
git clone https://github.com/你的用户名/GraphNet.git
cd GraphNet
```
3. **添加上游仓库引用**：

```bash
git remote add upstream https://github.com/PaddlePaddle/GraphNet.git
```
4. **安装依赖**

```bash
pip install torch, torchvision
```
5. **设置提取路径**：

```bash
export GRAPH_NET_EXTRACT_WORKSPACE=/home/yourname/graphnet_workspace
```
### 二、编辑脚本
#### 1. 编辑自动抽取脚本
> 我们鼓励借助LLM Agent的能力，把本文档中的example、仓库中不同模型的example与extractor装饰器和promts一起输入，面向所需新的模型开发抽取脚本。

核心流程示例：
```python
import torch
from torchvision.models import get_model, get_model_weights, list_models  # 或者其他模型库
from graph_net.torch.extractor import extract

def run_model(name: str, device_str: str) -> None:
    """
    对指定模型执行计算图抽取流程并导出结果。

    Args:
        name (str): 模型名称（例如 'resnet50'、'vit_b_16'、'bert-base-uncased' 等）。
        device_str (str): 运行设备标识（'cpu' 或 'cuda:0' 等）。
    """
    device = torch.device(device_str)
    print(f"\nTesting model: {name} on {device_str}")

    # 1. 加载模型权重
    weights = None
    try:
        w = get_model_weights(name)
        weights = w.DEFAULT
    except Exception:
        # 若模型库不支持权重自动加载，可留空或在此手动指定本地权重路径
        pass

    # 2. 实例化模型
    try:
        model = get_model(name, weights=weights)
    except Exception as e:
        print(f"[FAIL] {name}: instantiate model error - {e}")
        return

    # 3. 构造输入张量（默认适用于图像分类模型）
    cfg = getattr(model, "default_cfg", {})
    C, H, W = cfg.get("input_size", (3, 224, 224))
    input_data = torch.rand(1, C, H, W, device=device)

    # 4. 包装并抽取计算图
    model = model.to(device).eval()
    wrapped = extract(name=name)(model).eval()
    try:
        with torch.no_grad():
            wrapped(input_data)
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: extract error - {e}")
```


a. **准备模型**：加载模型定义及权重。

b. **构造输入**

> GraphNet的示例仅限于特定模型，在完成任务时可能需要改输入数据的构造。

不同模型对输入格式要求不一致，需根据模型类型生成合适的 `input_data`：


<table>
    <thead>
        <tr>
            <th>模型类型</th>
            <th>输入构造示例</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>图像分类 / CV</td>
            <td><code>torch.rand(1, C, H, W)</code>；如 ResNet、ViT 默认 <code>(3,224,224)</code></td>
        </tr>
        <tr>
            <td>视频模型</td>
            <td><code>torch.rand(1, C, T, H, W)</code>；如 R3D、MViT 中 <code>T=num_frames</code></td>
        </tr>
        <tr>
            <td>NLP 文本模型</td>
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
            <td>多输入 / 复杂</td>
            <td>根据模型<code>forward</code>签名构造，如同时输入图像特征和位置编码等时，将所有Tensor按顺序或命名打包成 <code>tuple</code>/<code>dict</code></td>
        </tr>
    </tbody>
</table>

> **提示**：如果不确定 `model.forward()` 需要哪些参数，可以先打印签名：
```python
import inspect
print(inspect.signature(model.forward))
```

 c. **使用装饰器**

i. 推荐直接使用简洁的链式调用，例如上面示例中的`wrapped = extract(name=name)(model).eval()`。

ii. 如果需要使用外置的显式`@extract`装饰器，则需要装饰模型的`nn.Module`子类。




#### 2. 【可选】调整extractor
在少数情况下，可能需要修改extract decorator从而适配特定模型。此时，请提前提起issue并**特别标注说明**，与我们一起规划该特性。

值得注意的是，这种操作更考验开发者的能力，同时我们也会对新的extractor代码质量及安全性进行人工Review。

同时，完成extractor的优化开发意味着可以适配新的一系列模型，从而使得您可以获得一大批的任务激励；

如果新方法能适配的模型数量有限，则ROI较低，这也体现了“Eat the frog first”的原则。



#### 3. TroubleShot指南：借助 LLM 自动补全输入构造
> 示例代码仅覆盖部分常见模型，实际使用中可能需要根据具体模型调整输入张量的构造。
> 如果在抽取计算图过程中遇到错误，可以直接把错误日志和代码文件送给LLM Agent处理。
> 下方示例展示了一个Debug和输入生成的流程记录。


在首次抽取计算图时，可能因 `input_data` 维度不匹配而导致失败。此时我们借助 LLM，根据日志提示补充或修正输入构造逻辑。

示例错误日志：
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
从日志可见，该模型在做 3D 卷积时，输入通道数（1）与权重通道数（3）不符。基于这一提示，向 LLM 请求补全如下输入逻辑，即为时序模型增加帧维度 `T`：

```python
if any(tok in name for tok in ("r2plus1d")):
    # 时序模型需按 (B, C, T, H, W) 构造输入
    T = cfg.get("num_frames", 16)
    input_data = torch.rand(1, C, T, H, W, device=device)
```
此时再次运行 `extract(name)(model)(input_data)`，即可成功抽取计算图。



#### 4. 常见问题
 * **模型名称不在 **`list_models()`** 中**：
     * 自行从第三方库加载模型或手动实现 `get_model` 接口。
     * 确保 `extract(name=name)` 中的 `name` 与导出文件名一致。

 * **输入维度不匹配**：
     * 捕获异常后，打印模型 `cfg` 或 `forward` 签名进行对比。
     * 若模型有动态输入长度（如可变 sequence length），可使用临时最大长度测试。

 * **显存不足 / 速度慢**：
     * 尝试先在 CPU 上小 batch 测试；
     * 对于超大模型，建议分阶段抽取或使用更小 `input_data`。




#### 5. 进阶用法
  * **多输入/多输出模型**：
      * `input_data` 可为 `tuple` 或 `dict`，并在 `wrapped(...)` 时一并传入。

  * **分布式 & 并行抽图**：
      * 可结合 `multiprocessing` 或 `torch.multiprocessing`，多进程并行抽取多模型。

  * **自定义 hooks**：
      * 如果需要在特定层插入钩子，可在 `wrapped = extract(...)(model, hooks=...)` 中传入自定义函数。



### 三、抽取计算图
1. **extract 抽取**

运行集成了`@graph_net.torch.extract`或`@graph_net.paddle.extract`的自动提取脚本，例如：

```bash
# Extract the ResNet‑18 computation graph
python -m graph_net.test.vision_model_test
```
按照预期，应当在您的`$GRAPH_NET_EXTRACT_WORKSPACE`目录下记录所抽取的文件。



2. **validate 自查**

```bash
python -m graph_net.torch.validate --model-path $GRAPH_NET_EXTRACT_WORKSPACE/model_name
```
`validate` 验证您刚刚抽取的计算图符合Dataset Construction Constraints，如果结果为Success，则可以继续。



### **四、提交**计算图
1. **配置贡献者用户名和email**

```bash
python -m graph_net.config \
    --global \
    --username "john_doe" \
    --email "john@example.com"
```

2. **提交修改**

移动新增的计算图样本到**samples**目录，然后提交。
```bash
git add <新计算图样本>
git commit -m "描述"
```
请注意，如果有第三方算子，需要贡献者自行打包到计算图压缩包内。

3. **推送分支到远程**（你的 Fork 仓库）

```bash
git push origin feature/your-branch-name
```
4. **提交 Pull Request**

> **注意**：为方便管理，每个PR应遵守Single Responsibility Principle (SRP)原则，**仅新增单一份计算图、或聚焦于单一功能改进**，避免将多个修改混合提交。例如，如果您修改了抓取方法，然后为支持某类模型收集了数据，那么其中每份单个模型的计算图、修改的新一份抓取方法，都应打开为独立的PR。

 1. 访问你的 Fork 仓库页面（`https://github.com/你的用户名/GraphNet`）。

 2. 页面会提示 **Compare & Pull Request**，点击它。

 3. 使用以下模版填写：

**// ------- PR 标题 --------**

`[Type (New Sample | Feature Enhancement | Bug Fix | Other)] Brief Description`

`eg. [Feature Enhancement] Support Bert Model Family on Pytorch`

`eg. [New Sample] Add "distilbert-base-uncased" Model Computational Graph`

**// ------- PR 内容 --------**

```markdown
Model:  eg. distilbert-base-uncased
Framework: eg. Pytorch/Paddle
Dependency: eg. torchvision, transformers
Content: Description
```

 4. 点击 **Create Pull Request**。
   
 5. GraphNet团队会在机器人辅助下审查并合入PR。

> 其它信息及未明确的规范，可参照 [Paddle社区统一的代码贡献流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/code_contributing_path_cn.html)
