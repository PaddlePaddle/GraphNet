# GraphNet Agent 使用指南

自动从 HuggingFace 模型抽取计算图的 Agent 工具。

---

## 环境准备

# 目录
在GraphNet目录下运行即可，不需要安装

# 设置代理（访问 HuggingFace 需要）
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891

# LLM 兜底功能需要 ducc CLI（可选）
export PATH="/root/.comate/baidu-cc/bin:$PATH"
```

---

## 快速开始

```python
from graph_net.agent import GraphNetAgent

agent = GraphNetAgent()
ok = agent.extract_sample("prajjwal1/bert-tiny")
print("成功" if ok else "失败")
```

默认 workspace 为 `/work/graphnet_workspace`，输出目录按 `组织_模型名` 命名，例如：
`/work/graphnet_workspace/prajjwal1_bert-tiny/`

---

## 初始化参数

```python
GraphNetAgent(
    workspace  = "/work/graphnet_workspace",  # 工作目录根路径
    hf_token   = None,                        # HuggingFace Token（私有模型需要）
    llm_retry  = True,                        # 失败时调用 LLM 兜底修复
)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `workspace` | `/work/graphnet_workspace` | 工作目录，自动创建子目录结构 |
| `hf_token` | `None` | HF access token，公开模型无需填写 |
| `llm_retry` | `True` | 模板脚本失败后，调用 `ducc -p` 让 LLM 修复并重试一次 |

---

## 批量抽取

```python
from graph_net.agent import GraphNetAgent

agent = GraphNetAgent()

models = [
    "prajjwal1/bert-tiny",
    "distilbert/distilgpt2",
    "hf-internal-testing/tiny-random-ViTModel",
    "hf-internal-testing/tiny-random-T5Model",
    "openai/clip-vit-base-patch32",
]

results = {}
for model_id in models:
    results[model_id] = agent.extract_sample(model_id)

# 打印汇总
for mid, ok in results.items():
    print(f"{'OK  ' if ok else 'FAIL'} {mid}")
```

---

## 工作目录结构

```
/work/graphnet_workspace/
├── models/                         # HuggingFace 下载缓存（仅 config，不含权重）
│   └── models--org--model-name/
├── generated/                      # 自动生成的抽取脚本
│   └── org_model-name/
│       ├── run_model.py            # 模板生成脚本
│       └── run_model_llm.py        # LLM 修复脚本（首次失败时生成）
├── org_model-name/                 # 计算图输出（以 组织_模型名 命名）
│   ├── model.py                    # 计算图结构
│   ├── graph_net.json              # 图结构 JSON
│   ├── input_meta.py               # 输入元信息
│   ├── input_tensor_constraints.py # 输入约束
│   ├── weight_meta.py              # 权重元信息
│   ├── graph_hash.txt              # 图结构哈希（用于去重）
│   └── run_model.py                # 归档的抽取脚本
├── samples/                        # 去重比对参考库
└── logs/                           # 运行日志
    └── agent_YYYYMMDD_HHMMSS.log
```

---

## 抽取流程

```
HuggingFace model_id
        │
        ▼
① 下载配置文件          仅下载 config.json 等配置，跳过权重文件（*.bin / *.safetensors
        │               / *.tflite / *.mlmodel / *.onnx 等），模型参数随机初始化
        ▼
② 解析配置元数据        读取 config.json，推断 model_type / vocab_size / input_shapes
        │
        ▼
③ 生成抽取脚本          模板生成 run_model.py，含随机输入构造 + graph_net.torch.extract 调用
        │
        ▼
④ 子进程执行脚本        独立 Python 进程运行，注入 GRAPH_NET_EXTRACT_WORKSPACE 环境变量
        │
        ├─ 成功 ──────────────────────────────────────────────────┐
        │                                                         │
        └─ 失败 → ⑤ LLM 兜底（llm_retry=True 且 ducc 可用）      │
                    │                                            │
                    ▼                                            │
             ducc -p "<prompt>"                                  │
             生成 run_model_llm.py                               │
                    │                                            │
                    ▼                                            │
             子进程重试执行                                       │
                    │                                            │
        ────────────┘                                            │
                                                                 ▼
⑥ 生成 graph_hash.txt + 去重检查 + 验证输出文件完整性 + 归档脚本
```

---

## LLM 兜底机制

当模板脚本执行失败时，若满足以下条件则触发 LLM 兜底：

- `llm_retry=True`（默认开启）
- `ducc` 命令可用（在 PATH 或 `/root/.comate/baidu-cc/bin/ducc`）

LLM 收到的信息包括：失败脚本原文、报错信息、`config.json` 内容。
LLM 必须遵守以下约束（写在 system prompt 里）：

1. 必须调用 `graph_net.torch.extract(name="...")(model).eval()`
2. 只能使用 `torch`、`transformers`、`graph_net` 三个包
3. 输入张量随机构造，无需真实数据
4. 设备必须用 `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
5. 不下载模型权重

每个模型**最多触发两次** LLM 兜底。第二次会把第一次修复后的脚本及其新的报错一并送给 LLM，方便它在上一轮基础上进一步修正。

---

## 常见问题

**Q：为什么某些模型下载很慢？**

HuggingFace 上部分模型除 PyTorch 权重外还包含 CoreML (`.mlmodel`)、TFLite (`.tflite`)、ONNX (`.onnx`) 等格式的文件，这些文件体积大（数百 MB）。Agent 已在 `ignore_patterns` 中跳过这些格式，若遇到新格式可在 [huggingface_fetcher.py](../graph_net/agent/model_fetcher/huggingface_fetcher.py) 里补充。

**Q：抽取结果目录名规则是什么？**

以 `组织_模型名` 命名（`/` 替换为 `_`），例如：
- `prajjwal1/bert-tiny` → `prajjwal1_bert-tiny/`
- `hf-internal-testing/tiny-random-ViTModel` → `hf-internal-testing_tiny-random-ViTModel/`

**Q：关闭 LLM 兜底怎么做？**

```python
agent = GraphNetAgent(llm_retry=False)
```

**Q：如何使用私有模型？**

```python
import os
agent = GraphNetAgent(hf_token=os.environ["HF_TOKEN"])
```

**Q：如何检查某次抽取是否成功？**

`extract_sample()` 返回 `True` 表示成功，同时可以检查输出目录是否存在 7 个文件：
`model.py`、`graph_net.json`、`input_meta.py`、`input_tensor_constraints.py`、
`weight_meta.py`、`graph_hash.txt`、`run_model.py`。
