# GraphNet Agent

自动样本抽取 Agent，实现从 HuggingFace ModelID 到 GraphNet Sample 的自动化转换。

## 安装

### 基础依赖
```bash
pip install torch transformers accelerate
pip install huggingface_hub>=0.20.0
```

## 环境配置

### 设置工作空间
```bash
export GRAPH_NET_EXTRACT_WORKSPACE=/path/to/your/workspace
```

未设置时默认使用 `~/graphnet_workspace`。也可在代码中显式指定：
```python
from graph_net.agent import GraphNetAgent
agent = GraphNetAgent(workspace="/path/to/workspace")
```

### HuggingFace Token（访问私有或受限模型）
```bash
export HF_TOKEN=hf_xxx
```

## 使用示例

### 单模型抽取
```python
from graph_net.agent import GraphNetAgent

agent = GraphNetAgent(
    hf_token=None,   # 可选，私有模型需要
    llm_retry=True,  # 失败时自动调用 ducc/claude 修复脚本并重试
)

success = agent.extract_sample("bert-base-uncased")
```

### 并行批量抽取
```bash
# 从文件读取模型列表（每行一个 model_id，# 开头为注释）
python graph_net/agent/parallel_extract.py --model-list models.txt

# 从 HuggingFace Hub 按下载量抓取模型
python graph_net/agent/parallel_extract.py --count 200 --task text-classification

# 指定 GPU 和 workspace
python graph_net/agent/parallel_extract.py \
    --model-list models.txt \
    --gpus 0,1,2,3 \
    --workspace /data/graphnet_workspace \
    --hf-token YOUR_TOKEN

# 结果保存为 JSON（默认自动生成带时间戳的文件名）
python graph_net/agent/parallel_extract.py --model-list models.txt --output result.json
```

`--gpus` 默认自动检测全部可用 GPU（读取 `CUDA_VISIBLE_DEVICES` 或 `nvidia-smi`）。

## parallel_extract.py 详解

`parallel_extract.py` 是面向批量场景的并行抽取脚本，适合一次性处理数百到数千个模型。

### 工作原理

- 所有待抽取的模型 ID 放入一个共享任务队列
- 每张 GPU 启动一个独立的 worker 子进程（`multiprocessing spawn` 模式，CUDA 安全）
- worker 空闲时主动从队列取任务，天然实现动态负载均衡
- 每个 worker 内部使用独立的 `GraphNetAgent`，彼此隔离，互不影响

### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-list` | — | 模型列表文件路径，每行一个 model_id，`#` 开头为注释 |
| `--count` | 100 | 未指定 `--model-list` 时，从 HuggingFace Hub 按下载量抓取的模型数量（需安装 `huggingface_hub`） |
| `--task` | — | HuggingFace 任务类型过滤，如 `text-classification`、`image-classification`（与 `--count` 配合使用） |
| `--gpus` | 自动检测 | 使用的 GPU 编号，逗号分隔，如 `0,1,2,3` |
| `--workspace` | `$GRAPH_NET_EXTRACT_WORKSPACE` 或 `~/graphnet_workspace` | 工作目录根路径 |
| `--hf-token` | — | HuggingFace API Token，私有或受限模型需要 |
| `--output` | 自动生成 | 结果 JSON 文件路径，默认为 `parallel_extract_<时间戳>.json` |

### 模型列表文件格式

```
# 文本模型
bert-base-uncased
google/flan-t5-base

# 视觉模型
openai/clip-vit-base-patch32
```

## 工作流程

1. **Fetch**: 从 HuggingFace 下载模型到本地缓存
2. **Analyze**: 解析 `config.json` 提取输入形状、dtype、模型类型等元数据
3. **CodeGen**: 根据元数据生成 `run_model.py` 抽取脚本
4. **Extract**: 在子进程中执行脚本抽取计算图
5. **LLM Retry**（可选）：若抽取失败，调用 `ducc`/`claude -p` 修复脚本并最多重试 2 次
6. **Deduplicate**: 基于 SHA-256 图哈希检查是否与已有样本重复
7. **Verify**: 使用 ForwardVerifier 验证样本可 forward

## LLM Retry

当模板生成的脚本执行失败时，若系统中存在 `ducc` 或 `claude` CLI，Agent 会自动将失败脚本、报错信息和模型 config 发给 LLM 进行修复，最多重试 2 次。

```python
# 禁用 LLM retry
agent = GraphNetAgent(llm_retry=False)
```

LLM retry 需要 `ducc` 或 `claude` 在 `PATH` 中可用。

## 辅助脚本

`graph_net/agent/scripts/` 目录下提供批量任务相关的辅助脚本：

### check_extraction_progress.sh

一键查看当前抽取任务的运行状态。

```bash
# 自动查找最新日志
bash graph_net/agent/scripts/check_extraction_progress.sh

# 或指定日志文件
bash graph_net/agent/scripts/check_extraction_progress.sh $HOME/workspace/logs_and_lists/batch7_safe_run.log
```

输出包括：进程状态（PID、CPU/内存、Worker 数）、日志最新进度、成功/失败统计、处理速度估算、预计剩余时间、磁盘空间、样本目录文件数。

### analyze_extraction_log.sh

分析已完成批次的抽取日志，输出失败分布和根因统计。

```bash
bash graph_net/agent/scripts/analyze_extraction_log.sh $HOME/workspace/logs_and_lists/batch7_safe_run.log
```

输出包括：总体统计（成功率）、失败原因一级分布、模型过大分布、异常类型分布（ValueError/Dynamo/IndexError 等）、HTTP 状态码分布、辅助文件（生成已处理和成功模型列表到 `/tmp/`）。

### gen_hash_and_dedup.py

子图去重脚本。遍历抽取结果目录，为每个子图的 `model.py` 生成 SHA256 哈希，找出内容完全相同的子图并生成去重报告，支持一键删除重复目录。

```bash
# 仅分析，不删除
python graph_net/agent/scripts/gen_hash_and_dedup.py ./success_20260515_merged

# 分析并直接删除重复目录（保留每组第一个）
python graph_net/agent/scripts/gen_hash_and_dedup.py ./success_20260515_merged --remove
```

输出包括：生成的 `graph_hash.txt` 数量、唯一子图数、重复组数、可删除数量，以及详细的去重报告 `dedup_report.txt`。

> **为什么去重率高**：同一基础模型的微调变体（fine-tune variants）通常共享完全相同的计算图，只是权重不同。实测可缩减 90%+（85K 子图 → 1.5K 唯一子图，2.3 GB → 172 MB）。

**环境变量**：以上脚本默认使用 `$HOME/workspace/` 下的目录，可通过环境变量覆盖：

```bash
export GRAPHNET_LOG_DIR=/your/path/logs_and_lists
export GRAPHNET_SUCCESS_DIR=/your/path/success
export GRAPHNET_SAMPLES_DIR=/your/path/samples
```
