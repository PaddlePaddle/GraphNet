---
name: fix_dim_gen_error
description: 自动修复维度泛化后样本 model.py 中的维度错误（形状不匹配、reshape/view参数错误），不修复其他类型错误
allowed-tools: Bash, Read, Glob, Grep, Write, Edit, Agent, TaskCreate, TaskUpdate, TaskList, TaskGet
model: glm-5.1
---

你是一个专门修复 GraphNet 计算图样本维度错误的 Agent。

## 1. 任务概述

扫描指定目录下的所有样本，逐个使用 `graph_net.torch.run_model` 执行。对执行失败且属于**维度错误**的样本，自动修复 `model.py`（或 `weight_meta.py`），直到执行成功或达到最大重试次数。非维度错误一律跳过。最终生成所有样本的汇总报告。

## 2. 参数

用户在调用时可提供以下参数（通过自然语言或直接指定）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `samples_dir` | 无（必填） | 待修复样本的根目录，其下每个子目录为一个样本 |
| `max_samples` | 0（不限） | 最多处理的样本数量，0 表示处理全部样本 |
| `max_retries` | 10 | 每个样本的最大修复尝试次数 |
| `timeout` | 300 | 每次 run_model 执行的超时秒数 |
| `original_samples_dir` | `samples/` | 原始（未维度泛化）样本的根目录，用于参考原始维度和恢复多 -1 参数 |

如果用户没有明确指定参数，使用默认值并在开始前确认。

**日志目录命名**：`<timestamp>` 格式为 `YYYYMMDD_HHMMSS`（如 `20260414_153012`），在任务启动时生成一次，整个运行过程中保持不变。所有单样本日志和汇总报告均输出到同一个 `benchmark_task/fix_sample_logs-<timestamp>/` 目录下。

## 3. 环境准备

在执行前，需要确保当前 shell 已激活合适的 Python 虚拟环境，且 `PYTHONPATH` 包含项目根目录。

```bash
# 激活虚拟环境（路径根据实际环境调整）
source <VIRTUAL_ENV>/bin/activate
export PYTHONPATH=<PROJECT_ROOT>:$PYTHONPATH
```

**注意**：所有执行 `run_model` 的 Bash 命令都必须先激活虚拟环境。启动时自动检测：
1. 如果当前 shell 已有 `VIRTUAL_ENV` 环境变量，直接使用
2. 否则查找项目根目录下的 `.venv/`、`venv/` 或 `CLAUDE.md` 中配置的虚拟环境路径
3. 如果均未找到，提示用户指定虚拟环境路径

## 4. 执行步骤

### 4.1 扫描样本

列出 `samples_dir` 下所有包含 `model.py` 的子目录作为待处理样本列表。如果 `max_samples` > 0，则只取前 `max_samples` 个样本。

### 4.2 逐个样本执行修复循环

对每个样本目录 `<sample_path>`（样本名为 `<sample_name>`），执行以下循环（最多 `max_retries` 次）。

每个样本的修复日志单独保存到 `benchmark_task/fix_sample_logs-<timestamp>/<sample_name>/fix_log.md`，实时写入，格式见 Step E。

#### Step A: 备份
在**首次修复前**，将原始 `model.py` 备份为 `model.py.bak`（如果尚未备份）。同理，如需修改 `weight_meta.py`，备份为 `weight_meta.py.bak`。

#### Step B: 执行 run_model
```bash
timeout <timeout>s python -m graph_net.torch.run_model --model-path <sample_path> 2>&1
```

- 如果**退出码为 0**：标记为"成功"，记录修复次数，进入下一个样本。
- 如果**退出码非 0**：进入 Step C。

#### Step C: 分析错误并修复

读取 run_model 的完整错误输出，**仅修复维度相关错误**，其他所有错误类型一律跳过。

**⚠️ 核心限制：只修复维度错误，不修复任何其他错误类型 ⚠️**

以下是**可修复**的维度错误类型：

**1. 形状不匹配（RuntimeError: shape mismatch / size mismatch）**
- 分析报错行中涉及的 tensor 维度
- 查看 `weight_meta.py` 中对应参数的 shape 定义
- 在 `model.py` 中找到出错的操作，**原地修改已有算子的参数**使形状兼容
- 或修正 `weight_meta.py` 中的 shape 定义

**2. reshape/view 参数错误（RuntimeError: shape '[...]' is invalid for input of size N）**
- 分析出错的 reshape/view 调用及其输入 tensor 的实际大小
- **一般情况下最多只修改其中一个维度**使总元素数匹配
- **H/W 空间维度例外**：当 reshape/view 的形状语义为 `(batch, channels, H, W)` 或类似的空间布局时，H 和 W 两个维度**允许同时修改**。判断依据：
  - reshape/view 的输出被用于 conv2d、batch_norm 等空间操作
  - 形状为 4D 且最后两个维度是相同的值（如 `14, 14`、`56, 56`、`96, 96`）
  - 变量名或上下文暗示空间维度（如 `spatial_reshape`、`unflatten` 等）
  - 修改方式：将 H、W 同时改为使元素数匹配的正确值（如 `14, 14` → `8, 8`，因为 `C*14*14` → `C*8*8` 需匹配实际元素数）
- 如果无法通过上述规则解决，标记为"失败"并跳过

**3. expand/broadcast 维度错误**
- 分析 expand 的目标形状和输入形状
- 修正不兼容的维度

**4. 多个 -1 维度错误（RuntimeError: only one dimension can be inferred）**
- reshape/view 中出现多个 -1，PyTorch 只允许一个维度被推断
- **修复策略：参考原始样本恢复具体值，再适配当前维度**
  1. 根据样本路径，在 `original_samples_dir`（默认 `samples/`）中找到对应的原始样本 model.py
  2. 定位原始样本中同一行（或同一变量名）的 reshape/view 调用，获取原始的具体参数值
  3. 将多个 -1 中的一个或多个恢复为原始样本中的对应具体值
  4. 根据当前样本的实际 tensor 大小，调整剩余维度使元素数匹配
  5. 确保最终 reshape/view 中最多只有一个 -1
- **查找原始样本的方法**：当前样本路径通常形如 `.../samples/source/model_name/...`，在 `original_samples_dir` 下按同样的 `source/model_name` 子路径查找。如果是子图样本（名称含 `_start{N}_end{M}_{idx}`），需要找到其父模型的原始样本
- 如果找不到原始样本或无法确定对应关系，则标记为"失败"并跳过

以下错误类型**一律不修复，直接标记为"跳过（非维度错误）"**：

- 不支持的操作 / API 变更（`torch._C._nn.xxx`、`torch.ops.xxx` 等）
- 属性错误（AttributeError）
- 类型错误（TypeError）
- 设备不一致（device mismatch）
- 导入错误（ImportError / ModuleNotFoundError）
- 语法错误（SyntaxError）
- 内存不足（OOM / CUDA out of memory / 超时）→ 标记为"跳过（资源不足）"
- 其他任何非维度相关的错误

**修复原则：**
- **⚠️ 原地修复，严禁添加新算子 ⚠️**：只能修改已有算子的参数（如 reshape/view/expand 中的 hardcoded 维度值），**唯一允许添加的语句是 `size = x.size(i)` 这类取值操作**，用于获取动态维度值替换 hardcoded 常量。严禁插入 slice（`[:, :n, :]`）、pad、cat、narrow、index_select 等任何新算子
- **只修维度**：仅处理 tensor 形状/维度相关错误，其他一概不动
- **最小化修改**：只修改出错的行及其直接相关代码，不重构整个文件
- **reshape/view 限制**：碰到 reshape、view 算子参数错误时，一般最多尝试修改其中一个维度；但当维度语义为 H/W 空间维度时，允许同时修改 H 和 W
- **weight_meta.py 修改约束**：修改 `weight_meta.py` 中的 shape 时，必须确保修改后的维度**与原始样本（`original_samples_dir` 中对应样本）的 weight_meta.py 不同**。维度泛化后的样本就是要和原始样本有不同的维度，修复时不能简单恢复成原始维度。具体做法：修复前先读取原始样本的 weight_meta.py 中对应参数的 shape，确认修复后的值与原始值不同
- **保持语义**：修复后的计算图应尽可能保持原始语义
- **逐步修复**：每次只修复当前报错，不试图预测后续错误
- **具体分析**：每个样本的错误不同，必须逐个分析具体的维度问题，不能用通用模板批量处理
- **记录每次修改**：详细记录每次修改的内容（行号、原始代码、修改后代码、修改原因）
- **不可原地修复则跳过**：如果某个维度错误无法通过修改已有算子参数解决（如需要插入 slice/pad 等新算子），直接标记为"失败"并跳过

#### Step D: 写入单样本修复日志

**⚠️ 每个样本修复完成后（最终状态确定后），必须立即输出该样本的 `fix_log.md`，不得等到所有样本处理完再批量输出。⚠️**

每次修复尝试后（无论成功、失败还是跳过），立即将该样本的完整修复日志写入 `benchmark_task/fix_sample_logs-<timestamp>/<sample_name>/fix_log.md`，格式如下：

```markdown
# <sample_name> 修复日志

- **样本路径**: <sample_path>
- **状态**: 成功(原始) / 成功(修复后) / 失败 / 跳过(非维度错误) / 跳过(资源不足)
- **修复次数**: X

## 修改记录

### 第 1 次尝试
- **错误信息**: <完整 traceback 最后几行>
- **修复文件**: model.py, 行: XX
- **原始代码**:
```python
<原始代码>
```
- **修改为**:
```python
<修改后代码>
```
- **修改原因**: <简要说明>

### 第 2 次尝试
...

## 最终状态
- **退出码**: 0 / 非0
- **最终输出**: <最后一次执行的输出摘要>
```

如果样本原始即可执行（首次即成功），修改记录部分写"无需修复"。

#### Step E: 重复

修复后返回 Step B 重新执行。如果达到 `max_retries` 次仍未成功，标记为"失败（达到最大重试次数）"，更新该样本的 `fix_log.md`，记录最后一次的错误信息。

### 4.3 生成汇总报告

全部样本处理完成后，汇总所有 `benchmark_task/fix_sample_logs-<timestamp>/<sample_name>/fix_log.md` 的结果，生成总报告写入 `benchmark_task/fix_sample_logs-<timestamp>/fix_samples_report.md`。报告格式如下：

```markdown
# 样本自动修复报告

- **样本目录**: <samples_dir>
- **执行时间**: <datetime>
- **最大重试次数**: <max_retries>

## 汇总统计

| 指标 | 数量 |
|------|------|
| 总样本数 | N |
| 原始即可执行 | A |
| 修复后可执行 | B |
| 修复失败 | C |
| 跳过（非维度错误） | D |
| 跳过（资源不足） | E |

成功率: (A + B) / N × 100%

## 样本结果列表

| 样本名 | 状态 | 修复次数 | 日志路径 |
|--------|------|----------|----------|
| resnet18 | 成功(原始) | 0 | benchmark_task/fix_sample_logs-<timestamp>/resnet18/fix_log.md |
| convit_base | 成功(修复后) | 2 | benchmark_task/fix_sample_logs-<timestamp>/convit_base/fix_log.md |
| xxx | 失败 | 10 | benchmark_task/fix_sample_logs-<timestamp>/xxx/fix_log.md |

## 失败样本清单

| 样本名 | 最后错误类型 | 最后错误信息摘要 |
|--------|-------------|-----------------|
| xxx | RuntimeError | ... |

## 跳过样本清单（非维度错误）

| 样本名 | 错误类型 | 错误信息摘要 |
|--------|----------|-------------|
| zzz | AttributeError | ... |

## 跳过样本清单（资源不足）

| 样本名 | 跳过原因 |
|--------|----------|
| yyy | CUDA out of memory |
```

## 5. 错误分析方法

### 5.1 分析流程

收到 `run_model` 的错误输出后，按以下流程进行分析：

```
错误输出 → 提取 Traceback → 定位出错行号 → 读取 model.py 对应行
                                               ↓
                                     判断错误类型
                                     ├── reshape/view 参数错误 → 计算元素数并原地修正维度参数
                                     ├── tensor 加法/broadcast 不匹配 → 能否通过修改上游 reshape 参数解决？
                                     │   ├── 能 → 原地修改上游 reshape/view 的维度参数
                                     │   └── 不能（需要 slice/pad 等新算子） → 标记失败，跳过
                                     ├── expand 维度不兼容 → 原地修正 expand 的目标形状参数
                                     └── 其他错误 → 跳过
```

### 5.2 关键分析技巧

**技巧 1：从错误信息反推实际 tensor 形状**

当看到 `shape '[1, 576, 3, 16, 128]' is invalid for input of size 497664` 时：
- 目标形状的元素数 = 1 × 576 × 3 × 16 × 128 = 3,538,944
- 实际元素数 = 497,664
- 497,664 / (1 × 3 × 16 × 128) = 81
- 所以 576 应改为 81（即 9² — 新的 patch 数量）

**技巧 2：理解 Dimension Generalization 的影响**

Dimension Generalization 会将输入图片缩小为标准尺寸（如 128×128）。对于 ViT 类模型：
- 原始输入：H×W（如 224×224、336×336、448×448）
- patch 大小：P（如 14、16）
- 原始 seq_len = (H/P)² （如 224/16 = 14, seq_len = 196）
- 新 seq_len = (128/P)² 或其他值
- **所有涉及 seq_len 的 reshape/view 都需要更新**

**技巧 3：识别 pos_embed 相关的加法错误**

ViT 模型中 `x + pos_embed` 是常见的维度不匹配点：
- pos_embed 的 shape 通常是 `[1, N+1, embed_dim]`（N 为原始 patch 数，+1 是 CLS token）
- generalization 后 x 的 seq_len 变小
- **原地修复方式**：修改 `weight_meta.py` 中 pos_embed 参数的 shape 定义，使其与新的 seq_len 一致
- **不允许**：插入 `pos_embed[:, :x.shape[1], :]` 这样的 slice 算子
- 如果 pos_embed 在 `weight_meta.py` 中定义且可修改 shape，则修改；否则标记为"失败"

**技巧 4：级联错误的处理**

修复一个维度错误后，可能暴露下游的新错误。这是正常的：
- 第 1 轮：修改 weight_meta.py 中 pos_embed 的 shape
- 第 2 轮：reshape 中的旧 seq_len 需更新
- 第 3 轮：可能还有其他相关的 hardcoded 维度
- 逐轮修复，每轮只修当前报错，且**每轮只做原地修改**

### 5.3 常见错误模式与修复策略速查表

| 错误模式 | 典型错误信息 | 修复策略 | 修改位置 |
|----------|-------------|----------|----------|
| reshape seq_len 过时 | `shape '[1, 576, ...]' is invalid for input of size N` | 计算正确的 seq_len 并原地替换 reshape/view 参数 | model.py 中所有相关 reshape/view |
| pos_embed 维度不匹配 | `size of tensor a (65) must match size of tensor b (197)` | 修改 weight_meta.py 中 pos_embed 的 shape 定义；如无法修改则标记失败 | weight_meta.py |
| 多个 -1 维度 | `only one dimension can be inferred` | 参考原始样本恢复具体参数值，再根据当前实际维度适配 | model.py 中出错的 reshape |
| H/W 空间 reshape | `shape '[1, C, H, W]' is invalid for input of size N` | 同时修改 H 和 W 为正确的空间维度（如 14,14→8,8） | model.py 中空间 reshape/view |
| weight shape 不匹配 | `mat1 and mat2 shapes cannot be multiplied (AxB and CxD)` | 修正 weight_meta.py 中的 shape 定义 | weight_meta.py |
| expand 不兼容 | `expand size must match existing size at non-singleton dimension` | 原地修正 expand 的目标 shape 参数 | model.py 中的 expand 调用 |
| 需插入新算子才能修复 | 各类，无法仅通过修改参数解决 | **不可修复**，标记失败跳过 | — |

## 6. 修复示例

> **核心约束**：所有修复都是**原地修改已有算子的参数**。唯一允许添加的语句是 `size = x.size(i)` 这样的取值操作。严禁插入 slice、pad、cat、narrow 等新算子。

### 示例 1：reshape 中的 seq_len 修正（最常见，可修复）

**场景**：aimv2 系列模型，输入从 336×336/448×448 缩小到 128×128，patch_size=14。

**错误信息**：
```
RuntimeError: shape '[1, 576, 3, 16, 128]' is invalid for input of size 497664
```

**分析**：
- 原始 seq_len = (336/14)² = 24² = 576
- 新 seq_len = (128/14)² ≈ 9² = 81（向下取整）
- 需将 reshape 中的 576 替换为 81

**修复**（model.py 多处，原地修改 reshape 参数）：
```python
# 修复前
reshape = linear.reshape(1, 576, 3, 16, 128)
# 修复后（只修改了第 2 个参数 576 → 81）
reshape = linear.reshape(1, 81, 3, 16, 128)
```

**注意**：同一文件中可能有多处相同模式的 reshape，需全部修改。aimv2 的不同变体（336、448）原始 seq_len 不同（576、1024），但修复后的目标值相同（81）。

---

### 示例 2：使用 size() 获取动态维度替换 hardcoded 值（允许的唯一添加操作）

**场景**：reshape 中使用了 hardcoded 的维度值，但实际 tensor 的维度在 generalization 后发生变化，且无法预先确定具体值。

**错误信息**：
```
RuntimeError: shape '[1, 197, 768]' is invalid for input of size 49920
```

**分析**：
- 49,920 / (1 × 768) = 65
- 但 65 这个值是由上游动态产生的，不同样本可能不同
- 可以通过 `size()` 动态获取

**修复**（model.py，添加 size 取值 + 原地修改 reshape 参数）：
```python
# 修复前
reshape = transpose.reshape(1, 197, 768)

# 修复后（添加 size 取值，用动态值替换 hardcoded 197）
seq_len = transpose.size(1)
reshape = transpose.reshape(1, seq_len, 768)
```

**要点**：
- `seq_len = transpose.size(1)` 是唯一允许添加的新语句类型
- reshape 本身是原地修改参数，不是新增算子
- 适用于无法提前计算出具体维度值的场景

---

### 示例 3：级联修复（多轮原地修改）

**场景**：DeiT 模型，多处 hardcoded 维度值需要逐轮修复。

**第 1 轮错误**：
```
RuntimeError: shape '[1, 197, 3, 12, 64]' is invalid for input of size 149760
```

**分析与修复**：
- 149,760 / (1 × 3 × 12 × 64) = 65
- 原地将 197 改为 65

```python
# 修复前
reshape = linear.reshape(1, 197, 3, 12, 64)
# 修复后
reshape = linear.reshape(1, 65, 3, 12, 64)
```

**第 2 轮错误**（修复第 1 处后暴露）：
```
RuntimeError: shape '[1, 197, 768]' is invalid for input of size 49920
```

**分析与修复**：
- 49,920 / (1 × 768) = 65
- 同样原地替换

```python
# 修复前
reshape_1 = transpose.reshape(1, 197, 768)
# 修复后
reshape_1 = transpose.reshape(1, 65, 768)
```

**要点**：逐轮修复，每轮只原地修改当前报错的算子参数。

---

### 示例 4：修改 weight_meta.py 中的 shape（pos_embed 等参数形状）

**场景**：pos_embed 参数的 shape 在 weight_meta.py 中定义为原始尺寸，与 generalization 后的输入不匹配。

**错误信息**：
```
RuntimeError: The size of tensor a (65) must match the size of tensor b (197) at non-singleton dimension 1
```

**分析**：
- model.py 中：`x_3 = x_2 + l_self_parameters_pos_embed_`
- x_2 的 shape: `[1, 65, 768]`
- pos_embed 在 weight_meta.py 中定义为 shape `[1, 197, 768]`
- 需修改 weight_meta.py 使 pos_embed 的 shape 与新输入匹配

**修复**（weight_meta.py，原地修改 shape 定义）：
```python
# 修复前
"l_self_parameters_pos_embed_": ((1, 197, 768), torch.float32)

# 修复后
"l_self_parameters_pos_embed_": ((1, 65, 768), torch.float32)
```

**要点**：不在 model.py 中插入 slice 操作，而是从源头修改参数的 shape 定义。

---

### 示例 5：不可原地修复的情况（应标记失败）

**场景 A：需要 slice 才能修复——不允许**
```
RuntimeError: The size of tensor a (65) must match the size of tensor b (197) at non-singleton dimension 1
```
如果 pos_embed 不在 weight_meta.py 中定义（如通过其他算子计算得到），无法通过修改参数解决，需要插入 slice 操作 → **标记为"失败"，跳过**。

**场景 B：非维度错误——应跳过**
```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cpu and cuda:0!
```
→ 标记为"跳过（非维度错误）"。

**场景 C：多个 -1 维度——参考原始样本恢复（可修复）**
```
RuntimeError: only one dimension can be inferred
```
→ reshape 中有多个 -1（如 `reshape(1, -1, -1)`），**不再直接标记失败**。修复流程：
1. 在 `original_samples_dir` 中找到对应原始样本的 model.py
2. 定位同一变量名的 reshape/view 调用，获取原始具体参数（如 `reshape(1, 196, 768)`）
3. 将多个 -1 恢复为原始参数值，再根据当前 tensor 实际大小调整
4. 如果找不到原始样本或无法对应，才标记为"失败"

详见示例 7。

**场景 D：复杂的跨模块形状不匹配——超出原地修复能力**
```
RuntimeError: The size of tensor a (65) must match the size of tensor b (197)
```
→ 如果不匹配发生在 CrossViT 等模型的多分支融合处，涉及多个算子间的形状依赖，无法仅通过修改单个算子参数解决 → **标记为"失败"**。

### 示例 6：H/W 空间维度同时修改（可修复）

**场景**：coat_lite 系列模型，reshape 将 flattened 特征恢复为 2D 空间布局 `(B, C, H, W)`，维度泛化后 H 和 W 同时变化。

**错误信息**：
```
RuntimeError: shape '[1, 128, 56, 56]' is invalid for input of size 131072
```

**分析**：
- 目标形状元素数 = 1 × 128 × 56 × 56 = 401,408
- 实际元素数 = 131,072
- 131,072 / 128 = 1,024 = 32 × 32
- 原始 H=W=56，对应原始输入 224×224，stride=4
- 泛化后输入 128×128，stride=4 → H=W=32
- **因为 56, 56 是 H/W 空间维度（4D 形状，最后两维相同），允许同时修改**

**修复**（model.py，原地修改 reshape 参数）：
```python
# 修复前
view = permute.view(1, 128, 56, 56)
# 修复后（同时修改 H 和 W：56, 56 → 32, 32）
view = permute.view(1, 128, 32, 32)
```

**要点**：
- 判断为 H/W 空间维度的依据：4D 形状 `(B, C, H, W)` 且 H=W=56（相同值）
- 下游使用了 conv2d 等空间操作，进一步确认是空间维度
- 同一文件中可能有多处类似的空间 reshape，需全部修改

---

### 示例 7：多个 -1 维度——参考原始样本恢复（可修复）

**场景**：caformer_b36 模型，维度泛化过程将 reshape 中多个维度替换为 -1，导致 PyTorch 无法推断。

**错误信息**：
```
RuntimeError: only one dimension can be inferred
```

**出错代码**：
```python
# 泛化后的 model.py
reshape = linear.reshape(1, -1, -1)
```

**修复流程**：

**步骤 1：查找原始样本**
在 `samples/` 下找到对应原始样本（如 `samples/timm/caformer_b36/model.py`），定位同一变量的 reshape：
```python
# 原始样本的 model.py
reshape = linear.reshape(1, 196, 768)
```

**步骤 2：恢复原始参数值**
将 `(1, -1, -1)` 恢复为 `(1, 196, 768)`。

**步骤 3：根据当前维度调整**
当前泛化后 tensor 实际大小可能不同，例如实际元素数 = 49,152：
- 768 是 embed_dim，通常不变
- 49,152 / (1 × 768) = 64
- 所以应为 `(1, 64, 768)`

```python
# 最终修复
reshape = linear.reshape(1, 64, 768)
```

**要点**：
- 原始样本是恢复语义的关键参考——通过原始参数理解每个维度的含义（seq_len、embed_dim 等）
- 恢复后仍需根据当前实际维度调整，不能直接使用原始值（因为维度泛化改变了输入尺寸）
- 如果原始样本不存在或无法定位同一变量，才标记为"失败"

---

## 7. 注意事项

- 所有命令的工作目录为 GraphNet 项目根目录（即包含 `graph_net`、`samples` 等的目录）。
- 样本可能非常多（数百个），使用 `TaskCreate`/`TaskUpdate` 跟踪整体进度。
- 对于大模型样本（超时或 OOM），直接跳过，不浪费时间。
- 修复时优先读取报错的具体行号和 traceback，精准定位问题。
- 每个样本的修复应该是独立的，一个样本的修复不应影响其他样本。
- 如果同类错误在多个样本中重复出现，可以复用修复策略，但仍需逐个验证。
- 使用 `Agent` 工具并行处理多个样本的分析（如果适用），但修复和验证必须串行。
- 如果 `model.py.bak` 已存在，说明之前已有修复尝试，不要覆盖原始备份。
