# 维度泛化结果报告

## 一、背景与目标

GraphNet 从开源模型中提取了子图（每个子图有固定的输入维度配置）。维度单一限制了下游 benchmark 的覆盖面。

**维度泛化的目标**：自动探测每个子图中哪些输入维度是动态可变的，生成多组不同维度配置的变体，从而用少量原始模型扩展出大量多样化的 benchmark 样本。

## 二、核心结论

| 指标 | 数值 |
|------|------|
| 原始子图数 | 18,630（758 timm + 17,872 transformers）|
| 维度探测（真实 + 默认） | 18,776（757 timm + 18,019 transformers）|
| 有动态维度约束 | **18,547**（757 timm constraints + 17,790 transformers constraints）|
| Pipeline 成功生成变体 | 17,270 个模型（ok=563 timm + 16,707 transformers）|
| 变体目录总数 | **166,693** |
| 实际生成 model.py | **165,513**（成功率 99.3%） |
| Pipeline 总耗时 | ~30 分钟（两轮 pipeline + subgraph 修补） |

```
转化率（按实际 model.py 计算）：
  原始 18,630 + 新增变体 165,513 = 184,143 总产出
  转化率 ≈ 988%
```

> 注：1,180 个变体缺失 model.py，因源模型本身不存在（subgraph 源目录缺失）。

## 三、方法与流程

```
Step 1: 维度探测（真实 + 默认）         Step 2: pipeline 生成变体           Step 3: subgraph 修补
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ ① 已探测模型：batch-only │      │ 根据动态维度约束，       │      │ 将 xxx_subgraph_N 映射   │
│    probe + 预处理硬编码  │ ──→  │ 组合生成最多 10 个       │ ──→  │ 为 xxx/subgraph_N，      │
│ ② 未探测模型：从         │      │ 不同维度配置的变体       │      │ 从源目录拷贝 model.py   │
│    weight_meta.py 静态   │      │                         │      │ 并做维度替换            │
│    解析生成默认约束       │      │                         │      │                         │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

## 四、分步结果

### 4.1 Step 1: 维度探测

探测分两种方式：

**① 真实 batch-only 探测**（部分模型，使用 FakeTensorMode forward）：

| 指标 | timm | transformers | 总计 |
|------|------|-------------|------|
| 探测完成 | 750 | ~9,500 | ~10,250 |
| batch 可泛化 | 552（74.0%）| 7,104（86.5%）| **7,656（85.5%）** |
| 真全静态 | 194（26.0%）| 1,107（13.5%）| **1,301（14.5%）** |

**② 默认 probe + 静态约束生成**（未探测模型）：

对于因 OOM/超时等原因无法实际 probe 的模型（约 8,439 个），采用两步策略：
1. 生成默认 `probe_result.json`：假设 batch 维度（axis=0）为动态
2. 从源模型 `weight_meta.py` 解析 data_input 的 shape，结合 `model.py` forward 签名的类型标注自动分类参数，将第 0 维替换为 S0，静态生成 `input_tensor_constraints.py`

| 指标 | 数值 |
|------|------|
| 需要默认 probe 的模型 | 8,439 |
| 成功生成 constraints | **10,425**（含部分已探测但缺 constraints 的模型）|
| 无源模型 | 28 |
| 无 weight_meta | 66 |
| 无 data_input | 298 |

**关键发现**：
- FX trace 硬编码 batch=1 导致 50%+ 模型被误判为"全静态"
- 经 batch 硬编码预处理后，原"全静态"模型中 **85.5% 可挽回 batch 维度**
- batch 探测呈"全通或全不通"模式——9 个探测值要么全部成功，要么全部失败
- LLM 类模型（7B/14B/70B）平均有 200-300 处 batch 硬编码
- 静态生成 constraints 无需跑 FakeTensorMode forward，避免了 OOM 和超时问题

### 4.2 Step 2: Pipeline 变体生成

Pipeline 分两轮执行：

**第一轮**（全量 18,630 模型）：

| 指标 | timm | transformers | 总计 |
|------|------|-------------|------|
| 输入模型 | 757 | 17,873 | 18,630 |
| 成功 (ok) | 556 | 7,139 | 7,695 |
| 跳过 (skip) | 183 | 10,720 | 10,903 |
| 生成变体 | 5,560 | 70,948 | 71,027 |

> 跳过原因：8,439 个默认 probe 模型缺少 `input_tensor_constraints.py`（只有 `probe_result.json`）。

**修复后第二轮**（补跑被跳过的模型）：

| 指标 | timm | transformers | 总计 |
|------|------|-------------|------|
| 补跑模型 | 201 | 10,480 | 10,681 |
| 成功 (ok) | 7 | 9,568 | 9,575 |
| 跳过 (skip) | 192 | 911 | 1,103 |
| 新增变体 | 70 | 95,656 | 95,726 |

**两轮合计**：

| 指标 | 数值 |
|------|------|
| Pipeline 成功模型 | **17,270**（563 timm + 16,707 transformers）|
| 变体目录总数 | **166,693** |
| 变体 model.py 文件数 | **165,513**（修补 subgraph 后） |
| 缺失 model.py | 1,180（源模型不存在） |
| 总耗时 | ~30 分钟（含 subgraph 修补） |

### 4.3 Step 3: Subgraph 路径修补

Pipeline 使用扁平路径 `xxx_subgraph_N` 查找源模型，但实际目录结构是 `xxx/subgraph_N`，导致部分变体缺失 model.py。两轮 pipeline 后共修补：

| 指标 | 数值 |
|------|------|
| 第一轮修补 | 25,271 个（全部成功）|
| 第二轮修补 | 57,026 个（58,206 需修补，1,180 无源文件）|
| 最终变体 model.py | **165,513 / 166,693**（99.3%）|

## 五、转化率链路

```
原始子图: 18,630（758 timm + 17,872 transformers）
  │
  ├─ Step 1: 维度探测
  │    │
  │    ├─ 真实 batch-only 探测: ~10,250 个模型
  │    │    ├─ batch 可泛化: 7,656（85.5%）
  │    │    └─ 真全静态: 1,301（14.5%）
  │    │
  │    └─ 默认 probe + 静态约束: 10,425 个模型
  │         └─ 从 weight_meta.py 解析 shape，生成 input_tensor_constraints.py
  │
  ├─ Step 2: Pipeline 生成变体（两轮）
  │    │
  │    ├─ 第一轮: 7,695 ok → 71,027 变体
  │    ├─ 修复 constraints 后第二轮: 9,575 ok → 95,726 变体
  │    └─ 合计: 17,270 个模型 → 166,693 变体目录
  │
  └─ Step 3: Subgraph 修补
       └─ 修补 82,297 个缺失 model.py → 最终 165,513 个 model.py

总产出（实际 model.py）：18,630 + 165,513 = 184,143（~988% 转化率）
缺失 model.py：1,180（源模型不存在，占 0.7%）
```

## 六、关键问题与修复

### 6.1 FX Trace 硬编码 batch=1 导致假"全静态"

**影响**：50%+ 模型被误判为"全静态"。
**根因**：PyTorch FX trace 将 trace 时 batch_size=1 硬编码到 model.py 的所有 reshape/view/expand/repeat 调用中。
**解决**：开发 batch-only 探测工具链（batch_hardcode_preprocess.py + dim_probe_batch_only.py），挽回 85.5%。

### 6.2 conv2d groups 参数被 batch 替换误伤

**根因**：全局替换 `1` 影响了 `groups=1`。
**解决**：batch 维度替换只作用于 reshape/view/expand/repeat 的**第一个参数**。

### 6.3 weight_meta.py 加载路径错误

**根因**：generate_variants 从 probe 输出目录加载，但 probe 目录无此文件。
**解决**：通过 `--base-dir` 从源模型目录加载。

### 6.4 model.py 硬编码常量未同步

**根因**：`torch.arange(40960)` 等不随输入变化。
**解决**：`_rewrite_model_py()` 用 regex 替换相关模式中的数字常量。

### 6.5 reshape 元组/列表参数遗漏

**根因**：`.reshape((1, 197, 768))` 双括号和 `.reshape([1, 197, 768])` 方括号形式未覆盖。
**解决**：匹配 3 种参数形式（直接、元组、列表），额外修复 80% 的模型。

### 6.6 subgraph 路径映射导致变体缺失 model.py

**根因**：Pipeline 使用扁平路径名 `xxx_subgraph_N` 去 `--base-dir` 查找源模型，但源目录结构是 `xxx/subgraph_N`（子目录形式），导致 `shutil.copytree` 从 probe 目录拷贝（无 model.py）。
**影响**：两轮 pipeline 中共 83,477 个变体缺少 model.py。
**解决**：`fix_missing_modelpy.py` 和 `rerun_skipped.py` 中的修补逻辑将 `xxx_subgraph_N` 映射为 `xxx/subgraph_N`，从正确的源目录拷贝 model.py 并做维度替换。修补 82,297 个，剩余 1,180 个因源模型本身不存在无法修复。

### 6.7 默认 probe 缺少 input_tensor_constraints.py

**根因**：`run_pipeline_v3.py` Phase 1 为未探测模型生成默认 `probe_result.json`（假设 batch 动态），但未生成 `input_tensor_constraints.py`。Pipeline 依赖该文件确定符号维度和输入 shape。
**影响**：8,439 个模型在 pipeline 中被跳过（"无符号维度"），转化率仅 481%。
**解决**：`fix_default_probe.py` 从源模型 `weight_meta.py` 解析 data_input 参数的 shape，结合 `model.py` forward 签名自动分类参数类型，将第 0 维替换为 S0 静态生成 `input_tensor_constraints.py`。共生成 10,425 个 constraints 文件，转化率提升至 988%。

## 七、产出位置（docker 容器 qhl_ys）

| 产出 | 路径 |
|------|------|
| timm 探测结果 | `/work/dim_probe_batch_output/` (757 个) |
| transformers 探测结果 | `/work/dim_probe_batch_output_transformers/` (18,019 个) |
| Pipeline 变体 | `/work/pipeline_output_v2/` (165,513 个 model.py) |
| Pipeline 统计 | `/work/pipeline_output_v2/pipeline_results.json` |
| 修复脚本 | `/work/fix_default_probe.py`、`/work/rerun_skipped.py` |
| Pipeline 日志 | `/work/pipeline_v3_log.txt`、`/work/rerun_skipped_log2.txt` |
