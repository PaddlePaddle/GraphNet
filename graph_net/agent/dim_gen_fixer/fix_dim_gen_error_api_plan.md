# fix_dim_gen_error Python API 版本 - 实现方案

## Context

将 `.claude/commands/fix_dim_gen_error.md`（一个 Claude Code slash command）转换为独立的 Python 脚本，通过 Anthropic API (Messages API + Tool Use) 调用 Claude 来实现智能维度错误分析和修复，而非依赖 Claude Code CLI。

## 核心思路

**分工**：
- **Python 脚本**负责：样本扫描、模型执行、备份、重试循环、日志记录、汇总报告
- **Claude API (Tool Use)** 负责：错误分析、判断是否为维度错误、读取文件、生成修复代码、编辑文件

这样既保持了 Claude 的智能分析能力，又通过 Python 控制流程更加可靠和高效。

## 文件结构

创建一个独立脚本：

```
benchmark_task/fix_dim_gen_error_api.py   # 主脚本
```

日志输出目录与原 command 保持一致：`benchmark_task/fix_sample_logs-<timestamp>/`

## 架构设计

### 1. CLI 参数（与原 command 一致）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--samples-dir` | 无（必填之一） | 样本根目录 |
| `--sample-list-with-status` | 无 | 带状态的样本列表文件 |
| `--max-samples` | 0 | 最多处理样本数 |
| `--max-retries` | 10 | 每样本最大修复尝试次数 |
| `--timeout` | 600 | run_model 超时秒数 |
| `--original-samples-dir` | `samples/` | 原始样本目录 |
| `--model` | `claude-sonnet-4-6` | Claude 模型名 |
| `--api-key` | 环境变量 `ANTHROPIC_API_KEY` | API Key |

### 2. Claude API Tool 定义

提供给 Claude 的工具：

| 工具名 | 功能 |
|--------|------|
| `read_file` | 读取文件内容（支持 offset/limit） |
| `edit_file` | 编辑文件（old_string → new_string，精确匹配替换） |
| `grep_content` | 在文件中搜索正则匹配内容 |
| `finish_fix` | 信号：修复完成，可以重新验证 |
| `skip_sample` | 信号：跳过当前样本（非维度错误/无法修复） |

### 3. Agent Loop（核心流程）

```
对每个样本:
  for retry in range(max_retries):
    1. Python: 执行 run_model，获取输出
    2. 若成功 → 记录结果，跳出
    3. 若失败 → 调用 Claude API:
       - System Prompt: 角色定义 + 修复规则 + 错误分析技巧 + 示例
       - User Message: 错误输出 + 样本路径 + 原始样本目录
       - Tools: read_file, edit_file, grep_content, finish_fix, skip_sample
       - 循环处理 Claude 的 tool_call，直到 finish_fix 或 skip_sample
    4. Claude 返回修改记录 → Python 记录日志
    5. 回到步骤 1 重新验证
```

### 4. System Prompt 策略

从 `fix_dim_gen_error.md` 提取关键内容作为 System Prompt：
- 角色定义（专门修复维度错误）
- 可修复的维度错误类型（4类）
- 不可修复的错误类型
- 修复原则（原地修复、最小化修改等）
- 错误分析方法（技巧1-4）
- 常见错误模式速查表
- 修复示例（7个）

不包含：扫描步骤、日志格式、工具脚本等由 Python 处理的部分。

### 5. 类设计

```python
class ToolHandler:
    """处理 Claude 的 tool_call，操作本地文件系统"""

class ClaudeFixAgent:
    """封装 Anthropic API 调用和 Agent Loop"""

class ModelRunner:
    """执行 run_model 并捕获输出"""

class FixLogger:
    """写入单样本 fix_log.md 和汇总报告"""

class SampleProcessor:
    """单个样本的完整修复流程（重试循环）"""

class MainOrchestrator:
    """主流程：扫描 → 逐样本处理 → 汇总报告"""
```

### 6. sample_list_with_status 文件处理

- 读取：解析 `STATUS<TAB>路径` 格式，过滤 FAILED 样本
- 更新：每个样本修复完成后原地更新状态（PASS/FAILED）
- 与原 command 行为完全一致

## 关键实现细节

1. **文件大小限制**：read_file 超过 500KB 的文件拒绝读取，防止 token 溢出
2. **Tool 迭代上限**：每次 fix 尝试最多 30 轮 tool call，防止死循环
3. **API 错误处理**：捕获 rate limit、API error，自动重试（指数退避）
4. **错误输出截断**：发送给 Claude 的错误输出只保留最后 5000 字符
5. **备份机制**：首次修复前备份 model.py → model.py.bak（不覆盖已有备份）
6. **日志实时写入**：每个样本完成后立即写 fix_log.md，不批量

## 验证方法

```bash
# 1. 安装依赖
pip install anthropic

# 2. 设置 API Key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. 小规模测试（1个样本，2次重试）
python benchmark_task/fix_dim_gen_error_api.py \
  --samples-dir /path/to/samples/timm \
  --max-samples 1 \
  --max-retries 2

# 4. 检查日志
cat benchmark_task/fix_sample_logs-*/fix_samples_report.md
cat benchmark_task/fix_sample_logs-*/*/fix_log.md

# 5. 验证修复后的样本
python -m graph_net.torch.run_model --model-path <sample_path>
```
