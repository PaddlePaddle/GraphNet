# PyTorch to Paddle 计算图转换设计方案

## 1. 背景与目标

### 1.1 背景
GraphNet 项目包含大量从 PyTorch 提取的计算图样本（位于 `samples/` 目录），需要将这些样本转换为 PaddlePaddle 格式，以扩展 Paddle 样本库并支持跨框架对比分析。

### 1.2 目标
- 使用 PaConvert 工具将 `samples/` 中的 PyTorch 样本批量转换为 Paddle 格式
- 转换后样本保存到 `torch_to_paddle_samples/` 目录
- 对转换后的样本进行批量测试验证
- 生成两个模型列表文件：
  - `torch_to_paddle_samples_list_full.txt`: 仅剔除转换失败的样本
  - `torch_to_paddle_samples_list.txt`: 剔除转换失败和测试失败的样本
- 记录详细的转换和测试日志，分析失败原因

## 2. 样本结构分析

### 2.1 PyTorch 样本文件结构
每个 PyTorch 样本目录包含以下文件：

```
samples/{source}/{model_name}/
├── model.py              # 必需：包含 GraphModule(torch.nn.Module) 类定义
├── graph_net.json        # 必需：元数据配置（framework: "torch"）
├── graph_hash.txt        # 必需：计算图哈希值（用于去重）
├── input_meta.py         # 必需：输入张量元数据（可能为空）
└── weight_meta.py        # 必需：权重张量元数据（包含 torch 类型字符串）
```

### 2.2 关键文件分析

#### 2.2.1 model.py
- **特点**：自动生成的计算图代码，包含大量低级 API 调用
  - 使用 `torch.conv2d()`, `torch.nn.functional.batch_norm()` 等函数式 API
  - 参数名自动生成（如 `L_self_modules_conv1_parameters_weight_`）
  - 代码结构扁平，不是典型的 PyTorch 模型定义
- **转换挑战**：PaConvert 主要针对标准 PyTorch 代码，可能无法完全处理这种格式
- **处理策略**：
  1. 优先使用 PaConvert 自动转换
  2. 对于转换失败或部分转换的情况，需要手动修复或使用 AST 解析进行后处理
  3. 记录所有需要手动修复的案例

#### 2.2.2 weight_meta.py
- **特点**：包含元数据类定义，其中 `dtype` 字段为字符串（如 `"torch.float32"`）
- **需要转换**：`"torch.float32"` → `"paddle.float32"`, `"torch.int64"` → `"paddle.int64"` 等
- **处理策略**：使用正则表达式或 AST 解析进行批量替换

#### 2.2.3 graph_net.json
- **特点**：JSON 配置文件
- **需要修改**：`"framework": "torch"` → `"framework": "paddle"`
- **处理策略**：直接修改 JSON 文件

#### 2.2.4 input_meta.py
- **特点**：可能为空文件或包含输入元数据
- **处理策略**：检查是否包含 torch 相关代码，如有则转换

#### 2.2.5 graph_hash.txt
- **特点**：计算图哈希值
- **处理策略**：转换后重新生成（使用 `graph_net.paddle.validate` 的 `--no-dump-graph-hash-key` 选项）

## 3. 系统架构设计

### 3.1 目录结构

```
tools/torch_to_paddle/
├── __init__.py
├── convert.py              # 转换主脚本
├── test.py                 # 测试主脚本
├── utils.py                # 工具函数模块
├── file_processors.py      # 文件处理模块
└── logs/                   # 日志目录（gitignore）
    ├── conversion/         # 转换日志
    │   ├── {timestamp}_conversion_summary.json
    │   └── {sample_name}_conversion.log
    └── testing/            # 测试日志
        ├── {timestamp}_testing_summary.json
        └── {sample_name}_test.log
```

### 3.2 模块设计

#### 3.2.1 utils.py - 工具函数模块

**功能**：
- 路径处理：相对路径 ↔ 绝对路径转换
- 日志管理：统一日志格式、日志文件命名
- 报告生成：统计转换/测试成功率、失败原因分析
- 列表文件生成：生成模型列表文件

**关键函数**：
```python
def get_torch_samples_list(config_path: str) -> list[str]:
    """从 torch_samples_list.txt 读取所有样本路径"""
    
def normalize_path(path: str, root_dir: str) -> str:
    """标准化路径（相对路径转绝对路径）"""
    
def setup_logging(log_dir: str, sample_name: str, log_type: str) -> logging.Logger:
    """设置日志记录器"""
    
def generate_summary_report(results: list[dict], output_path: str):
    """生成汇总报告（JSON 格式）"""
    
def generate_list_file(sample_paths: list[str], output_path: str):
    """生成模型列表文件"""
```

#### 3.2.2 file_processors.py - 文件处理模块

**功能**：
- 处理不同类型的文件转换
- 提供统一的文件处理接口

**关键函数**：
```python
def convert_model_py(source_path: str, target_path: str, log: logging.Logger) -> dict:
    """
    转换 model.py 文件
    返回: {
        "status": "success|failed|partial",
        "error": "错误信息（如有）",
        "requires_manual_fix": bool
    }
    """
    
def convert_weight_meta_py(source_path: str, target_path: str, log: logging.Logger) -> dict:
    """转换 weight_meta.py 文件（替换 torch 类型字符串）"""
    
def convert_input_meta_py(source_path: str, target_path: str, log: logging.Logger) -> dict:
    """转换 input_meta.py 文件（如需要）"""
    
def convert_graph_net_json(source_path: str, target_path: str, log: logging.Logger) -> dict:
    """修改 graph_net.json（framework: torch -> paddle）"""
    
def copy_other_files(source_dir: str, target_dir: str, log: logging.Logger):
    """复制其他文件（如 graph_hash.txt，转换后需要重新生成）"""
```

#### 3.2.3 convert.py - 转换主脚本

**功能**：
- 读取 `torch_samples_list.txt` 获取所有样本路径
- 对每个样本调用文件处理模块进行转换
- 记录转换日志和失败案例
- 生成转换报告和 `torch_to_paddle_samples_list_full.txt`

**工作流程**：
```
1. 读取配置和样本列表
2. 创建输出目录结构
3. 对每个样本：
   a. 创建目标目录
   b. 转换 model.py（使用 PaConvert + 后处理）
   c. 转换 weight_meta.py
   d. 转换 input_meta.py（如需要）
   e. 修改 graph_net.json
   f. 复制其他文件
   g. 记录转换结果
4. 生成汇总报告
5. 生成 torch_to_paddle_samples_list_full.txt
```

**命令行接口**：
```bash
python -m tools.torch_to_paddle.convert \
    --torch-samples-list graph_net/config/torch_samples_list.txt \
    --output-dir torch_to_paddle_samples \
    --log-dir tools/torch_to_paddle/logs/conversion \
    [--paconvert-path /path/to/paconvert] \
    [--parallel-workers 4] \
    [--dry-run]
```

#### 3.2.4 test.py - 测试主脚本

**功能**：
- 读取转换后的样本目录
- 对每个样本运行 Paddle 验证测试
- 记录测试日志和失败案例
- 生成测试报告和 `torch_to_paddle_samples_list.txt`

**工作流程**：
```
1. 扫描转换后的样本目录
2. 对每个样本：
   a. 运行 graph_net.paddle.validate
   b. 捕获测试结果（成功/失败）
   c. 记录测试日志
3. 生成汇总报告
4. 生成 torch_to_paddle_samples_list.txt（剔除失败样本）
```

**命令行接口**：
```bash
python -m tools.torch_to_paddle.test \
    --samples-dir torch_to_paddle_samples \
    --log-dir tools/torch_to_paddle/logs/testing \
    [--parallel-workers 4] \
    [--timeout 300] \
    [--skip-validation]
```

## 4. 详细设计

### 4.1 PaConvert 集成策略

#### 4.1.1 使用方式
PaConvert 支持两种使用方式：
1. **命令行方式**：`paconvert --in_dir <input> --out_dir <output>`
2. **Python API**：`from paconvert import convert`（需要确认 API 是否存在）

**优先策略**：
- 优先使用命令行方式（更稳定）
- 如果命令行不支持单文件转换，则：
  1. 创建临时目录
  2. 复制 `model.py` 到临时目录
  3. 调用 PaConvert 转换整个目录
  4. 复制转换后的文件到目标位置

#### 4.1.2 转换后处理
由于 `model.py` 的特殊格式，转换后可能需要后处理：

1. **类名检查**：确保 `torch.nn.Module` → `paddle.nn.Layer`
2. **导入语句**：确保 `import torch` → `import paddle`
3. **API 映射验证**：检查关键 API 是否正确转换
4. **语法检查**：使用 `ast.parse()` 验证语法正确性

#### 4.1.3 失败处理策略
- **完全失败**：记录错误，标记为转换失败
- **部分转换**：尝试自动修复常见问题，如修复失败则标记为需要手动修复
- **手动修复标记**：在日志中明确标记需要人工介入的样本

### 4.2 文件转换详细设计

#### 4.2.1 model.py 转换流程

```python
def convert_model_py(source_path, target_path, log):
    """
    1. 检查源文件是否存在
    2. 创建临时工作目录
    3. 调用 PaConvert 转换
    4. 检查转换结果：
       - 语法检查（ast.parse）
       - 类名检查（GraphModule 应为 paddle.nn.Layer）
       - 导入检查（应导入 paddle 而非 torch）
    5. 后处理修复（如需要）：
       - 修复常见的 API 映射问题
       - 修复类型注解
    6. 保存到目标位置
    7. 返回转换状态
    """
```

#### 4.2.2 weight_meta.py 转换流程

```python
def convert_weight_meta_py(source_path, target_path, log):
    """
    1. 读取源文件内容
    2. 使用正则表达式替换：
       - "torch.float32" → "paddle.float32"
       - "torch.float64" → "paddle.float64"
       - "torch.int32" → "paddle.int32"
       - "torch.int64" → "paddle.int64"
       - "torch.bool" → "paddle.bool"
       - "cuda:0" → "gpu:0"（如需要）
    3. 保存到目标位置
    """
```

**类型映射表**：
```python
TORCH_TO_PADDLE_TYPE_MAP = {
    "torch.float32": "paddle.float32",
    "torch.float64": "paddle.float64",
    "torch.int32": "paddle.int32",
    "torch.int64": "paddle.int64",
    "torch.int8": "paddle.int8",
    "torch.uint8": "paddle.uint8",
    "torch.bool": "paddle.bool",
    "torch.complex64": "paddle.complex64",
    "torch.complex128": "paddle.complex128",
}
```

#### 4.2.3 graph_net.json 修改流程

```python
def convert_graph_net_json(source_path, target_path, log):
    """
    1. 读取 JSON 文件
    2. 修改 "framework" 字段：torch → paddle
    3. 保存到目标位置
    """
```

### 4.3 测试验证设计

#### 4.3.1 测试流程
使用现有的 `graph_net.paddle.validate` 模块进行验证：

```python
def test_single_sample(sample_path, log):
    """
    1. 构建验证命令：
       python -m graph_net.paddle.validate --model-path {sample_path}
    2. 执行命令并捕获输出
    3. 解析输出判断成功/失败
    4. 记录测试日志
    5. 返回测试结果
    """
```

#### 4.3.2 错误分类
- **编译错误**：语法错误、导入错误
- **运行时错误**：执行时异常
- **验证失败**：graph_hash 不匹配等
- **超时**：测试执行超时

### 4.4 日志和报告设计

#### 4.4.1 日志格式

**转换日志**（JSON 格式）：
```json
{
    "sample_path": "samples/torchvision/resnet18",
    "conversion_status": "success|failed|partial|manual_fix_required",
    "conversion_timestamp": "2024-01-01T12:00:00",
    "files_converted": {
        "model.py": {"status": "success", "error": null},
        "weight_meta.py": {"status": "success", "error": null},
        "graph_net.json": {"status": "success", "error": null}
    },
    "conversion_errors": [],
    "requires_manual_fix": false,
    "manual_fix_reason": null
}
```

**测试日志**（JSON 格式）：
```json
{
    "sample_path": "torch_to_paddle_samples/torchvision/resnet18",
    "test_status": "success|failed|timeout",
    "test_timestamp": "2024-01-01T12:00:00",
    "test_duration": 5.2,
    "test_errors": [],
    "validation_output": "..."
}
```

#### 4.4.2 汇总报告

**转换汇总报告**：
```json
{
    "summary": {
        "total_samples": 2478,
        "conversion_success": 2000,
        "conversion_failed": 300,
        "conversion_partial": 150,
        "manual_fix_required": 28,
        "success_rate": 0.807
    },
    "failure_analysis": {
        "paconvert_errors": 200,
        "syntax_errors": 50,
        "api_mapping_errors": 30,
        "other_errors": 20
    },
    "failed_samples": [...],
    "manual_fix_samples": [...]
}
```

## 5. 实施计划

### 5.1 阶段 1：探索验证（1-2 天）
**目标**：验证 PaConvert 对计算图代码的转换能力

**任务**：
1. 安装和配置 PaConvert
2. 选择 5-10 个代表性样本进行手动转换测试
3. 记录转换问题和解决方案
4. 确定需要转换的文件范围
5. 评估自动转换的成功率

**输出**：
- 验证报告
- 问题清单
- 转换策略调整建议

### 5.2 阶段 2：核心功能实现（3-5 天）
**目标**：实现基本的转换和测试功能

**任务**：
1. 实现 `file_processors.py` 模块
2. 实现 `convert.py` 主脚本
3. 实现 `test.py` 主脚本
4. 实现 `utils.py` 工具函数
5. 实现日志和报告生成功能

**输出**：
- 可运行的转换脚本
- 可运行的测试脚本
- 初步的转换和测试结果

### 5.3 阶段 3：完善和优化（2-3 天）
**目标**：处理边界情况，优化错误处理

**任务**：
1. 处理各种边界情况
2. 优化错误处理和日志记录
3. 实现并行处理（如需要）
4. 完善报告生成
5. 生成模型列表文件

**输出**：
- 完善的转换和测试工具
- 完整的转换和测试报告
- 两个模型列表文件

### 5.4 阶段 4：文档和提交（1 天）
**目标**：完善文档，准备 PR

**任务**：
1. 完善设计文档
2. 编写使用说明
3. 准备 PR 描述
4. 整理转换和测试结果

**输出**：
- 完整的设计文档
- PR 描述和结果报告

## 6. 风险与应对

### 6.1 技术风险

#### 风险 1：PaConvert 无法处理计算图代码格式
**概率**：中等  
**影响**：高  
**应对**：
- 阶段 1 进行充分验证
- 如果自动转换失败率高，考虑：
  1. 使用 AST 解析进行预处理
  2. 开发自定义转换规则
  3. 标记需要手动转换的样本

#### 风险 2：转换后代码无法通过测试
**概率**：中等  
**影响**：中  
**应对**：
- 实现详细的错误分类和记录
- 提供手动修复指南
- 对于常见问题，实现自动修复规则

#### 风险 3：测试时间过长
**概率**：高  
**影响**：中  
**应对**：
- 实现并行测试
- 设置超时机制
- 支持断点续传（记录已测试样本）

### 6.2 数据风险

#### 风险 4：样本数量大，处理时间长
**概率**：高  
**影响**：中  
**应对**：
- 实现进度显示
- 支持分批处理
- 实现断点续传功能

## 7. 成功标准

### 7.1 转换阶段
- 转换成功率 > 70%（自动转换）
- 所有转换结果都有详细日志
- 失败案例有明确的错误分类

### 7.2 测试阶段
- 测试成功率 > 60%（在转换成功的样本中）
- 所有测试结果都有详细日志
- 失败案例有明确的错误分类

### 7.3 文档和交付
- 完整的设计文档
- 清晰的使用说明
- 详细的转换和测试报告
- 两个模型列表文件格式正确

## 8. 后续工作

### 8.1 优化方向
1. **提高转换成功率**：
   - 分析失败案例，开发更多自动修复规则
   - 与 PaConvert 团队沟通，改进工具支持

2. **提高测试成功率**：
   - 分析测试失败原因
   - 修复常见的兼容性问题

3. **性能优化**：
   - 实现更高效的并行处理
   - 优化文件 I/O

### 8.2 维护计划
1. 定期更新 PaConvert 版本
2. 收集用户反馈，持续改进
3. 维护转换规则和修复规则

## 9. 附录

### 9.1 相关工具和文档
- PaConvert 官方文档：https://github.com/PaddlePaddle/PaConvert
- GraphNet 贡献指南：`docs/CONTRIBUTE_TUTORIAL.md`
- Paddle 验证工具：`graph_net/paddle/validate.py`

### 9.2 参考实现
- 现有的测试脚本：`graph_net/paddle/test_target_device.py`
- 样本检查工具：`tools/check_and_count_samples.py`

