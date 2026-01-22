# GraphNet 自动样本抽取 Agent 设计文档

## 1. 任务背景

为了丰富 GraphNet 的样本库，我们需要从 Hugging Face (HF) 上自动下载模型，并将其转换为 GraphNet 可用的子图样本。目前这一过程需要人工编写 `run_model.py` 代码，效率较低。本 Agent 旨在自动化这一流程，实现从"HF 模型链接"到"GraphNet 样本提交"的端到端自动化。

## 2. 核心架构

Agent 采用模块化设计，主要包含以下组件：

### 2.1 架构图

```mermaid
graph TD
    User[用户输入: HF Model ID] --> Manager[Agent Manager]
    Manager --> Fetcher[Model Fetcher]
    Fetcher -- 下载模型 --> Local[本地模型文件]
    Manager --> Analyzer[Model Analyzer]
    Analyzer -- 分析 config.json --> Meta[模型元数据(Input Shape/Dtype)]
    Manager --> Coder[Code Generator]
    Meta --> Coder
    Coder -- 生成代码 --> Script[run_model.py]
    Manager --> Extractor[Graph Extractor]
    Script --> Extractor
    Extractor -- 运行 & 抽图 --> Sample[GraphNet Sample]
    Manager --> Verifier[Sample Verifier]
    Sample --> Verifier
    Verifier -- 验证通过 --> Git[Git Submitter]
```

### 2.2 模块说明

#### 2.2.1 Model Fetcher (`agent.fetcher`)
- **功能**: 调用 `huggingface_hub` 下载模型快照。
- **输入**: `model_id` (e.g., `bert-base-uncased`)
- **输出**: 本地路径。

#### 2.2.2 Model Analyzer (`agent.analyzer`)
- **功能**: 解析模型目录下的 `config.json` 或 `README.md`。
- **目标**: 推断模型的 `input_shape` 和 `input_dtype`。例如 BERT 通常需要 `input_ids` [batch, seq_len] (int64)。

#### 2.2.3 Code Generator (`agent.codegen`)
- **功能**: 生成 `run_model.py`。
- **策略**（按优先级）:
    1. **LLM Mode（优先）**: 调用 LLM API（DeepSeek/OpenAI）基于模型配置生成代码，灵活且覆盖广。
    2. **通用模板 Mode（Fallback）**: 使用单一通用模板 + 配置驱动，通过参数化支持不同模型类型，避免为每个架构单独写模板。
    3. **手动模板 Mode（特殊场景）**: 仅针对极少数特殊模型提供手动模板。

#### 2.2.4 Graph Extractor (`agent.extractor`)
- **功能**: 在子进程中运行生成的 `run_model.py`。
- **依赖**: 复用 `graph_net.torch.run_model` 或直接调用脚本。

#### 2.2.5 Sample Verifier (`agent.verifier`)
- **功能**: 检查生成的 `graph_net.json`, `model.py`, `input_meta.py` 是否存在且合法。

## 3. 接口设计

### 3.1 核心接口：`GraphNetAgent` 类

`GraphNetAgent` 是 Agent 的主控制器，负责协调各个模块完成端到端的样本抽取流程。

```python
from graph_net.agent.core import GraphNetAgent
from graph_net.agent.fetcher import HFFetcher
from graph_net.agent.analyzer import ConfigAnalyzer
from graph_net.agent.codegen import LLMCodeGenerator, TemplateCodeGenerator
from graph_net.agent.extractor import SubprocessExtractor
from graph_net.agent.verifier import BasicVerifier

class GraphNetAgent:
    """
    GraphNet 自动样本抽取 Agent 主类
    
    职责：
    - 协调各模块执行完整的样本抽取流程
    - 管理错误处理和重试逻辑
    - 记录执行日志
    """
    def __init__(self, workspace: str, hf_token: str = None):
        """
        Args:
            workspace: 工作空间根目录路径
            hf_token: HuggingFace API Token（可选，用于访问私有模型）
        """
        self.workspace = workspace
        self.fetcher = HFFetcher(token=hf_token, cache_dir=f"{workspace}/models")
        self.analyzer = ConfigAnalyzer()
        # 优先使用 LLM 生成，Fallback 到通用模板
        self.coder = self._init_code_generator(workspace)
        self.extractor = SubprocessExtractor(workspace=workspace)
        self.verifier = BasicVerifier()
    
    def _init_code_generator(self, workspace: str):
        """初始化代码生成器：优先 LLM，Fallback 模板"""
        try:
            # 优先使用 LLM 生成器
            return LLMCodeGenerator(provider="deepseek")
        except Exception:
            # Fallback 到通用模板生成器
            template_path = f"{workspace}/config/templates/run_model_template.py.j2"
            return TemplateCodeGenerator(template_path=template_path)

    def run(self, model_id: str) -> bool:
        """
        执行完整的样本抽取流程
        
        Args:
            model_id: HuggingFace 模型 ID (e.g., "bert-base-uncased")
            
        Returns:
            bool: 是否成功生成并验证样本
            
        Raises:
            ModelFetchError: 模型下载失败
            AnalysisError: 模型分析失败
            CodeGenError: 代码生成失败
            ExtractionError: 图提取失败
            VerificationError: 样本验证失败
        """
        try:
            # 1. Download: 下载模型到本地
            model_dir = self.fetcher.download(model_id)
            
            # 2. Analyze: 分析模型配置，提取元数据
            meta_info = self.analyzer.analyze(model_dir)
            
            # 3. Generate Code: 生成 run_model.py
            code_path = self.coder.generate(
                model_dir=model_dir,
                meta_info=meta_info,
                output_dir=f"{self.workspace}/generated/{model_id}"
            )
            
            # 4. Extract: 运行脚本提取计算图
            output_dir = self.extractor.extract(
                code_path=code_path,
                model_id=model_id
            )
            
            # 5. Verify: 验证生成的样本
            is_valid = self.verifier.verify(output_dir)
            
            return is_valid
            
        except Exception as e:
            # 错误处理和日志记录
            self._handle_error(model_id, e)
            return False
```

### 3.2 模块接口定义

#### 3.2.1 Fetcher 接口

```python
from abc import ABC, abstractmethod

class BaseFetcher(ABC):
    """模型下载器基类"""
    
    @abstractmethod
    def download(self, model_id: str) -> str:
        """
        下载模型到本地
        
        Returns:
            str: 本地模型目录路径
        """
        pass
```

#### 3.2.2 Analyzer 接口

```python
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    input_shapes: Dict[str, List[int]]  # {"input_ids": [batch, seq_len]}
    input_dtypes: Dict[str, str]        # {"input_ids": "int64"}
    model_type: str                     # "bert", "resnet", etc.

class BaseAnalyzer(ABC):
    """模型分析器基类"""
    
    @abstractmethod
    def analyze(self, model_dir: str) -> ModelMetadata:
        """
        分析模型配置，提取元数据
        
        Returns:
            ModelMetadata: 模型元数据对象
        """
        pass
```

#### 3.2.3 CodeGenerator 接口

代码生成采用**LLM 优先 + 通用模板 Fallback** 的策略，避免为每个架构单独写模板：

```python
class BaseCodeGenerator(ABC):
    """代码生成器基类"""
    
    @abstractmethod
    def generate(self, model_dir: str, meta_info: ModelMetadata, 
                 output_dir: str) -> str:
        """
        生成 run_model.py 脚本
        
        Args:
            model_dir: 模型目录路径
            meta_info: 模型元数据（包含 input_shapes, input_dtypes 等）
            output_dir: 输出目录
            
        Returns:
            str: 生成的脚本文件路径
        """
        pass

# 实现示例 1：LLM 生成器（优先使用，工程量最小）
class LLMCodeGenerator(BaseCodeGenerator):
    """
    使用 LLM 生成代码，灵活且覆盖广
    
    优势：
    - 无需为每个架构写模板，工程量最小
    - 能处理各种模型变体和特殊场景
    - 自动适配不同的输入格式
    - 只需维护 prompt 模板，而非代码模板
    """
    def __init__(self, provider: str = "deepseek", api_key: str = None):
        self.provider = self._get_provider(provider, api_key)
    
    def generate(self, model_dir: str, meta_info: ModelMetadata, 
                 output_dir: str) -> str:
        # 读取模型配置文件
        config = self._load_config(model_dir)
        
        # 构建 prompt（包含模型配置和元数据）
        prompt = f"""
        请为 HuggingFace 模型生成 run_model.py 脚本。
        
        模型信息：
        - Model ID: {meta_info.model_id}
        - Model Type: {meta_info.model_type}
        - Config: {config}
        - Input Shapes: {meta_info.input_shapes}
        - Input Dtypes: {meta_info.input_dtypes}
        
        要求：
        1. 使用 transformers 或 torchvision 加载模型
        2. 根据 input_shapes 和 input_dtypes 构造输入
        3. 使用 graph_net.torch.extract 包装模型并运行
        4. 参考 graph_net/torch/run_model.py 的结构
        """
        
        # 调用 LLM 生成代码
        code = self.provider.generate(prompt)
        
        # 保存到文件
        code_path = f"{output_dir}/run_model.py"
        with open(code_path, 'w') as f:
            f.write(code)
        return code_path

# 实现示例 2：通用模板生成器（Fallback，不依赖外部 API）
class TemplateCodeGenerator(BaseCodeGenerator):
    """
    使用单一通用模板 + 配置驱动生成代码
    
    优势：
    - 不依赖外部 API，稳定可靠
    - 只需维护一个模板文件
    - 通过参数化支持多种模型类型
    """
    def __init__(self, template_path: str):
        from jinja2 import Template
        with open(template_path, 'r') as f:
            self.template = Template(f.read())
    
    def generate(self, model_dir: str, meta_info: ModelMetadata, 
                 output_dir: str) -> str:
        # 根据元数据自动推断模型加载代码
        load_code = self._infer_model_loader(meta_info, model_dir)
        # 根据 input_shapes 和 input_dtypes 生成输入构造代码
        input_code = self._generate_input_code(meta_info)
        
        # 使用 Jinja2 渲染通用模板（只需一个模板文件）
        code = self.template.render(
            model_id=meta_info.model_id,
            model_type=meta_info.model_type,
            model_load_code=load_code,
            input_code=input_code,
            input_shapes=meta_info.input_shapes,
            input_dtypes=meta_info.input_dtypes,
        )
        
        code_path = f"{output_dir}/run_model.py"
        with open(code_path, 'w') as f:
            f.write(code)
        return code_path
    
    def _infer_model_loader(self, meta_info: ModelMetadata, model_dir: str) -> str:
        """根据模型类型自动推断加载代码"""
        if meta_info.model_type in ['bert', 'gpt', 't5']:
            return f'model = AutoModel.from_pretrained("{model_dir}")'
        elif meta_info.model_type in ['resnet', 'vgg', 'densenet']:
            return f'model = torchvision.models.{meta_info.model_type}(pretrained=True)'
        else:
            # 通用加载方式
            return f'model = AutoModel.from_pretrained("{model_dir}")'
```

**设计说明**：
- **优先使用 LLM**：覆盖所有模型类型，无需维护多个模板
- **通用模板作为 Fallback**：单一模板 + 参数化，支持常见场景
- **工程量对比**：
  - 原方案：需要为 N 个架构写 N 个模板（BERT、ResNet、GPT、T5...）
  - 新方案：只需 1 个 LLM prompt 模板 + 1 个通用代码模板

#### 3.2.4 Extractor 接口

```python
class BaseExtractor(ABC):
    """图提取器基类"""
    
    @abstractmethod
    def extract(self, code_path: str, model_id: str) -> str:
        """
        运行脚本提取计算图
        
        Returns:
            str: 输出样本目录路径
        """
        pass
```

#### 3.2.5 Verifier 接口

```python
class BaseVerifier(ABC):
    """样本验证器基类"""
    
    @abstractmethod
    def verify(self, sample_dir: str) -> bool:
        """
        验证样本是否合法
        
        Returns:
            bool: 验证是否通过
        """
        pass
```

## 4. 目录结构设计

### 4.1 模块组织原则

Agent 模块采用**职责单一、接口清晰**的设计原则：
- **核心模块**：负责主要业务流程（下载、分析、生成、提取、验证）
- **工具模块**：提供通用工具类和辅助函数
- **配置模块**：管理模板、配置等静态资源
- **接口抽象**：通过基类定义统一接口，支持多种实现策略

### 4.2 完整目录结构

```text
graph_net/
└── agent/                          # Agent 核心模块包
    ├── __init__.py                 # 包初始化，导出主要接口
    │
    ├── core.py                     # Agent 核心控制器
    │                               # - GraphNetAgent 主类
    │                               # - 协调各模块执行流程
    │                               # - 错误处理和日志记录
    │
    ├── fetcher/                    # 模型下载模块
    │   ├── __init__.py
    │   ├── base.py                 # Fetcher 基类接口
    │   ├── hf_fetcher.py           # HuggingFace 模型下载实现
    │   └── cache.py                # 本地缓存管理
    │
    ├── analyzer/                   # 模型分析模块
    │   ├── __init__.py
    │   ├── base.py                 # Analyzer 基类接口
    │   ├── config_analyzer.py      # 基于 config.json 的分析器
    │   ├── readme_analyzer.py      # 基于 README.md 的分析器
    │   └── model_metadata.py       # 模型元数据数据结构定义
    │
    ├── codegen/                    # 代码生成模块
    │   ├── __init__.py
    │   ├── base.py                 # CodeGenerator 基类接口
    │   ├── llm_generator.py        # LLM 代码生成（优先策略）
    │   │                           # - 基于模型配置和元数据生成代码
    │   │                           # - 支持 DeepSeek/OpenAI 等提供商
    │   ├── template_generator.py  # 通用模板生成（Fallback 策略）
    │   │                           # - 单一通用模板 + 配置驱动
    │   │                           # - 通过参数化支持不同模型类型
    │   └── providers/              # LLM 提供商实现
    │       ├── __init__.py
    │       ├── base_provider.py    # LLM 提供商基类
    │       ├── deepseek_provider.py
    │       └── openai_provider.py
    │
    ├── extractor/                  # 图提取模块
    │   ├── __init__.py
    │   ├── base.py                 # Extractor 基类接口
    │   ├── subprocess_extractor.py # 子进程执行提取
    │   └── runner.py               # run_model.py 执行器
    │
    ├── verifier/                   # 样本验证模块
    │   ├── __init__.py
    │   ├── base.py                 # Verifier 基类接口
    │   ├── file_verifier.py        # 文件存在性验证
    │   ├── schema_verifier.py      # JSON Schema 验证
    │   └── graph_verifier.py       # 计算图合法性验证
    │
    ├── utils/                      # 工具函数模块
    │   ├── __init__.py
    │   ├── workspace.py            # 工作空间管理
    │   ├── logger.py               # 日志工具
    │   └── exceptions.py           # 自定义异常类
    │
    └── config/                     # 配置和模板资源
        ├── templates/              # 代码生成模板目录
        │   ├── run_model_template.py.j2  # Jinja2 模板
        │   └── ...
        └── schemas/                # JSON Schema 定义
            └── graph_net_schema.json
```

### 4.3 工作空间结构

Agent 运行时会在指定工作空间创建以下目录结构：

```text
{workspace}/
├── models/                         # 下载的模型缓存目录
│   └── {model_id}/                 # 按模型 ID 组织
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
│
├── generated/                      # 生成的代码目录
│   └── {model_id}/
│       └── run_model.py            # 生成的运行脚本
│
├── samples/                        # 提取的样本输出目录
│   └── {model_id}/
│       ├── model.py                # 提取的计算图代码
│       ├── graph_net.json          # 元数据配置
│       ├── input_meta.py           # 输入元数据
│       └── graph_hash.txt          # 图哈希值
│
└── logs/                           # 日志目录
    └── {model_id}_{timestamp}.log
```

### 4.4 模块依赖关系

```
core.py (GraphNetAgent)
    ├── fetcher/        # 模型下载
    ├── analyzer/       # 模型分析（依赖 fetcher 输出）
    ├── codegen/        # 代码生成（依赖 analyzer 输出）
    ├── extractor/      # 图提取（依赖 codegen 输出）
    └── verifier/       # 样本验证（依赖 extractor 输出）

所有模块共享：
    └── utils/          # 通用工具（日志、异常、工作空间）
```

## 5. 扩展性计划

### 5.1 代码生成策略优化

**当前设计优势**：
- **工程量最小**：优先使用 LLM 生成，无需为每个架构写模板
- **灵活性强**：LLM 能处理各种模型变体和特殊场景
- **稳定可靠**：通用模板作为 Fallback，不依赖外部 API

**后续优化方向**：
- 优化 LLM prompt，提升代码生成成功率
- 收集常见错误模式，改进通用模板
- 支持更多 HF 任务类型（NLP, CV, Audio, Speech）

### 5.2 功能扩展

- 自动化 PR 提交功能
- 批量处理多个模型
- 失败重试和错误恢复机制
- 样本质量评估和筛选
