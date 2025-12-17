# GraphNet 自动样本抽取 Agent 设计文档

## 1. 任务背景
为了丰富 GraphNet 的样本库，我们需要从 Hugging Face (HF) 上自动下载模型，并将其转换为 GraphNet 可用的子图样本。目前这一过程需要人工编写 `run_model.py` 代码，效率较低。本 Agent 旨在自动化这一流程，实现从“HF 模型链接”到“GraphNet 样本提交”的端到端自动化。

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

#### 1. Model Fetcher (`agent.fetcher`)
- **功能**: 调用 `huggingface_hub` 下载模型快照。
- **输入**: `model_id` (e.g., `bert-base-uncased`)
- **输出**: 本地路径。

#### 2. Model Analyzer (`agent.analyzer`)
- **功能**: 解析模型目录下的 `config.json` 或 `README.md`。
- **目标**: 推断模型的 `input_shape` 和 `input_dtype`。例如 BERT 通常需要 `input_ids` [batch, seq_len] (int64)。

#### 3. Code Generator (`agent.coder`)
- **功能**: 生成 `run_model.py`。
- **策略**:
    - **Template Mode**: 针对常见架构（如 Bert, ResNet, GPT）使用预定义模板。
    - **LLM Mode (可选)**: 调用外部 LLM API 生成代码（预留接口）。

#### 4. Graph Extractor (`agent.extractor`)
- **功能**: 在子进程中运行生成的 `run_model.py`。
- **依赖**: 复用 `graph_net.torch.run_model` 或直接调用脚本。

#### 5. Sample Verifier (`agent.verifier`)
- **功能**: 检查生成的 `graph_net.json`, `model.py`, `input_meta.py` 是否存在且合法。

## 3. 接口设计

### `GraphNetAgent` 类
```python
class GraphNetAgent:
    def __init__(self, workspace: str, hf_token: str = None):
        self.workspace = workspace
        self.fetcher = HFFetcher(token=hf_token)
        self.analyzer = ConfigAnalyzer()
        self.coder = TemplateCoder()
        self.extractor = SubprocessExtractor()
        self.verifier = BasicVerifier()

    def run(self, model_id: str) -> bool:
        # 1. Download
        model_dir = self.fetcher.download(model_id)
        
        # 2. Analyze
        meta_info = self.analyzer.analyze(model_dir)
        
        # 3. Generate Code
        code_path = self.coder.generate(model_dir, meta_info)
        
        # 4. Extract
        output_dir = self.extractor.extract(code_path)
        
        # 5. Verify
        return self.verifier.verify(output_dir)
```

## 4. 目录结构
```text
graph_net/
  agent/
    __init__.py
    core.py          # Agent 主逻辑
    fetcher.py       # 下载模块
    analyzer.py      # 分析模块
    coder/
      base.py
      template.py    # 模板生成
      llm.py         # LLM 生成 (Interface)
    extractor.py     # 运行模块
    verifier.py      # 验证模块
```

## 5. 扩展性计划
- 支持更多的 HF 任务类型（NLP, CV, Audio）。
- 接入 DeepSeek/OpenAI API 提升代码生成成功率。
- 自动化 PR 提交功能。
