# GraphNet Agent

自动样本抽取 Agent，实现从 HuggingFace ModelID 到 GraphNet Sample 的自动化转换。

## 安装

### 基础依赖
```bash
# 已包含在 GraphNet 主依赖中
pip install torch torchvision
```

### Agent 可选依赖
```bash
# 安装 Agent 相关依赖（包括 huggingface_hub）
pip install -e ".[agent]"

# 或单独安装
pip install huggingface_hub>=0.20.0
```

## 环境配置

### 设置工作空间
```bash
export GRAPH_NET_EXTRACT_WORKSPACE=/path/to/your/workspace
```

或在代码中指定：
```python
from graph_net.agent import GraphNetAgent
agent = GraphNetAgent(workspace="/path/to/workspace")
```

## 使用示例

```python
from graph_net.agent import GraphNetAgent

# 初始化 Agent
agent = GraphNetAgent(
    workspace="./agent_workspace",
    hf_token=None  # 可选，用于访问私有模型
)

# 运行提取
success = agent.extract_sample("bert-base-uncased")

if success:
    print("✅ Sample extracted successfully")
else:
    print("❌ Extraction failed")
```

## 工作流程

1. **Fetch**: 从 HuggingFace 下载模型
2. **Analyze**: 解析 config.json 提取元数据
3. **CodeGen**: 生成 run_model.py 脚本
4. **Extract**: 执行脚本提取计算图
5. **Deduplicate**: 检查是否与已有样本重复
6. **Verify**: 验证样本完整性
7. **Archive**: 保存 run_model.py 到样本目录

## 测试

```bash
# 运行所有测试
pytest graph_net/agent/tests/ -v

# 运行实际模型测试（需要设置环境变量）
TEST_REAL_RUN=1 pytest graph_net/agent/tests/test_real_run.py -v
```
