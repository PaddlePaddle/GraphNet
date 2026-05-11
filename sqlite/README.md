# GraphNet SQLite 操作指南

## 目录结构

```
sqlite/
├── migrates/                           # SQL 迁移文件（按时间戳顺序执行）
├── orm_models.py                       # SQLAlchemy ORM 模型定义
├── init_db.py                          # 数据库初始化
├── build_db.py                         # 批量建库（推荐）
├── graphsample_insert.py               # 单条样本插入
├── graphsample_delete.py               # 单条样本删除（软删除）
├── merge_db.py                         # 数据库合并
├── graph_sample_bucket_generator.py    # 样本分桶元数据生成
├── graph_sample_groups_insert.py       # 采样分组生成
├── upload_dataset.py                    # HuggingFace 上传
├── download_dataset.py                  # HuggingFace 下载
```

## 数据表概览

| 表名 | 用途 |
|------|------|
| `repo` | 仓库源信息 |
| `graph_sample` | 计算图样本主表 |
| `subgraph_source` | 子图来源映射 |
| `dimension_generalization_source` | 维度泛化来源 |
| `datatype_generalization_source` | 数据类型泛化来源 |
| `backward_graph_source` | 反向图来源 |
| `sample_op_name` / `sample_op_name_list` | 算子名称序列 |
| `sample_input_tensor_meta` | 输入张量元信息 |
| `graph_net_sample_buckets` | 样本分桶元数据 |
| `graph_net_sample_groups` | 采样分组 |

所有删除操作均为软删除（`deleted` 字段标记），不物理删除数据。

## 数据库初始化

从 `migrates/` 目录按时间戳顺序执行 SQL 文件，创建所有表结构。**库文件已存在时会被删除重建。**

```bash
# 默认路径 GraphNet.db
python init_db.py 2>&1 | tee logs/init_db_$(date +"%Y%m%d_%H%M%S").log

# 自定义路径
python init_db.py --db_path xxx.db
```

## 批量建库（推荐）

一次性处理 `full_graph`、`typical_graph`、`fusible_graph`、`sole_op_graph` 四种样本类型，自动收集目录或读取 list 文件后逐条插入。库文件不存在时自动初始化。

```bash
python build_db.py \
    --db_path GraphNet.db \
    --dataset_root /path/to/dataset \
    --repo_uid "hf_torch_samples" \
    --op_names_path_prefix /path/to/sample_op_names
```

## 单条样本操作

```bash
# 插入单条
python graphsample_insert.py \
    --model_path_prefix /path/to/dataset/full_graph \
    --relative_model_path models/torch/resnet18 \
    --repo_uid "hf_torch_samples" \
    --sample_type "full_graph" \
    --order_value 0 \
    --db_path GraphNet.db

# 删除单条（软删除，设置 deleted=1）
python graphsample_delete.py \
    --db_path GraphNet.db \
    --repo_uid "hf_torch_samples" \
    --relative_model_path "models/torch/resnet18"
```

## Shell 批量脚本

```bash
# 批量插入（从 list 文件逐行读取）
bash graphsample_insert.sh [db_path]

# 批量删除（从 graph_net/config/delete_list.txt 读取）
bash graphsample_delete.sh [db_path]
```

## 数据库合并

将新库的所有记录合并到主库，自动跳过已存在的 repo 和 graph_sample。

```bash
python merge_db.py \
    --main_db_path GraphNet.db \
    --new_db_path new.db
```

## 样本分桶与分组

分两步：先生成样本的分桶元数据（op 序列、input shape、dtype 的哈希 ID），再基于分桶结果按策略生成采样分组。

```bash
# 生成分桶元数据 → 写入 graph_net_sample_buckets 表
python graph_sample_bucket_generator.py --db_path GraphNet.db

# 生成采样分组 → 写入 graph_net_sample_groups 表
# 策略: bucket_policy_v1 (stride-16 + cross-shape) + bucket_policy_v2 (dtype coverage + sparse)
python graph_sample_groups_insert.py --db_path GraphNet.db --num_dtypes 3
```

## HuggingFace 上传/下载

```bash
# 上传：打包 dataset 目录 + GraphNet.db 到 HF Hub
python upload_dataset.py \
    --hf_token <your_token> \
    --base_dir /path/to/dataset \
    --repo_id "PaddlePaddle/GraphNet" \
    --revision "20260203" \
    --split "GraphNet"

# 下载：从 HF Hub 拉取 dataset 和 GraphNet.db
python download_dataset.py \
    --repo_id "PaddlePaddle/GraphNet" \
    --revision "20260224" \
    --save_dir ./workspace \
    --split "GraphNet"
```

## 关联资源

- ORM 模型定义: [orm_models.py](orm_models.py)
- SQL 迁移文件: [migrates/](migrates/)