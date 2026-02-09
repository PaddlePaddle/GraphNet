# SQLite Operations Guide

**Working Directory:** `/GraphNet`

## Setup

```bash
# Create logs directory
mkdir -p sqlite/logs
```

## Initialize Database

```bash
# Default DB path (sqlite/GraphNet.db)
python ./sqlite/init_db.py 2>&1 | tee sqlite/logs/init_db_$(date +"%Y%m%d_%H%M%S").log

# Custom DB path
python ./sqlite/init_db.py --db_path sqlite/xxx.db 2>&1 | tee sqlite/logs/init_db_$(date +"%Y%m%d_%H%M%S").log
```

## Insert Graph Samples

```bash
# Insert to custom DB
bash ./sqlite/graphsample_insert.sh sqlite/xxx.db 2>&1 | tee sqlite/logs/insert_$(date +"%Y%m%d_%H%M%S").log

# Insert to default DB (sqlite/GraphNet.db)
bash ./sqlite/graphsample_insert.sh 2>&1 | tee sqlite/logs/insert_$(date +"%Y%m%d_%H%M%S").log
```

## Delete Graph Samples

```bash
# Delete from custom DB
bash ./sqlite/graphsample_delete.sh sqlite/xxx.db 2>&1 | tee sqlite/logs/delete_$(date +"%Y%m%d_%H%M%S").log

# Delete from default DB (sqlite/GraphNet.db)
bash ./sqlite/graphsample_delete.sh 2>&1 | tee sqlite/logs/delete_$(date +"%Y%m%d_%H%M%S").log
```

## Merge Databases and Upload to Hugging Face

```bash
# Usage: python ./sqlite/upload.py --main_db_path <path> --new_db_path <path>
python ./sqlite/upload.py --main_db_path <path> --new_db_path <path>
```
