#!/bin/bash


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GRAPH_NET_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_ROOT=$(dirname "$GRAPH_NET_DIR")

# 将项目根目录加入Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

TOLERANCE_LIST=(-2 -1 0 1 2)
LOG_FILE_PATH="log_file_for_test.txt"

python3 - <<END
from graph_net_bench import analysis_util

result = analysis_util.get_incorrect_models($TOLERANCE_LIST, '$LOG_FILE_PATH')

for item in result:
    print(item)
END