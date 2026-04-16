#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GRAPH_NET_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_ROOT=$(dirname "$GRAPH_NET_DIR")

# 将项目根目录加入Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

LOG_FILE_PATH="log_file_for_test.txt"

python3 - <<END
from graph_net_bench import analysis_util

print_detail = False
for tolerance in range(-10, 3, 1):
    incorrect_models = analysis_util.get_incorrect_models(tolerance, '$LOG_FILE_PATH')
    if tolerance <= 1:
        incorrect_models_next = analysis_util.get_incorrect_models(tolerance + 1, '$LOG_FILE_PATH')
        incorrect_models = incorrect_models - incorrect_models_next
    print(f"- tolerance=[{tolerance}, {tolerance + 1}), number of incorrect_models={len(incorrect_models)}")
    if print_detail:
        for idx, sample in enumerate(incorrect_models):
            print(f"- {idx=}, {sample=}")
        print()

END