#!/bin/bash
set -euo pipefail

MODEL_PATH_PREFIX="${1:?Usage: $0 <model_path_prefix>}"
MODEL_PATH_LIST="${2:-$MODEL_PATH_PREFIX/device_rewrited_sample_list_successed.txt}"

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/tools/_update_sym_dim_reifier.py",
    "handler_class_name": "UpdateSymDimReifier",
    "handler_config": {
        "reifier_factory_path": "$GRAPH_NET_ROOT/torch/reifier_factory.py",
        "reifier_factory_class_name": "ReifierFactory",
        "model_path_prefix": "$MODEL_PATH_PREFIX/"
    }
}
EOF
)
CONFIG=$(echo $config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler \
    --model-path-list "$MODEL_PATH_LIST" \
    --handler-config=$CONFIG
