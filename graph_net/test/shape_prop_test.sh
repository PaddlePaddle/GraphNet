#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/torch/constraint_util.py",
    "handler_class_name": "ShapePropagatablePredicator"
}
EOF
)
CONFIG=$(echo $config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler --model-path $1 --handler-config=$CONFIG
