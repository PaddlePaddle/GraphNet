#!/bin/bash
set -x

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DECOMPOSE_PATH=/tmp/decompose_workspace
# DECOMPOSE_PATH=$GRAPH_NET_ROOT/decompose_test_level5_100

mkdir -p "$DECOMPOSE_PATH"

model_list="$GRAPH_NET_ROOT/graph_net/test/dev_model_list/cumsum_num_kernels_sample_list.txt"

python3 -m graph_net.model_path_handler \
    --model-path-list $model_list \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/typical_sequence_split_points.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$DECOMPOSE_PATH"
    }
}
EOF
)

