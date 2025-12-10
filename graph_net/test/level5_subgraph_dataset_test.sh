#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DECOMPOSE_PATH=$GRAPH_NET_ROOT/decompose_level5_100

mkdir -p "$DECOMPOSE_PATH"

model_list="$GRAPH_NET_ROOT/graph_net/config/torch_samples_list.txt"
# model_list="$GRAPH_NET_ROOT/graph_net/test/dev_model_list/tmp_torch_samples_list.txt"

# Split models into sequences
python3 -m graph_net.torch.typical_sequence_split_points \
    --model-list "$model_list" \
    --device "cuda" \
    --window-size 10 \
    --fold-policy default \
    --fold-times 10 \
    --output-json "$DECOMPOSE_PATH/split_results.json"

# Decompose models
decompose_config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_decomposer.py",
    "handler_class_name": "RangeDecomposerExtractor",
    "handler_config": {
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$DECOMPOSE_PATH",
        "split_results_path": "$DECOMPOSE_PATH/split_results.json",
        "group_head_and_tail": true,
        "chain_style": false
    }
}
EOF
)
DECOMPOSE_CONFIG=$(echo $decompose_config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler \
    --model-path-list $model_list \
    --handler-config=$DECOMPOSE_CONFIG \
    --use-subprocess

# Rename graph variable
config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_variable_renamer.py",
    "handler_class_name": "GraphVariableRenamer",
    "handler_config": {
        "model_path_prefix": "",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "$DECOMPOSE_PATH/graph_variable_rename_workspace"
    }
}
EOF
)
CONFIG=$(echo $config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler \
    --model-path $DECOMPOSE_PATH \
    --handler-config=$CONFIG \
    --use-subprocess
