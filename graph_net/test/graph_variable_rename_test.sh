#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
RENAMED_PATH=/tmp/graph_variable_rename_workspace

mkdir -p "$RENAMED_PATH"
model_list="$GRAPH_NET_ROOT/graph_net/config/small100_torch_samples_list.txt"

python3 -m graph_net.model_path_handler \
    --model-path-list $model_list \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_variable_renamer.py",
    "handler_class_name": "GraphVariableRenamer",
    "handler_config": {
        "device": "cuda",
        "resume": true,
        "try_run": true,
        "model_path_prefix": "$GRAPH_NET_ROOT/",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "$RENAMED_PATH"
    }
}
EOF
) \
    2>&1 | tee "$RENAMED_PATH/graph_rename.log"

python3 -m graph_net.torch.test_compiler \
    --model-path-prefix $GRAPH_NET_ROOT \
    --allow-list $model_list \
    --compiler graph_variable_renamer_validator \
    --device cuda \
    --config $(base64 -w 0 <<EOF
{
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "renamed_root": "$RENAMED_PATH"
}
EOF
) \
    2>&1 | tee "$RENAMED_PATH/validation.log"

python3 -m graph_net.plot_ESt \
    --benchmark-path "$RENAMED_PATH/validation.log" \
    --output-dir "$RENAMED_PATH"
