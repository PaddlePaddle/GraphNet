#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")
WORKSPACE=/tmp/graph_variable_rename_workspace

# input model path
MODEL_NAME=resnet18
MODEL_PATH_IN_SAMPLES=/timm/$MODEL_NAME
config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/torch/graph_variable_renamer.py",
    "handler_class_name": "GraphVariableRenamer",
    "handler_config": {
        "model_path_prefix": "$GRAPH_NET_ROOT/../",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "$WORKSPACE"
    }
}
EOF
)
CONFIG=$(echo $config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler --model-path samples/$MODEL_PATH_IN_SAMPLES --handler-config=$CONFIG
# python3 -m graph_net.model_path_handler --model-path-list $GRAPH_NET_ROOT/config/decomposition_error_tmp_torch_samples_list.txt --handler-config=$CONFIG

test_compiler_config_json_str=$(cat <<EOF
{
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "renamed_root": "$WORKSPACE"
}
EOF
)
TEST_COMPILER_CONFIG=$(echo $test_compiler_config_json_str | base64 -w 0)

python3 -m graph_net.torch.test_compiler \
    --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES \
    --compiler graph_variable_renamer_validator \
    --device cuda \
    --config $TEST_COMPILER_CONFIG \
    > "$WORKSPACE/validation.log" 2>&1

python3 -m graph_net.plot_ESt \
    --benchmark-path "$WORKSPACE/validation.log" \
    --output-dir "$WORKSPACE"
