#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
MODEL_NAME=resnet18
MODEL_PATH_IN_SAMPLES=/timm/$MODEL_NAME
config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/constraint_util.py",
    "handler_class_name": "UpdateInputTensorConstraints",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$GRAPH_NET_ROOT/../",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ShapePropagatablePredicator",
        "dimension_generalizer_filepath": "$GRAPH_NET_ROOT/torch/static_to_dynamic.py",
        "dimension_generalizer_class_name": "StaticToDynamic",
        "last_model_log_file": "/tmp/a.py"
    }
}
EOF
)
CONFIG=$(echo $config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler --model-path-list $GRAPH_NET_ROOT/config/torch_samples_list.txt --handler-config=$CONFIG
