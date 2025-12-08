#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
# model_runnable_predicator=ShapePropagatablePredicator
model_runnable_predicator=ModelRunnablePredicator
config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/dimension_generalizer.py",
    "handler_class_name": "ApplyDimGenPasses",
    "handler_config": {
        "resume": false,
        "output_dir": "/tmp/dimension_generalized_samples",
        "model_path_prefix": "$GRAPH_NET_ROOT/../",
        "dimension_generalizer_filepath": "$GRAPH_NET_ROOT/torch/static_to_dynamic.py",
        "dimension_generalizer_class_name": "StaticToDynamic",
        "limits_handled_models": 10,
        "last_model_log_file": "/tmp/a.py"
    }
}
EOF
)
CONFIG=$(echo $config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler --model-path-list $GRAPH_NET_ROOT/config/torch_samples_list.txt --handler-config=$CONFIG
