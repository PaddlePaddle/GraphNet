#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
# model_runnable_predicator=ShapePropagatablePredicator
model_runnable_predicator=ModelRunnablePredicator
config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/torch/check_graph_module_parsable.py",
    "handler_class_name": "CheckGraphModuleParsable",
    "handler_config": {
        "model_path_prefix": "$GRAPH_NET_ROOT/../",
        "resume": true,
        "limits_handled_models": 999999,
        "output_dir": "/tmp/check_graph_module_parsable"
    }
}
EOF
)
CONFIG=$(echo $config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler --model-path-list $GRAPH_NET_ROOT/test/dev_model_list/graph_module_parse_error_torch_sample_list.txt --handler-config=$CONFIG --use-subprocess
