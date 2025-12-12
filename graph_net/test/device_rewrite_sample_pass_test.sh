#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
model_path_handler_config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_passes/device_rewrite_sample_pass.py",
    "handler_class_name": "DeviceRewriteSamplePass",
    "handler_config": {
        "device": "cuda",
        "resume": false,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "/tmp/device_rewrited"
    }
}
EOF
)

model_path_handler_model_path_list="$GRAPH_NET_ROOT/graph_net/test/dev_model_list/validation_error_model_list.txt"
MODEL_PATH_HANDLER_CONFIG=$(echo $model_path_handler_config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler \
    --model-path-list $model_path_handler_model_path_list \
    --handler-config $MODEL_PATH_HANDLER_CONFIG \

unset model_path_handler_model_path_list
unset MODEL_PATH_HANDLER_CONFIG

