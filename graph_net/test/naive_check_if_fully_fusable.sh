#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
MODEL_NAME=resnet18d.ra2_in1k
MODEL_PATH_IN_SAMPLES=/timm/$MODEL_NAME
checker_config_json_str=$(cat <<EOF
{
    "post_extract_process_config": {
        "post_extract_process_path":"$GRAPH_NET_ROOT/torch/post_extract_process_count_kernels.py",
        "post_extract_process_class_name": "GraphFullyFusionable"
    }
}
EOF
)
CHECKER_CONFIG=$(echo $checker_config_json_str | base64 -w 0)

python3 -m graph_net.torch.check_model_fusability --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --checker-config=$CHECKER_CONFIG
