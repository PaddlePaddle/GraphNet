#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
MODEL_NAME=resnet18
MODEL_PATH_IN_SAMPLES=/timm/$MODEL_NAME
decorator_config_json_str=$(cat <<EOF
{
    "decorator_path": "$GRAPH_NET_ROOT/torch/extractor.py",
    "decorator_config": {
        "name": "$MODEL_NAME",
        "custom_extractor_path": "$GRAPH_NET_ROOT/torch/fully_fusible_subgraph_extractor.py",
        "custom_extractor_config": {
            "output_dir": "/tmp/find_fully_fusible_output",
            "split_positions": [],
            "group_head_and_tail": true,
            "max_step": 3,
            "min_step": 2,
            "max_nodes": 5
        }
    }
}
EOF
)
DECORATOR_CONFIG=$(echo $decorator_config_json_str | base64 -w 0)

python3 -m graph_net.torch.run_model --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --decorator-config=$DECORATOR_CONFIG
