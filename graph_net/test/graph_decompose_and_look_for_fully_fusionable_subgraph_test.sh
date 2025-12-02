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
        "custom_extractor_path": "$GRAPH_NET_ROOT/torch/graph_decompose_and_look_for_fully_fusionable_subgraph.py",
        "custom_extractor_config": {
            "output_dir": "/tmp/naive_decompose_workspace",
            "split_positions": [8,16,17,18,19,20,32],
            "group_head_and_tail": true,
            "filter_path":"$GRAPH_NET_ROOT/torch/naive_subgraph_filter.py",
            "filter_config": {},
            "post_extract_process_path":"$GRAPH_NET_ROOT/torch/post_extract_process_count_kernels.py",
            "post_extract_process_class_name": "GraphFullyFusionable",
            "max_step": 4
        }
    }
}
EOF
)
DECORATOR_CONFIG=$(echo $decorator_config_json_str | base64 -w 0)

python3 -m graph_net.torch.run_model --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --decorator-config=$DECORATOR_CONFIG
