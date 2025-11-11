#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
MODEL_PATH_IN_SAMPLES=/timm/resnet18 
extractor_config_json_str=$(cat <<EOF
{
    "custom_extractor_path": "$GRAPH_NET_ROOT/torch/naive_graph_decomposer.py",
    "custom_extractor_config": {
        "output_dir": "/tmp/chain_naive_decompose_workspace",
        "split_positions": [8, 16, 32],
        "group_head_and_tail": true,
        "chain_style": true
    }
}
EOF
)
EXTRACTOR_CONFIG=$(echo $extractor_config_json_str | base64 -w 0)

mkdir -p /tmp/naive_decompose_workspace
python3 -m graph_net.torch.single_device_runner --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --enable-extract True --extract-name resnet18 --dump-graph-hash-key --extractor-config=$EXTRACTOR_CONFIG