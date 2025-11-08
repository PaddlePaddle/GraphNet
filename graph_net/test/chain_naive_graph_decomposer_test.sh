#!/bin/bash
set -x

# input model path
MODEL_PATH_IN_SAMPLES=/timm/resnet18 
# extract subgraph 0-8, 8-16
read -r -d '' json_str <<'EOF'
{
    "output_dir": "/tmp/naive_decompose_workspace",
    "split_positions": [2, 4],
    "group_head_and_tail": false,
    "chain_style": true
}
EOF
CONFIG=$(echo $json_str | base64 -w 0) 

mkdir -p /tmp/naive_decompose_workspace
GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")
python3 -m graph_net.torch.single_device_runner --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --enable-extract True --extract-name resnet18 --dump-graph-hash-key --custom-extractor-path=$GRAPH_NET_ROOT/torch/naive_graph_decomposer.py --custom-extractor-config=$CONFIG
