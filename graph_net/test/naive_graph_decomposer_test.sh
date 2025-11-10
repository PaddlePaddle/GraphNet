#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
MODEL_PATH_IN_SAMPLES=/timm/resnet18
MODEL_NAME=$(basename "$MODEL_PATH_IN_SAMPLES")
OUTPUT_DIR="${NAIVE_DECOMPOSE_WORKSPACE:-$(pwd)/naive_decompose_workspace}"

extractor_config_json_str=$(cat <<EOF
{
    "custom_extractor_path": "$GRAPH_NET_ROOT/torch/naive_graph_decomposer.py",
    "custom_extractor_config": {
        "output_dir": "$OUTPUT_DIR/${MODEL_NAME}_decomposed",
        "split_positions": [8, 16, 32],
        "group_head_and_tail": true,
        "filter_path":"$GRAPH_NET_ROOT/torch/naive_subgraph_filter.py",
        "filter_config": {}
    }
}
EOF
)
EXTRACTOR_CONFIG=$(echo $extractor_config_json_str | base64 -w 0)

python3 -m graph_net.torch.single_device_runner --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --enable-extract True --extract-name $MODEL_NAME --dump-graph-hash-key --extractor-config=$EXTRACTOR_CONFIG
