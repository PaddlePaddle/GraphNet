#!/bin/bash
# input model path
MODEL_PATH_IN_SAMPLES=/timm/resnet18 
# output model path
OUTPUT_DIR=/tmp/naive_decompose_workspace

mkdir -p $OUTPUT_DIR
# extract subgraph 0-8, 8-16
export GRAPH_NET_NAIVE_DECOMPOSER_SPLIT_POS=0,8,16
export GRAPH_NET_EXTRACT_WORKSPACE=$OUTPUT_DIR 
GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")
python3 -m graph_net.torch.single_device_runner --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --enable-extract True --extract-name resnet18 --dump-graph-hash-key --custom-extractor-path=$GRAPH_NET_ROOT/torch/naive_graph_decomposer.py
