#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

MODEL1="$GRAPH_NET_ROOT/samples/torchvision/resnet18"
MODEL2="$GRAPH_NET_ROOT/samples/torchvision/resnet34"
MODEL_LIST_FILE=$(mktemp)
echo "$MODEL1" > "$MODEL_LIST_FILE"
echo "$MODEL2" >> "$MODEL_LIST_FILE"

python3 -m graph_net.torch.typical_sequence_split_points \
    --model-list "$MODEL_LIST_FILE" \
    --device "cuda" \
    --window-size 10 \
    --output-json "$GRAPH_NET_ROOT/split_results.json"

rm -f "$MODEL_LIST_FILE"


MODEL_PATH_IN_SAMPLES=/torchvision/resnet18
MODEL_NAME=$(basename "$MODEL_PATH_IN_SAMPLES")

decomposer_config_json_str=$(cat <<EOF
{
    "split_results_path": "$GRAPH_NET_ROOT/split_results.json",
    "workspace_path": "$GRAPH_NET_ROOT/decompose_workspace",
    "chain_style": "True"
}
EOF
)
DECOMPOSER_CONFIG=$(echo $decomposer_config_json_str | base64 -w 0)

python3 -m graph_net.torch.test_compiler --model-path $GRAPH_NET_ROOT/samples/$MODEL_PATH_IN_SAMPLES --compiler range_decomposer --device cuda --config=$DECOMPOSER_CONFIG


DECOMPOSE_PATH=$GRAPH_NET_ROOT/decompose_workspace
cp -r "$GRAPH_NET_ROOT/samples/$MODEL_PATH_IN_SAMPLES" "$DECOMPOSE_PATH/"

python3 -m graph_net.torch.test_compiler \
    --model-path $DECOMPOSE_PATH/$MODEL_NAME \
    --compiler range_decomposer_validator \
    --device cuda > "$DECOMPOSE_PATH/log.log" 2>&1

python3 -m graph_net.plot_ESt \
  --benchmark-path $DECOMPOSE_PATH/log.log \
  --output-dir $DECOMPOSE_PATH \