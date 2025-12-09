#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")

if [ -z "$GRAPH_NET_DECOMPOSE_PATH" ]; then
    GRAPH_NET_DECOMPOSE_PATH="$(pwd)/graphnet_decompose"
fi

MODEL_PATH_IN_SAMPLES=/timm/resnet18
MODEL_NAME=$(basename "$MODEL_PATH_IN_SAMPLES")
OUTPUT_DIR="${GRAPH_NET_DECOMPOSE_PATH:-$(pwd)}"
cp -r "$GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES" "$OUTPUT_DIR/"

extractor_config_json_str=$(cat <<EOF
{
    "decorator_path": "$GRAPH_NET_ROOT/torch/extractor.py",
    "decorator_config": {
        "name": "$MODEL_NAME",
        "custom_extractor_path": "$GRAPH_NET_ROOT/torch/graph_decomposer.py",
        "custom_extractor_config": {
            "output_dir": "$OUTPUT_DIR/${MODEL_NAME}_decomposed",
            "split_positions": [8, 16, 32],
            "group_head_and_tail": true,
            "chain_style": true
        }
    }
}
EOF
)
EXTRACTOR_CONFIG=$(echo $extractor_config_json_str | base64 -w 0)
python3 -m graph_net.torch.run_model --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --decorator-config=$EXTRACTOR_CONFIG

FILE_PATH=$GRAPH_NET_DECOMPOSE_PATH/decomposer
mkdir -p "$(dirname "$FILE_PATH/log.log")"
MODEL_PATH="$GRAPH_NET_DECOMPOSE_PATH/$MODEL_NAME"

python -m graph_net.torch.test_compiler \
    --model-path $MODEL_PATH \
    --compiler range_decomposer_validator \
    --device cuda > "$FILE_PATH/log.log" 2>&1

python3 -m graph_net.plot_ESt \
  --benchmark-path $FILE_PATH/log.log \
  --output-dir $FILE_PATH \

echo "=================================================="
echo "Results saved in: $FILE_PATH/ES_result.png"
echo ""
echo "IMPORTANT: Please verify if the curve in ES_result.png is a straight line"
echo "If the curve is NOT a straight line, please check the log file: $FILE_PATH/log.log"
echo "=================================================="
