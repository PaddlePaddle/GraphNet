#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DECOMPOSE_PATH=$GRAPH_NET_ROOT/decompose_workspace

mkdir -p "$DECOMPOSE_PATH"

temp_model_list=$(mktemp)
cat "$GRAPH_NET_ROOT/graph_net/config/torch_samples_list.txt" > "$temp_model_list"

python3 -m graph_net.torch.typical_sequence_split_points \
    --model-list "$temp_model_list" \
    --device "cuda" \
    --window-size 10 \
    --output-json "$DECOMPOSE_PATH/split_results.json"

while IFS= read -r MODEL_PATH_IN_SAMPLES; do
    if [[ -n "$MODEL_PATH_IN_SAMPLES" ]]; then
        MODEL_FULL_PATH="$GRAPH_NET_ROOT/$MODEL_PATH_IN_SAMPLES"
        MODEL_NAME=$(basename "$MODEL_PATH_IN_SAMPLES")

        echo "== Decomposing $MODEL_PATH_IN_SAMPLES. =="

        decomposer_config_json_str=$(cat <<EOF
{
    "split_results_path": "$DECOMPOSE_PATH/split_results.json",
    "workspace_path": "$DECOMPOSE_PATH",
    "chain_style": true,
    "target_model_name": "$MODEL_NAME"
}
EOF
        )
        DECOMPOSER_CONFIG=$(echo $decomposer_config_json_str | base64 -w 0)

        python3 -m graph_net.torch.test_compiler \
            --model-path "$MODEL_FULL_PATH" \
            --compiler range_decomposer \
            --device cuda \
            --config="$DECOMPOSER_CONFIG"

        cp -r "$MODEL_FULL_PATH" "$DECOMPOSE_PATH/"

        echo "== Validating $MODEL_PATH_IN_SAMPLES. =="

        python3 -m graph_net.torch.test_compiler \
            --model-path "$DECOMPOSE_PATH/$MODEL_NAME" \
            --compiler range_decomposer_validator \
            --device cuda > "$DECOMPOSE_PATH/${MODEL_NAME}_validation.log" 2>&1

        echo "== Finished processing $MODEL_PATH_IN_SAMPLES. =="
    fi
done < $temp_model_list

rm -f "$temp_model_list"

cat $DECOMPOSE_PATH/*_validation.log >> $DECOMPOSE_PATH/combined.log

python3 -m graph_net.plot_ESt \
    --benchmark-path "$DECOMPOSE_PATH/combined.log" \
    --output-dir "$DECOMPOSE_PATH"