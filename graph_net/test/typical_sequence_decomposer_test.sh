#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DECOMPOSE_PATH=/tmp/decompose_workspace
# DECOMPOSE_PATH=$GRAPH_NET_ROOT/decompose_test_level5_100

mkdir -p "$DECOMPOSE_PATH"

# model_list="$GRAPH_NET_ROOT/graph_net/config/small100_torch_samples_list.txt"
model_list="$GRAPH_NET_ROOT/graph_net/test/dev_model_list/validation_error_model_list.txt"

op_names_extractor_config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/typical_sequence_split_points.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$DECOMPOSE_PATH"
    }
}
EOF
)
OP_NAMES_EXTRACTOR_CONFIG=$(echo $op_names_extractor_config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler \
    --model-path-list $model_list \
    --handler-config=$OP_NAMES_EXTRACTOR_CONFIG \

python3 -m graph_net.torch.typical_sequence_split_points \
    --enable-resume \
    --model-list "$model_list" \
    --op-names-path-prefix "$DECOMPOSE_PATH" \
    --device "cuda" \
    --window-size 10 \
    --fold-policy default \
    --fold-times 10 \
    --output-json "$DECOMPOSE_PATH/split_results.json"

decompose_config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_decomposer.py",
    "handler_class_name": "RangeDecomposerExtractor",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$DECOMPOSE_PATH",
        "split_results_path": "$DECOMPOSE_PATH/split_results.json",
        "group_head_and_tail": true,
        "chain_style": true
    }
}
EOF
)
DECOMPOSE_CONFIG=$(echo $decompose_config_json_str | base64 -w 0)

python3 -m graph_net.model_path_handler \
    --model-path-list $model_list \
    --handler-config=$DECOMPOSE_CONFIG \

test_compiler_config_json_str=$(cat <<EOF
{
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "decomposed_root": "$DECOMPOSE_PATH"
}
EOF
)
TEST_COMPILER_CONFIG=$(echo $test_compiler_config_json_str | base64 -w 0)

python3 -m graph_net.torch.test_compiler \
    --allow-list $model_list \
    --compiler range_decomposer_validator \
    --device cuda \
    --config $TEST_COMPILER_CONFIG \
    --model-path-prefix $GRAPH_NET_ROOT \
    2>&1 | tee "$DECOMPOSE_PATH/validation.log"

python3 -m graph_net.plot_ESt \
    --benchmark-path "$DECOMPOSE_PATH/validation.log" \
    --output-dir "$DECOMPOSE_PATH"
