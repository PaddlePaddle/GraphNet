#!/bin/bash
set -x

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DECOMPOSE_PATH=/tmp/decompose_workspace
# DECOMPOSE_PATH=$GRAPH_NET_ROOT/decompose_test_level5_100

mkdir -p "$DECOMPOSE_PATH"

# model_list="$GRAPH_NET_ROOT/graph_net/config/small100_torch_samples_list.txt"
model_list="$GRAPH_NET_ROOT/graph_net/test/dev_model_list/validation_error_model_list.txt"

python3 -m graph_net.model_path_handler \
    --model-path-list $model_list \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/op_names_extractor.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$DECOMPOSE_PATH"
    }
}
EOF
)

python3 -m graph_net.apply_sample_pass \
    --model-path-list $model_list \
    --sample-pass-file-path $GRAPH_NET_ROOT/graph_net/torch/sample_pass/typical_sequence_split_points.py \
    --sample-pass-class-name TypicalSequenceSplitPointsGenerator \
    --sample-pass-config=$(base64 -w 0 <<EOF
{
    "resume": false,
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "output_dir": "$DECOMPOSE_PATH/workspace_typical_sequence_ranges",
    "op_names_path_prefix": "$DECOMPOSE_PATH",
    "device": "cuda",
    "window_size": 10,
    "fold_policy": "default",
    "fold_times": 10,
    "min_seq_ops": 4,
    "max_seq_ops": 16,
    "output_json": "$DECOMPOSE_PATH/split_results.json",
    "subgraph_ranges_json": "$DECOMPOSE_PATH/subgraph_ranges.json"
}
EOF
)

python3 -m graph_net.model_path_handler \
    --model-path-list $model_list \
    --handler-config=$(base64 -w 0 <<EOF
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

device_rewrite_sample_list=$DECOMPOSE_PATH/device_rewrite_sample_list.txt
cat $model_list \
    | grep -v '# ' \
    | xargs -I {} find $DECOMPOSE_PATH/{} -name "model.py" \
    | xargs dirname \
    | xargs realpath --relative-to=$DECOMPOSE_PATH \
    | tee $device_rewrite_sample_list

DEVICE_REWRITE_WORKSPACE=$DECOMPOSE_PATH/device_rewrite

python3 -m graph_net.model_path_handler \
    --model-path-list $device_rewrite_sample_list \
    --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/device_rewrite_sample_pass.py",
    "handler_class_name": "DeviceRewriteSamplePass",
    "handler_config": {
        "device": "cuda",
        "resume": false,
        "model_path_prefix": "$DECOMPOSE_PATH",
        "output_dir": "$DEVICE_REWRITE_WORKSPACE"
    }
}
EOF
)


python3 -m graph_net_bench.torch.test_compiler \
    --model-path-prefix $GRAPH_NET_ROOT \
    --allow-list $model_list \
    --compiler range_decomposer_validator \
    --device cuda \
    --config $(base64 -w 0 <<EOF
{
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "decomposed_root": "$DEVICE_REWRITE_WORKSPACE"
}
EOF
) \
    2>&1 | tee "$DECOMPOSE_PATH/validation.log"

python3 -m graph_net_visual.plot_ESt \
    --benchmark-path "$DECOMPOSE_PATH/validation.log" \
    --output-dir "$DECOMPOSE_PATH"
