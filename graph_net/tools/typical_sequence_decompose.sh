#!/bin/bash
set -x

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DECOMPOSE_WORKSPACE=/tmp/workspace_typical_sequence_decompose

mkdir -p "$DECOMPOSE_WORKSPACE"

model_list="$GRAPH_NET_ROOT/graph_net/test/dev_model_list/validation_error_model_list.txt"

python3 -m graph_net.apply_sample_pass \
    --model-path-list $model_list \
    --sample-pass-file-path $GRAPH_NET_ROOT/graph_net/torch/typical_sequence_split_points.py \
    --sample-pass-class-name OpNamesExtractor \
    --sample-pass-config=$(base64 -w 0 <<EOF
{
    "resume": true,
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "output_dir": "$DECOMPOSE_WORKSPACE"
}
EOF
)

TYPICAL_SEQUENCE_RANGES_WORKSPACE=$DECOMPOSE_WORKSPACE/workspace_typical_sequence_ranges

python3 -m graph_net.torch.typical_sequence_split_points \
    --model-list "$model_list" \
    --op-names-path-prefix "$DECOMPOSE_WORKSPACE" \
    --device "cuda" \
    --window-size 10 \
    --fold-policy default \
    --fold-times 10 \
    --min-seq-ops 4 \
    --max-seq-ops 16 \
    --output-dir "$TYPICAL_SEQUENCE_RANGES_WORKSPACE" \
    --subgraph-ranges-file-name "subgraph_ranges.json" \
    --subgraph-ranges-json "$DECOMPOSE_WORKSPACE/subgraph_ranges.json" \
    --output-json "$DECOMPOSE_WORKSPACE/split_results.json"

python3 -m graph_net.apply_sample_pass \
    --model-path-list $model_list \
    --sample-pass-file-path $GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py \
    --sample-pass-class-name SubgraphGenerator \
    --sample-pass-config=$(base64 -w 0 <<EOF
{
    "resume": false,
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "output_dir": "$DECOMPOSE_WORKSPACE",
    "subgraph_ranges_json_root": "$TYPICAL_SEQUENCE_RANGES_WORKSPACE",
    "group_head_and_tail": true,
    "chain_style": false
}
EOF
)
