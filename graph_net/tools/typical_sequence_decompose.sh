#!/bin/bash
set -x

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DECOMPOSE_WORKSPACE=/tmp/typical_sequence_decompose_workspace

mkdir -p "$DECOMPOSE_WORKSPACE"

model_list="$GRAPH_NET_ROOT/graph_net/test/dev_model_list/validation_error_model_list.txt"

python3 -m graph_net.model_path_handler \
    --model-path-list $model_list \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/typical_sequence_split_points.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$DECOMPOSE_WORKSPACE"
    }
}
EOF
)

python3 -m graph_net.torch.typical_sequence_split_points \
    --enable-resume \
    --model-list "$model_list" \
    --op-names-path-prefix "$DECOMPOSE_WORKSPACE" \
    --device "cuda" \
    --window-size 10 \
    --fold-policy default \
    --fold-times 10 \
    --output-json "$DECOMPOSE_WORKSPACE/split_results.json"

python3 -m graph_net.model_path_handler \
    --model-path-list $model_list \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_decomposer.py",
    "handler_class_name": "RangeDecomposerExtractor",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$DECOMPOSE_WORKSPACE",
        "split_results_path": "$DECOMPOSE_WORKSPACE/split_results.json",
        "group_head_and_tail": true,
        "chain_style": false
    }
}
EOF
)

subgraph_sample_list=$DECOMPOSE_WORKSPACE/subgraph_sample_list.txt
cat $model_list \
    | grep -v '# ' \
    | xargs -I {} find $DECOMPOSE_WORKSPACE/{} -name "model.py" \
    | xargs dirname \
    | xargs realpath --relative-to=$DECOMPOSE_WORKSPACE \
    | tee $subgraph_sample_list

GRAPH_VAR_RENAME_WORKSPACE=$DECOMPOSE_WORKSPACE/graph_var_renamed

python3 -m graph_net.model_path_handler \
    --model-path-list $subgraph_sample_list \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_variable_renamer.py",
    "handler_class_name": "GraphVariableRenamer",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$DECOMPOSE_WORKSPACE",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "$GRAPH_VAR_RENAME_WORKSPACE"
    }
}
EOF
)

