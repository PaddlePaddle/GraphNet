#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/work/GraphNet:/work/abstract_pass/Athena:$PYTHONPATH

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

OP_NUM=64
DECOMPOSE_WORKSPACE=/work/graphnet_test_workspace/subgraph_dataset_20251218
LEVEL_DECOMPOSE_WORKSPACE=$DECOMPOSE_WORKSPACE/decomposed_${OP_NUM}ops
OP_NAMES_OUTPUT_DIR=${LEVEL_DECOMPOSE_WORKSPACE}/sample_op_names
RANGE_DECOMPOSE_OUTPUT_DIR="${LEVEL_DECOMPOSE_WORKSPACE}/range_decompose"
GRAPH_VAR_RENAME_WORKSPACE=$LEVEL_DECOMPOSE_WORKSPACE/graph_var_renamed

mkdir -p "$LEVEL_DECOMPOSE_WORKSPACE"

model_list="$GRAPH_NET_ROOT/graph_net/config/torch_samples_list.txt"
subgraph_sample_list=$RANGE_DECOMPOSE_OUTPUT_DIR/subgraph_sample_list.txt

generate_op_names() {
    echo ">>> [1] Generate op_names.txt for samples in ${model_list}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list $model_list \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/typical_sequence_split_points.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "${OP_NAMES_OUTPUT_DIR}"
    }
}
EOF
)
}

generate_split_point() {
    echo ">>> [2] Generate split points for samples in ${model_list}."
    echo ">>>"
    python3 -m graph_net.torch.typical_sequence_split_points \
        --enable-resume \
        --model-list "$model_list" \
        --op-names-path-prefix "${OP_NAMES_OUTPUT_DIR}" \
        --device "cuda" \
        --window-size ${OP_NUM} \
        --fold-policy default \
        --fold-times 16 \
        --min-seq-ops $((${OP_NUM} / 2)) \
        --max-seq-ops $((${OP_NUM} * 2)) \
        --output-json "$DECOMPOSE_WORKSPACE/split_results_${OP_NUM}.json"
}

range_decompose() {
    echo ">>> [3] Decompose according to split_results.json for samples in ${model_list}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list "$model_list" \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_decomposer.py",
    "handler_class_name": "RangeDecomposerExtractor",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "${RANGE_DECOMPOSE_OUTPUT_DIR}",
        "split_results_path": "$DECOMPOSE_WORKSPACE/split_results_${OP_NUM}.json",
        "group_head_and_tail": true,
        "chain_style": false
    }
}
EOF
)
}

generate_subgraph_list() {
    echo ">>> [4] Generate subgraph_sample_list for samples under ${RANGE_DECOMPOSE_OUTPUT_DIR}."
    echo ">>>"
    cat $model_list \
        | grep -v '# ' \
        | xargs -I {} find ${RANGE_DECOMPOSE_OUTPUT_DIR}/{} -name "model.py" \
        | xargs dirname \
        | xargs realpath --relative-to=$RANGE_DECOMPOSE_OUTPUT_DIR \
        | tee $subgraph_sample_list
}

rename_subgraph() {
    echo ">>> [5] Rename subgraph samples under ${RANGE_DECOMPOSE_OUTPUT_DIR}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list $subgraph_sample_list \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_variable_renamer.py",
    "handler_class_name": "GraphVariableRenamer",
    "handler_config": {
        "device": "cuda",
        "resume": true,
        "model_path_prefix": "$RANGE_DECOMPOSE_OUTPUT_DIR",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "$GRAPH_VAR_RENAME_WORKSPACE"
    }
}
EOF
)
}

main() {
    generate_op_names
    generate_split_point
    range_decompose
    generate_subgraph_list
    rename_subgraph
}

main