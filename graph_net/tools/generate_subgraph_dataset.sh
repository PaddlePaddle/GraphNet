#!/bin/bash
set -x

OP_NUM=${1:-64}
GPU_ID=${2:-4}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH=/work/GraphNet:/work/abstract_pass/Athena:$PYTHONPATH

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

DECOMPOSE_WORKSPACE=/work/graphnet_test_workspace/subgraph_dataset_20251221
LEVEL_DECOMPOSE_WORKSPACE=$DECOMPOSE_WORKSPACE/decomposed_${OP_NUM}ops
OP_NAMES_OUTPUT_DIR=${DECOMPOSE_WORKSPACE}/sample_op_names
RANGE_DECOMPOSE_OUTPUT_DIR="${LEVEL_DECOMPOSE_WORKSPACE}/range_decompose"
GRAPH_VAR_RENAME_OUTPUT_DIR=$LEVEL_DECOMPOSE_WORKSPACE/graph_var_renamed
DEDUPLICATED_OUTPUT_DIR=$LEVEL_DECOMPOSE_WORKSPACE/deduplicated
UNITTESTS_OUTPUT_DIR=$LEVEL_DECOMPOSE_WORKSPACE/unittests

mkdir -p "$LEVEL_DECOMPOSE_WORKSPACE"

model_list="$GRAPH_NET_ROOT/graph_net/config/torch_samples_list.txt"
range_decomposed_subgraph_list=${LEVEL_DECOMPOSE_WORKSPACE}/range_decomposed_subgraph_sample_list.txt
deduplicated_subgraph_list=${LEVEL_DECOMPOSE_WORKSPACE}/deduplicated_subgraph_sample_list.txt

function generate_subgraph_list() {
    local target_dir="$1"
    local sample_list="$2"
    echo ">>> Generate subgraph_sample_list for samples under ${target_dir}."
    echo ">>>"
    cat $model_list \
        | grep -v '# ' \
        | xargs -I {} find ${target_dir}/{} -name "model.py" \
        | xargs dirname \
        | xargs realpath --relative-to=$target_dir \
        | tee $sample_list
}

function generate_op_names() {
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

function generate_split_point() {
    # MIN_SEQ_OPS, MAX_SEQ_OPS
    # level 1:  2,  4
    # level 2:  4,  8
    # level 3:  8, 16
    # level 4: 16, 32
    # level 5: 32, 64
    MIN_SEQ_OPS=$((${OP_NUM} / 2))
    MAX_SEQ_OPS=${OP_NUM}
    echo ">>> [2] Generate split points for samples in ${model_list}."
    echo ">>>   OP_NUM: ${OP_NUM}, MIN_SEQ_OPS: ${MIN_SEQ_OPS}, MAX_SEQ_OPS: ${MAX_SEQ_OPS}"
    echo ">>>"
    python3 -m graph_net.torch.typical_sequence_split_points \
        --model-list "$model_list" \
        --op-names-path-prefix "${OP_NAMES_OUTPUT_DIR}" \
        --device "cuda" \
        --window-size ${OP_NUM} \
        --fold-policy default \
        --fold-times 16 \
        --min-seq-ops ${MIN_SEQ_OPS} \
        --max-seq-ops ${MAX_SEQ_OPS} \
        --subgraph-ranges-json "$LEVEL_DECOMPOSE_WORKSPACE/subgraph_ranges_${OP_NUM}.json" \
        --output-json "$LEVEL_DECOMPOSE_WORKSPACE/split_results_${OP_NUM}.json"
}

function range_decompose() {
    echo ">>> [3] Decompose according to split_results.json for samples in ${model_list}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list "$model_list" \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/graph_decomposer.py",
    "handler_class_name": "RangeDecomposerExtractor",
    "handler_config": {
        "resume": false,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "${RANGE_DECOMPOSE_OUTPUT_DIR}",
        "split_results_path": "$LEVEL_DECOMPOSE_WORKSPACE/split_results_${OP_NUM}.json",
        "subgraph_ranges_path": "$LEVEL_DECOMPOSE_WORKSPACE/subgraph_ranges_${OP_NUM}.json",
        "group_head_and_tail": true,
        "chain_style": false
    }
}
EOF
)
}

function rename_subgraph() {
    echo ">>> [4] Rename subgraph samples under ${RANGE_DECOMPOSE_OUTPUT_DIR}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list ${range_decomposed_subgraph_list} \
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
        "output_dir": "$GRAPH_VAR_RENAME_OUTPUT_DIR"
    }
}
EOF
)
}

function remove_duplicates() {
    echo ">>> [5] Remove duplicated subgraph samples under ${GRAPH_VAR_RENAME_OUTPUT_DIR}."
    echo ">>>"
    python3 -m graph_net.tools.deduplicated \
        --samples-dir ${GRAPH_VAR_RENAME_OUTPUT_DIR} \
        --target-dir ${DEDUPLICATED_OUTPUT_DIR}
}

function generate_unittests() {
    echo ">>> [6] Generate unittests for subgraph samples under ${DEDUPLICATED_OUTPUT_DIR}."
    echo ">>>"
    python3 -m graph_net.model_path_handler \
        --model-path-list ${deduplicated_subgraph_list} \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/agent_unittest_generator.py",
    "handler_class_name": "AgentUnittestGeneratorPass",
    "handler_config": {
        "framework": "torch",
        "model_path_prefix": "${DEDUPLICATED_OUTPUT_DIR}",
        "output_dir": "$UNITTESTS_OUTPUT_DIR",
        "device": "cuda",
        "generate_main": true,
        "try_run": true,
        "resume": true,
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",                                                                                     
        "data_input_predicator_class_name": "RenamedDataInputPredicator"
    }
}
EOF
)
}

main() {
    timestamp=`date +%Y%m%d_%H%M`
    suffix="${OP_NUM}ops_${timestamp}"
    #generate_op_names 2>&1 | tee ${LEVEL_DECOMPOSE_WORKSPACE}/log_op_names_${suffix}.txt
    generate_split_point 2>&1 | tee ${LEVEL_DECOMPOSE_WORKSPACE}/log_split_point_${suffix}.txt
    range_decompose 2>&1 | tee ${LEVEL_DECOMPOSE_WORKSPACE}/log_range_decompose_${suffix}.txt

    generate_subgraph_list ${RANGE_DECOMPOSE_OUTPUT_DIR} ${range_decomposed_subgraph_list}
    rename_subgraph 2>&1 | tee ${LEVEL_DECOMPOSE_WORKSPACE}/log_rename_subgraph_${suffix}.txt
    remove_duplicates 2>&1 | tee ${LEVEL_DECOMPOSE_WORKSPACE}/log_remove_duplicates_${suffix}.txt

    generate_subgraph_list ${DEDUPLICATED_OUTPUT_DIR} ${deduplicated_subgraph_list}
    generate_unittests 2>&1 | tee ${LEVEL_DECOMPOSE_WORKSPACE}/log_generate_unittests_${suffix}.txt
}

main