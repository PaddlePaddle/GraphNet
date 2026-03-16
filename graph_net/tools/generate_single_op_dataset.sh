#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="6"

# ==============================================================================
# Configuration Area
# ==============================================================================

# Dynamic Path Retrieval
PYTHON_EXEC=$(which python3)
if [ -z "$PYTHON_EXEC" ]; then
    echo "Error: 'python3' not found in PATH. Please activate your virtualenv."
    exit 1
fi

GRAPH_NET_ROOT=$($PYTHON_EXEC -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
if [ -z "$GRAPH_NET_ROOT" ]; then
    echo "Error: Could not determine GRAPH_NET_ROOT. Ensure 'graph_net' is installed or in PYTHONPATH."
    exit 1
fi

RESUME="true"

# Workspace Setup
WORKSPACE="/work/graphnet_test_workspace/single_op_dataset_20260228"
MODEL_LIST="${GRAPH_NET_ROOT}/graph_net/config/torch_samples_list.txt"

# Output Directories
OP_NAMES_DIR="${WORKSPACE}/01_op_names"
RANGES_DIR="${WORKSPACE}/02_ranges"
RAW_SUBGRAPH_DIR="${WORKSPACE}/03_raw_subgraphs"
RENAMED_DIR="${WORKSPACE}/04_renamed"
DEDUPLICATED_DIR="${WORKSPACE}/05_deduplicated"

if [[ "$MODEL_LIST" == *"/torch_samples_list.txt" ]]; then
    USE_SUBPROCESS_ARGS="--use-subprocess"
else
    USE_SUBPROCESS_ARGS=""
fi

mkdir -p "$WORKSPACE"

# ==============================================================================
# Main Pipeline
# ==============================================================================

echo ">>> Starting Pipeline..."
echo "    Python: $PYTHON_EXEC"
echo "    Root:   $GRAPH_NET_ROOT"

if [ ! -f "$MODEL_LIST" ]; then
    echo "Error: Model list not found at $MODEL_LIST"
    exit 1
fi

function generate_generalized_subgraph_list() {
    local target_dir="$1"
    local sample_list="$2"
    echo ">>> Generate subgraph_sample_list for samples under ${target_dir}."
    echo ">>>"
    find ${target_dir} -name "model.py" \
        | xargs dirname \
        | xargs realpath --relative-to=${target_dir} \
        | tee $sample_list
}

function generate_op_names() {
    # Stage 1: Op Names
    echo ">>> Running Stage 1: Op Names..."
    python3 -m graph_net.model_path_handler \
        --model-path-list "${MODEL_LIST}" \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/op_names_extractor.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": {
        "resume": $RESUME,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$OP_NAMES_DIR"
    }
}
EOF
)
}

function extract_op_points() {
    # Stage 2: Ranges
    echo ">>> Running Stage 2: Ranges..."
    python3 -m graph_net.apply_sample_pass \
        --model-path-list "${MODEL_LIST}" \
        --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/sample_pass/op_extract_points_generator.py" \
        --sample-pass-class-name "OpExtractPointsGenerator" \
        --sample-pass-config=$(base64 -w 0 <<EOF
{
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "op_names_path_prefix": "$OP_NAMES_DIR",
    "output_dir": "$RANGES_DIR",
    "subgraph_ranges_file_name": "subgraph_ranges.json"
}
EOF
)
}

function generate_subgraphs() {
    # Stage 3: Decompose
    echo ">>> Running Stage 3: Decompose..."
    python3 -m graph_net.model_path_handler $USE_SUBPROCESS_ARGS \
        --model-path-list "${MODEL_LIST}" \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "resume": $RESUME,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$RAW_SUBGRAPH_DIR",
        "subgraph_ranges_json_root": "$RANGES_DIR",
        "subgraph_ranges_json_file_name": "subgraph_ranges.json",
        "group_head_and_tail": false,
        "chain_style": false,
        "device": null
    }
}
EOF
)
}

function rename_subgraphs() {
    # Stage 4: Rename
    echo ">>> Running Post-processing: Rename..."
    python3 -m graph_net.model_path_handler \
        --model-path-list "${WORKSPACE}/generated_subgraphs_list.txt" \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/ast_graph_variable_renamer.py",
    "handler_class_name": "AstGraphVariableRenamer",
    "handler_config": {
        "device": "cuda",
        "try_run": false,
        "resume": $RESUME,
        "model_path_prefix": "${RAW_SUBGRAPH_DIR}",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "${RENAMED_DIR}"
    }
}
EOF
)
}

function deduplicate_subgraphs() {
    # Stage 5: Deduplicate
    echo ">>> Running Post-processing: Deduplicate..."
    if [ -d "${DEDUPLICATED_DIR}" ]; then rm -rf "${DEDUPLICATED_DIR}"; fi

    python3 -m graph_net.tools.deduplicated \
        --samples-dir ${RENAMED_DIR} \
        --target-dir ${DEDUPLICATED_DIR}
}

function main() {
    TIMESTAMP=$(date +%Y%m%d_%H%M)

    generate_op_names 2>&1 | tee ${WORKSPACE}/log_op_names_${TIMESTAMP}.txt
    extract_op_points 2>&1 | tee ${WORKSPACE}/log_extract_op_points_${TIMESTAMP}.txt
    generate_subgraphs 2>&1 | tee ${WORKSPACE}/log_generate_subgraphs_${TIMESTAMP}.txt
    generate_generalized_subgraph_list ${RAW_SUBGRAPH_DIR} ${WORKSPACE}/generated_subgraphs_list.txt
    
    rename_subgraphs 2>&1 | tee ${WORKSPACE}/log_rename_subgraphs_${TIMESTAMP}.txt
    deduplicate_subgraphs 2>&1 | tee ${WORKSPACE}/log_deduplicated_subgraphs_${TIMESTAMP}.txt
    generate_generalized_subgraph_list ${DEDUPLICATED_DIR} ${WORKSPACE}/deduplicated_subgraphs_list.txt

    echo ">>> ALL DONE. Final dataset located at: ${DEDUPLICATED_DIR}"
}