#!/bin/bash
set -e

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

RESUME="false"

# Workspace Setup
TIMESTAMP=$(date +%Y%m%d_%H%M)
WORKSPACE="/tmp/single_op_workspace_${TIMESTAMP}"
MODEL_LIST="${MODEL_LIST:-${GRAPH_NET_ROOT}/graph_net/config/small100_torch_samples_list.txt}"

# Output Directories
OP_NAMES_DIR="${WORKSPACE}/01_op_names"
RANGES_DIR="${WORKSPACE}/02_ranges"
RAW_SUBGRAPH_DIR="${WORKSPACE}/03_raw_subgraphs"
RENAMED_DIR="${WORKSPACE}/04_renamed"
DEDUPLICATED_DIR="${WORKSPACE}/05_deduplicated"

mkdir -p "$WORKSPACE"

# ==============================================================================
# Main Pipeline
# ==============================================================================

echo ">>> Starting Pipeline..."
echo "    Python: $PYTHON_EXEC"
echo "    Root:   $GRAPH_NET_ROOT"

# 1. Prepare Data
if [ ! -f "$MODEL_LIST" ]; then
    echo "Error: Model list not found at $MODEL_LIST"
    exit 1
fi

# 2. Stage 1: Op Names
echo ">>> Running Stage 1: Op Names..."
python3 -m graph_net.model_path_handler \
    --model-path-list "${MODEL_LIST}" \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/op_names_extractor.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": { "resume": $RESUME, "model_path_prefix": "$GRAPH_NET_ROOT", "output_dir": "$OP_NAMES_DIR" }
}
EOF
)

# 3. Stage 2: Ranges
echo ">>> Running Stage 2: Ranges..."
python3 -m graph_net.apply_sample_pass \
    --model-path-list "${MODEL_LIST}" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/sample_pass/op_extract_points_generator.py" \
    --sample-pass-class-name "OpExtractPointsGenerator" \
    --sample-pass-config=$(base64 -w 0 <<EOF
{
    "model_path_prefix": "$GRAPH_NET_ROOT", "op_names_path_prefix": "$OP_NAMES_DIR",
    "output_dir": "$RANGES_DIR", "subgraph_ranges_file_name": "subgraph_ranges.json"
}
EOF
)

# 4. Stage 3: Decompose
echo ">>> Running Stage 3: Decompose..."
python3 -m graph_net.model_path_handler \
    --model-path-list "${MODEL_LIST}" \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py",
    "handler_class_name": "SubgraphGenerator",
    "handler_config": {
        "resume": $RESUME, "model_path_prefix": "$GRAPH_NET_ROOT", "output_dir": "$RAW_SUBGRAPH_DIR",
        "subgraph_ranges_json_root": "$RANGES_DIR", "subgraph_ranges_json_file_name": "subgraph_ranges.json",
        "group_head_and_tail": false, "chain_style": false, "device": "cuda"
    }
}
EOF
)

# 5. Generate generated_subgraphs_list.txt
echo ">>> Generating generated_subgraphs_list.txt..."
find ${RAW_SUBGRAPH_DIR} -name "model.py" \
    | xargs dirname \
    | xargs realpath --relative-to=${RAW_SUBGRAPH_DIR} \
    > "${WORKSPACE}/generated_subgraphs_list.txt"

# 6. Post-processing: Rename
echo ">>> Running Post-processing: Rename..."
python3 -m graph_net.model_path_handler \
    --model-path-list "${WORKSPACE}/generated_subgraphs_list.txt" \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/ast_graph_variable_renamer.py",
    "handler_class_name": "AstGraphVariableRenamer",
    "handler_config": {
        "device": "cuda", "try_run": false, "resume": $RESUME,
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

# 7. Post-processing: Deduplicate
echo ">>> Running Post-processing: Deduplicate..."
if [ -d "${DEDUPLICATED_DIR}" ]; then rm -rf "${DEDUPLICATED_DIR}"; fi

python3 -m graph_net.tools.deduplicated \
    --samples-dir ${RENAMED_DIR} \
    --target-dir ${DEDUPLICATED_DIR}

# Copy generated_subgraphs_list.txt to final output
cp "${WORKSPACE}/generated_subgraphs_list.txt" "${DEDUPLICATED_DIR}/"

echo ">>> ALL DONE. Final dataset located at: ${DEDUPLICATED_DIR}"
echo ">>> generated_subgraphs_list.txt also saved to: ${DEDUPLICATED_DIR}/generated_subgraphs_list.txt"
