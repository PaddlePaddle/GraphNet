#!/bin/bash
set -e

# ==============================================================================
# Configuration Area
# ==============================================================================

# [CRITICAL NOTICE]
# This script now uses dynamic path detection. 
# Ensure you are running inside the correct Virtual Environment.

# 1. Dynamic Path Retrieval (Fixing Hardcoded Paths)
# Detect python executable from current PATH
PYTHON_EXEC=$(which python3)
if [ -z "$PYTHON_EXEC" ]; then
    echo "Error: 'python3' not found in PATH. Please activate your virtualenv."
    exit 1
fi

# Detect Project Root dynamically by importing the module
GRAPH_NET_ROOT=$($PYTHON_EXEC -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
if [ -z "$GRAPH_NET_ROOT" ]; then
    echo "Error: Could not determine GRAPH_NET_ROOT. Ensure 'graph_net' is installed or in PYTHONPATH."
    exit 1
fi

# 2. Parallel Processing Config
AUTO_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$AUTO_GPUS" -eq 0 ]; then AUTO_GPUS=1; fi

# 逻辑：
# 1. 如果你运行命令带了参数 (e.g., ./script.sh 8)，就用参数值。
# 2. 否则，使用自动检测到的 GPU 数量。
NUM_GPUS=${1:-$AUTO_GPUS}

echo ">>> Detected/Set NUM_GPUS: ${NUM_GPUS}"

RESUME="false"

# 3. Workspace Setup
TIMESTAMP=$(date +%Y%m%d_%H%M)
WORKSPACE="/tmp/single_op_workspace_${TIMESTAMP}"
# You can override this via argument or env var
MODEL_LIST="${MODEL_LIST:-${GRAPH_NET_ROOT}/graph_net/config/small100_torch_samples_list.txt}"

# 4. Output Directories
OP_NAMES_DIR="${WORKSPACE}/01_op_names"
RANGES_DIR="${WORKSPACE}/02_ranges"
RAW_SUBGRAPH_DIR="${WORKSPACE}/03_raw_subgraphs"
RENAMED_DIR="${WORKSPACE}/04_renamed"
DEDUPLICATED_DIR="${WORKSPACE}/05_deduplicated"
LOG_DIR="${WORKSPACE}/logs"  # New: Dedicated log directory

export PYTHONPATH="${GRAPH_NET_ROOT}:${PYTHONPATH}"
export GRAPH_NET_ROOT PYTHON_EXEC WORKSPACE OP_NAMES_DIR RANGES_DIR RAW_SUBGRAPH_DIR RESUME LOG_DIR

mkdir -p "$WORKSPACE" "$LOG_DIR"

# ==============================================================================
# Core Logic: Single Model Processing (V3: Strict Error Checking)
# ==============================================================================
process_single_model() {
    local model_path=$1
    local gpu_id=$2
    
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    
    local safe_name=$(basename "$model_path")
    local tmp_list="${WORKSPACE}/tmp_list_${BASHPID}.txt"
    local log_file="${LOG_DIR}/${safe_name}_${BASHPID}.log"

    echo "${model_path}" > "${tmp_list}"
    echo "=== Processing ${model_path} ===" > "$log_file"

    run_step() {
        local step_name=$1
        local cmd_str=$2
        
        echo "---------------------------------------------------" >> "$log_file"
        echo ">>> Running Stage: ${step_name}" >> "$log_file"
        

        if ! eval "$cmd_str" >> "$log_file" 2>&1; then
            echo "[GPU ${gpu_id}] System Failed at ${step_name}: ${model_path}"
            return 1
        fi
        
        if grep -q -E "Traceback \(most recent call last\)|Error:|Exception:" "$log_file"; then
            echo "[GPU ${gpu_id}] Logic Failed at ${step_name} (Found Traceback): ${model_path}"
            echo "    -> Log saved at: ${log_file}"
            tail -n 5 "$log_file" | sed "s/^/[GPU ${gpu_id}] /"
            return 1
        fi
        
        return 0
    }

    # --- Stage 1: Op Names ---
    cmd_s1="$PYTHON_EXEC -m graph_net.model_path_handler --model-path-list ${tmp_list} --handler-config=\$(base64 -w 0 <<EOF
{
    \"handler_path\": \"$GRAPH_NET_ROOT/graph_net/torch/sample_pass/op_names_extractor.py\",
    \"handler_class_name\": \"OpNamesExtractor\",
    \"handler_config\": { \"resume\": $RESUME, \"model_path_prefix\": \"$GRAPH_NET_ROOT\", \"output_dir\": \"$OP_NAMES_DIR\" }
}
EOF
)"
    
    run_step "OpNames" "$cmd_s1" || { rm -f "${tmp_list}"; return 1; }

    # --- Stage 2: Ranges ---
    cmd_s2="$PYTHON_EXEC -m graph_net.apply_sample_pass --model-path-list ${tmp_list} --sample-pass-file-path \"$GRAPH_NET_ROOT/graph_net/sample_pass/op_extract_points_generator.py\" --sample-pass-class-name \"OpExtractPointsGenerator\" --sample-pass-config=\$(base64 -w 0 <<EOF
{
    \"model_path_prefix\": \"$GRAPH_NET_ROOT\", \"op_names_path_prefix\": \"$OP_NAMES_DIR\",
    \"output_dir\": \"$RANGES_DIR\", \"subgraph_ranges_file_name\": \"subgraph_ranges.json\"
}
EOF
)"
    run_step "Ranges" "$cmd_s2" || { rm -f "${tmp_list}"; return 1; }

    # --- Stage 3: Decompose ---
    cmd_s3="$PYTHON_EXEC -m graph_net.model_path_handler --model-path-list ${tmp_list} --handler-config=\$(base64 -w 0 <<EOF
{
    \"handler_path\": \"$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py\",
    \"handler_class_name\": \"SubgraphGenerator\",
    \"handler_config\": {
        \"resume\": $RESUME, \"model_path_prefix\": \"$GRAPH_NET_ROOT\", \"output_dir\": \"$RAW_SUBGRAPH_DIR\",
        \"subgraph_ranges_json_root\": \"$RANGES_DIR\", \"subgraph_ranges_json_file_name\": \"subgraph_ranges.json\",
        \"group_head_and_tail\": false, \"chain_style\": false, \"device\": \"cuda\"
    }
}
EOF
)"
    run_step "Decompose" "$cmd_s3" || { rm -f "${tmp_list}"; return 1; }

    rm -f "${tmp_list}"

    echo "[GPU ${gpu_id}] Done: ${model_path}"
}

export -f process_single_model

# ==============================================================================
# Helper Function: Subgraph List Generation
# ==============================================================================
function generate_subgraph_list() {
    local target_dir="$1"
    local sample_list="$2"
    echo ">>> Generating subgraph list for ${target_dir}..."
    find ${target_dir} -name "model.py" \
        | xargs dirname \
        | xargs realpath --relative-to=${target_dir} \
        | tee $sample_list > /dev/null
}

# ==============================================================================
# Main Pipeline Dispatcher
# ==============================================================================
function main() {
    echo ">>> Starting Pipeline..."
    echo "    Python: $PYTHON_EXEC"
    echo "    Root:   $GRAPH_NET_ROOT"
    echo "    Logs:   $LOG_DIR"
    
    # 1. Prepare Data
    if [ ! -f "$MODEL_LIST" ]; then
        echo "Error: Model list not found at $MODEL_LIST"
        exit 1
    fi
    grep -v "^#" "${MODEL_LIST}" | grep -v "^$" > "${WORKSPACE}/clean_list.txt"
    total_lines=$(wc -l < "${WORKSPACE}/clean_list.txt")
    
    echo ">>> Total Models: $total_lines | GPUS: $NUM_GPUS"

    # 2. Sharding
    lines_per_gpu=$(( (total_lines + NUM_GPUS - 1) / NUM_GPUS ))
    split -l ${lines_per_gpu} -d "${WORKSPACE}/clean_list.txt" "${WORKSPACE}/gpu_chunk_"

    # 3. Parallel Execution
    for (( i=0; i<NUM_GPUS; i++ )); do
        suffix=$(printf "%02d" $i)
        chunk_file="${WORKSPACE}/gpu_chunk_${suffix}"
        
        [ ! -f "$chunk_file" ] && continue
        
        echo ">>> Launching Worker for GPU $i..."
        (
            while read -r model_path; do
                process_single_model "$model_path" "$i" || true 
            done < "$chunk_file"
        ) & 
    done

    # 4. Wait
    echo ">>> Waiting for workers..."
    wait
    echo ">>> Generation Phase Complete."

    # ==========================================================================
    # Post-processing
    # ==========================================================================
    
    echo ">>> Starting Renaming Phase..."
    generate_subgraph_list ${RAW_SUBGRAPH_DIR} "${WORKSPACE}/raw_list.txt"

    # We redirect output to a main log file here because it's a single process
    $PYTHON_EXEC -m graph_net.model_path_handler \
        --model-path-list "${WORKSPACE}/raw_list.txt" \
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
) >> "${LOG_DIR}/renaming.log" 2>&1

    echo ">>> Starting Deduplication Phase..."
    if [ -d "${DEDUPLICATED_DIR}" ]; then rm -rf "${DEDUPLICATED_DIR}"; fi

    $PYTHON_EXEC -m graph_net.tools.deduplicated \
        --samples-dir ${RENAMED_DIR} \
        --target-dir ${DEDUPLICATED_DIR} >> "${LOG_DIR}/deduplication.log" 2>&1

    echo ">>> ALL DONE. Final dataset located at: ${DEDUPLICATED_DIR}"
    echo ">>> Check ${LOG_DIR} for error logs if any failures occurred."
}

main
