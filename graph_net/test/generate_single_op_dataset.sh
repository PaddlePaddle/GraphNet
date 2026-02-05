#!/bin/bash 
set -x

################################################################################
# [CRITICAL NOTICE] BEFORE RUNNING THIS SCRIPT:
#
#  /graph_net/test/generate_single_op_dataset.sh
#
# 1. Check 'PYTHON_EXEC': Ensure the variable below points to the correct
#    Python interpreter in your virtual environment.
#
# 2. Check 'INPUT_LIST': Look for the 'INPUT_LIST' variable inside the
#    internal Python script (Stage 1 section). It is currently hardcoded
#    to 'small10_torch_samples_list.txt'. Please switch it to your full
#    dataset list file before running large-scale generation.
################################################################################

# ==============================================================================
# Configuration Area
# ==============================================================================

# [TODO] HARDCODED: Paths are currently hardcoded; needs dynamic retrieval or arguments in the future.
# Virtual Environment Python Executable Path
PYTHON_EXEC="/workspace/venv_graphnet/bin/python3"
# Project Root Directory
GRAPH_NET_ROOT="/workspace/GraphNet"

# Script Runtime Arguments
GPU_ID=${1:-0}
RESUME="false"

# Export environment variables to ensure Python can find graph_net
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${GRAPH_NET_ROOT}:${PYTHONPATH}"

# Workspace Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORKSPACE="/tmp/single_op_workspace_${TIMESTAMP}"

# Define standardized output directory structure
RAW_OUTPUT_DIR="${WORKSPACE}/01_raw_single_op_subgraphs"
RENAMED_OUTPUT_DIR="${WORKSPACE}/02_renamed_single_op_subgraphs"
DEDUPLICATED_OUTPUT_DIR="${WORKSPACE}/03_deduplicated_single_op_subgraphs"

# Define intermediate list file paths
RAW_SAMPLE_LIST="${WORKSPACE}/sample_list_01_raw.txt"
RENAMED_SAMPLE_LIST="${WORKSPACE}/sample_list_02_renamed.txt"

# Create workspace
mkdir -p "$WORKSPACE" "$RAW_OUTPUT_DIR"

# ==============================================================================
# Helper Functions
# ==============================================================================

# Subgraph list generation function (mimics the original script)
function generate_subgraph_list() {
    local target_dir="$1"
    local sample_list="$2"
    echo ">>> Generate subgraph_sample_list for samples under ${target_dir}."
    echo ">>>"
    
    # Find parent directories of all model.py files to identify valid samples
    find ${target_dir} -name "model.py" \
        | xargs dirname \
        | xargs realpath --relative-to=${target_dir} \
        | tee $sample_list
}

# ==============================================================================
# Stage 1: Generation (Black Box Mode)
# ==============================================================================

function generate_raw_data() {
    echo ">>> [1] Generating Single Operator Subgraphs (Running Python Script)..."
    echo ">>>"

    local TEMP_GEN_SCRIPT="${WORKSPACE}/_internal_gen.py"

    # 1. Write the Python script to a temporary file
    # Note: The Python logic is preserved exactly as provided
    cat << 'EOF' > "$TEMP_GEN_SCRIPT"
import os
import sys
import time
import math
import subprocess
import datetime
import multiprocessing
import json
import base64

# [TODO] HARDCODED: Keep sync with Shell script
PYTHON_EXEC = "/workspace/venv_graphnet/bin/python3"
PROJECT_ROOT = "/workspace/GraphNet"
# [dependency] WARNING: This is currently pointing to the small 10 sample list
INPUT_LIST = os.path.join(PROJECT_ROOT, "graph_net/config/small10_torch_samples_list.txt")

NUM_GPUS = 2
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Temporary directory generated inside the Python script
BASE_DIR = f"/tmp/decompose_run_{TIMESTAMP}"

def make_config_b64(config_dict):
    json_str = json.dumps(config_dict)
    return base64.b64encode(json_str.encode('utf-8')).decode('utf-8')

def run_stage_cmd(env, cwd, cmd_args, stage_name, log_file):
    cmd = [PYTHON_EXEC, "-u", "-m", "graph_net.apply_sample_pass"] + cmd_args
    try:
        result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        with open(log_file, "a") as f:
            if result.returncode != 0:
                f.write(f"\n[FAIL] {stage_name} Error (Exit {result.returncode}):\n")
                f.write(result.stderr[-2000:] + "\n")
                return False
            else:
                return True
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"\n[CRITICAL] {stage_name} Exception: {str(e)}\n")
        return False

def worker_process(gpu_id, models, base_dir):
    log_file = os.path.join(base_dir, "logs", f"worker_gpu{gpu_id}.log")
    workspace = base_dir
    ranges_dir = os.path.join(workspace, "workspace_single_operator_ranges")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["GRAPH_NET_ROOT"] = PROJECT_ROOT
    env["PYTHONPATH"] = PROJECT_ROOT
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_file, "w") as f:
        f.write(f"==== Worker GPU {gpu_id} Started ====\n")
    
    for idx, model_path in enumerate(models):
        model_output_dir = os.path.join(workspace, model_path)
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(ranges_dir, exist_ok=True)

        # Stage 1: OpNamesExtractor
        cfg_s1 = make_config_b64({
            "resume": False, 
            "model_path_prefix": PROJECT_ROOT, 
            "output_dir": workspace
        })
        run_stage_cmd(env, PROJECT_ROOT, [
            "--model-path", model_path,
            "--sample-pass-file-path", f"{PROJECT_ROOT}/graph_net/torch/sample_pass/op_names_extractor.py",
            "--sample-pass-class-name", "OpNamesExtractor",
            "--sample-pass-config", cfg_s1
        ], "Stage 1", log_file)

        # Stage 2: OpExtractPointsGenerator
        cfg_s2 = make_config_b64({
            "resume": False, 
            "model_path_prefix": PROJECT_ROOT, 
            "op_names_path_prefix": workspace, 
            "output_dir": ranges_dir, 
            "subgraph_ranges_file_name": "subgraph_ranges.json"
        })
        run_stage_cmd(env, PROJECT_ROOT, [
            "--model-path", model_path,
            "--sample-pass-file-path", f"{PROJECT_ROOT}/graph_net/sample_pass/op_extract_points_generator.py",
            "--sample-pass-class-name", "OpExtractPointsGenerator",
            "--sample-pass-config", cfg_s2
        ], "Stage 2", log_file)

        # Stage 3: SubgraphGenerator
        cfg_s3 = make_config_b64({
            "resume": False, 
            "model_path_prefix": PROJECT_ROOT, 
            "output_dir": workspace, 
            "subgraph_ranges_json_root": ranges_dir, 
            "group_head_and_tail": False, 
            "chain_style": False
        })
        run_stage_cmd(env, PROJECT_ROOT, [
            "--model-path", model_path,
            "--sample-pass-file-path", f"{PROJECT_ROOT}/graph_net/torch/sample_pass/subgraph_generator.py",
            "--sample-pass-class-name", "SubgraphGenerator",
            "--sample-pass-config", cfg_s3
        ], "Stage 3", log_file)

def main():
    if not os.path.exists(PYTHON_EXEC): return
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    
    # This line is the key anchor for the Shell script to capture the path
    print(f"Workspace: {BASE_DIR}") 
    print(f"Dataset Generation Started...")

    with open(INPUT_LIST, 'r') as f:
        all_models = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    
    chunk_size = math.ceil(len(all_models) / NUM_GPUS)
    processes = []
    for i in range(NUM_GPUS):
        chunk = all_models[i*chunk_size : (i+1)*chunk_size]
        if not chunk: continue
        p = multiprocessing.Process(target=worker_process, args=(i, chunk, BASE_DIR))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
EOF

    # 2. Execute the Python script and capture the output directory
    # We use tee to output logs to screen and grep to capture the "Workspace: " line
    echo ">>> Running internal python generator..."
    PYTHON_OUTPUT=$($PYTHON_EXEC $TEMP_GEN_SCRIPT | tee /dev/tty)
    
    # Extract the generated temporary path
    TEMP_SRC_DIR=$(echo "$PYTHON_OUTPUT" | grep "Workspace:" | awk '{print $2}' | tr -d '\r')

    if [ -z "$TEMP_SRC_DIR" ]; then
        echo "Error: Could not capture workspace path from python script."
        exit 1
    fi

    echo ">>> Python script finished. Temporary output at: $TEMP_SRC_DIR"
    
    # 3. Move Step
    echo ">>> Moving data from temp dir to standardized dir: $RAW_OUTPUT_DIR"
    # We only move the generated subgraph folders, excluding logs and range files.
    # Assuming subgraphs are generated inside model directories under BASE_DIR, 
    # we move everything first, then clean up.
    
    # Move all content
    cp -r ${TEMP_SRC_DIR}/* ${RAW_OUTPUT_DIR}/
    
    # Clean up unnecessary intermediate artifacts (ranges and logs), keeping only subgraphs
    rm -rf ${RAW_OUTPUT_DIR}/logs
    rm -rf ${RAW_OUTPUT_DIR}/workspace_single_operator_ranges
    
    echo ">>> Data moved and cleaned."
}

# ==============================================================================
# Stage 2: Renaming
# ==============================================================================

function rename_subgraphs() {
    echo ">>> [2] Rename subgraph samples under ${RAW_OUTPUT_DIR}."
    echo ">>>"
    
    # First, generate the list
    generate_subgraph_list ${RAW_OUTPUT_DIR} ${RAW_SAMPLE_LIST}

    $PYTHON_EXEC -m graph_net.model_path_handler \
        --model-path-list ${RAW_SAMPLE_LIST} \
        --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/ast_graph_variable_renamer.py",
    "handler_class_name": "AstGraphVariableRenamer",
    "handler_config": {
        "device": "cuda",
        "try_run": false,
        "resume": ${RESUME},
        "model_path_prefix": "${RAW_OUTPUT_DIR}",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "ModelRunnablePredicator",
        "output_dir": "$RENAMED_OUTPUT_DIR"
    }
}
EOF
)
}

# ==============================================================================
# Stage 3: Deduplication
# ==============================================================================

function deduplicate_subgraphs() {
    echo ">>> [3] Remove duplicated subgraph samples under ${RENAMED_OUTPUT_DIR}."
    echo ">>>"

    
    if [ -d "${DEDUPLICATED_OUTPUT_DIR}" ]; then
        echo ">>> Target directory exists. Cleaning up..."
        rm -rf "${DEDUPLICATED_OUTPUT_DIR}"
    fi
    
    $PYTHON_EXEC -m graph_net.tools.deduplicated \
        --samples-dir ${RENAMED_OUTPUT_DIR} \
        --target-dir ${DEDUPLICATED_OUTPUT_DIR}
}

# ==============================================================================
# Main Workflow
# ==============================================================================

main() {
    echo "=========================================================="
    echo "START: Single Operator Dataset Generation Pipeline"
    echo "Workspace: $WORKSPACE"
    echo "=========================================================="

    # 1. Generate raw data
    generate_raw_data

    # 2. Rename variables (Standardization)
    rename_subgraphs

    # 3. Deduplicate
    deduplicate_subgraphs

    echo "=========================================================="
    echo "FINISH: Dataset generated at ${DEDUPLICATED_OUTPUT_DIR}"
    echo "=========================================================="
}

main
