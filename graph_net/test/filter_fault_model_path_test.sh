#!/bin/bash

# Dynamically locate the graph_net package root for the log file path
GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")

# LOG_FILE needs to be dynamic to find the data on any machine
LOG_FILE="$GRAPH_NET_ROOT/test/data_calculate_es_scores/evaluation.log"

# Static prefix as recorded in the logs
MODEL_PREFIX="/workspace/GraphNet"
OUTPUT_FILE="/tmp/workspace_fault_filter/faulty_models.txt"

echo "[Info] Running FaultModelPathFilter..."

# Inline execution with base64 encoded config
python3 -m graph_net.filter_fault_model_path \
    "$(base64 -w 0 <<EOF
{
    "log_file_path": "$LOG_FILE",
    "output_txt_file_path": "$OUTPUT_FILE",
    "model_path_prefix": "$MODEL_PREFIX"
}
EOF
)"

# Check success and display the result file
if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
    echo "------------------------------------------"
    echo "[Success] Faulty models list (relative to $MODEL_PREFIX):"
    cat "$OUTPUT_FILE"
    echo "------------------------------------------"
else
    echo "[Error] Filter failed or output file not found."
    exit 1
fi
