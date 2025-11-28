#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")

LOG_FILE="$GRAPH_NET_ROOT/test/log_file_for_subgraph_decompose_and_evaluation_step.log"
OUTPUT_DIR="/tmp/decompose_and_evaluation_workspace"
TOLERANCE=3
INITIAL_MAX_SIZE=2048

test_config_json_str=$(cat <<EOF
{ 
    "module_name": "graph_net.torch.test_compiler",
    "arguments": {
        "compiler": "nope",
        "device": "cuda",
        "warmup": 5,
        "trials": 20
    }
}
EOF
)

TEST_CONFIG_B64=$(echo "$test_config_json_str" | base64 -w 0)

echo "Starting GraphNet Auto-Debugger"
echo "--------------------------------------------------------"
echo "Log File:    $LOG_FILE"
echo "Output Dir:  $OUTPUT_DIR"
echo "Init Size:   $INITIAL_MAX_SIZE"
echo "--------------------------------------------------------"

python3 -m graph_net.torch.subgraph_decompose_and_evaluation_step \
    --log-file="$LOG_FILE" \
    --output-dir="$OUTPUT_DIR" \
    --test-config="$TEST_CONFIG_B64" \
    --tolerance="$TOLERANCE" \
    --max-subgraph-size="$INITIAL_MAX_SIZE"

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Task failed! Please check logs and fix bugs before proceeding."
    exit 1
fi

echo ""
echo ">>> Pass execution finished."
echo ">>> Run this script again to execute the NEXT pass if needed."