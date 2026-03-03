#!/bin/bash

# Usage:
# 1. Startup server on reference side, use following command:
#   python3 -m graph_net_rpc.grpc.server --port 50052
#
# 2. Change the machine to the actual ip of server, then run this scipt on target side:
#   bash run_auto_fault_locator.sh

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")

FRAMEWORK="torch"
LOG_FILE="$GRAPH_NET_ROOT/test/log_file_for_subgraph_decompose_and_evaluation_step.log"
OUTPUT_DIR="/tmp/decompose_and_evaluation_workspace"
TOLERANCE="0 2"
INITIAL_MAX_SIZE=2048
DECOMPOSE_METHOD="fixed-start"
REFERENCE_DEVICE="cuda"
TARGET_DEVICE="cuda"
MACHINE="localhost"
PORT=50052

python3 -m graph_net.auto_fault_locator \
    --log-file "$LOG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --framework "${FRAMEWORK}" \
    --decompose-method "${DECOMPOSE_METHOD}" \
    --tolerance $TOLERANCE \
    --max-subgraph-size="$INITIAL_MAX_SIZE" \
    --reference-device "${REFERENCE_DEVICE}" \
    --target-device "${TARGET_DEVICE}" \
    --machine "${MACHINE}" \
    --port "${PORT}"