#!/usr/bin/env bash

set -e

MACHINE="${MACHINE:-localhost}"
PORT="${PORT:-50052}"
MODEL_PATH="${MODEL_PATH:-}"
RANDOM_SEED="${RANDOM_SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/sample_remote_executor_output}"
RPC_CMD="${RPC_CMD:-python3 -m graph_net.torch.test_reference_device}"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 --model-path PATH [--machine HOST] [--port PORT] [--random-seed SEED] [--output-dir DIR] [--rpc-cmd CMD]"
    exit 1
fi

python3 -m graph_net_rpc.test.sample_remote_executor_test \
    --machine "$MACHINE" \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --random-seed "$RANDOM_SEED" \
    --output-dir "$OUTPUT_DIR" \
    --rpc-cmd "$RPC_CMD"