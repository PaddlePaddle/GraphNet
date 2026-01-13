#!/usr/bin/env bash

set -e

MODEL_PATH="${MODEL_PATH:-/tmp/demo_model}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/remote_output}"
MACHINE="${MACHINE:-localhost}"
PORT="${PORT:-50052}"
RPC_CMD="${RPC_CMD:-python3 -m graph_net.torch.test_reference_device}"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 --model-path PATH [--machine HOST] [--port PORT] [--random-seed SEED] [--output-dir DIR] [--rpc-cmd CMD]"
    exit 1
fi

python3 -m graph_net.torch.test_remote_reference_device \
    --model-path "$MODEL_PATH" \
    --reference-dir "$OUTPUT_DIR" \