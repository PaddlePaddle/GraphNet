#!/bin/bash

AI4C_ROOT=$(python3 -c "import graph_net_bench; import os; print(os.path.dirname(os.path.dirname(graph_net_bench.__file__)))")
OUTPUT_PATH=/tmp/workspace_eval_device_diff_test
REFERENCE_DIR="$OUTPUT_PATH/reference"

mkdir -p "$OUTPUT_PATH"
mkdir -p "$REFERENCE_DIR"

MODEL_PATH="$AI4C_ROOT/samples/ultralytics/yolov3-tinyu"

echo "=========================================="
echo "Step 1: Generate reference on device A (simulated)"
echo "=========================================="
python3 -m graph_net.torch.test_reference_device \
    --model-path "$MODEL_PATH" \
    --compiler nope \
    --device cuda \
    --warmup 1 \
    --trials 1 \
    --reference-dir "$REFERENCE_DIR" \
    2>&1 | tee "$OUTPUT_PATH/reference.log"

echo ""
echo "=========================================="
echo "Step 2: Compare on device B (simulated)"
echo "=========================================="
python3 -m graph_net.torch.test_target_device \
    --model-path "$MODEL_PATH" \
    --device cuda \
    --reference-dir "$REFERENCE_DIR" \
    2>&1 | tee "$OUTPUT_PATH/target.log"

echo ""
echo "=========================================="
echo "Test completed. Logs saved to: $OUTPUT_PATH"
echo "=========================================="