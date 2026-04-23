#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH_PREFIX="${1:?Usage: $0 <model_path_prefix> [num_gpus]}"
NUM_GPUS="${2:-8}"
LOG_DIR="${3:-/tmp/init-constraints-logs}"

mkdir -p "$LOG_DIR"

echo "Launching $NUM_GPUS GPU tasks for: $MODEL_PATH_PREFIX"
echo "Logs: $LOG_DIR/"

for ((i = 0; i < NUM_GPUS; i += 1)); do
    suffix=$(printf "%02d" $i)
    echo "Starting GPU $i (suffix=$suffix) ..."
    bash "$SCRIPT_DIR/init_input_tensor_constraints.sh" "$MODEL_PATH_PREFIX" "$i" "$suffix" \
        > "$LOG_DIR/gpu_${i}.log" 2>&1 &
done

echo "All tasks launched. Use 'bash $0/../monitor_tasks.sh $MODEL_PATH_PREFIX status' to check."
