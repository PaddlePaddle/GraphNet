#!/bin/bash
set -euo pipefail

MODEL_PATH_PREFIX="${1:?Usage: $0 <model_path_prefix>}"

echo "Counting samples with initialized input tensor constraints..."
count=$(find "$MODEL_PATH_PREFIX" -name 'input_tensor_constraints.py' \
    | xargs grep -l 'dynamic_dim_constraint_symbols = \[' 2>/dev/null \
    | wc -l)
echo "Total handled: $count"
