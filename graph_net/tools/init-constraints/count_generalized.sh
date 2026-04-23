#!/bin/bash
set -euo pipefail

MODEL_PATH_PREFIX="${1:?Usage: $0 <model_path_prefix>}"

echo "Counting samples with generalized (symbolic) dimensions..."
count=$(find "$MODEL_PATH_PREFIX" -name 'input_tensor_constraints.py' \
    | xargs grep -l 'dynamic_dim_constraint_symbols = \[S' 2>/dev/null \
    | wc -l)
echo "Total generalized: $count"
