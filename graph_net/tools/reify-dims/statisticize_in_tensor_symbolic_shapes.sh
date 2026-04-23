#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH_PREFIX="${1:?Usage: $0 <model_path_prefix>}"
MODEL_PATH_LIST="${2:-$MODEL_PATH_PREFIX/device_rewrited_sample_list_successed.txt}"

bash "$SCRIPT_DIR/get_in_tensor_symbolic_shapes.sh" "$MODEL_PATH_PREFIX" "$MODEL_PATH_LIST" \
    | grep get-in-tensor-symbolic-shapes \
    | awk '{print $2}' \
    | sort \
    | uniq -c \
    | sort -nk1
