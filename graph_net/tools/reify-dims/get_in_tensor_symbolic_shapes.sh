#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH_PREFIX="${1:?Usage: $0 <model_path_prefix>}"
MODEL_PATH_LIST="${2:-$MODEL_PATH_PREFIX/device_rewrited_sample_list_successed.txt}"

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

python3 -m graph_net.model_path_handler \
    --model-path-list "$MODEL_PATH_LIST" \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/tools/_get_in_tensor_symbolic_shapes.py",
    "handler_class_name": "GetInTensorSymbolicShapes",
    "handler_config": {
        "ignore_reified": true,
        "model_path_prefix": "$MODEL_PATH_PREFIX/"
    }
}
EOF
)
