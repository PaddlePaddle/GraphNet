#!/bin/bash
set -x

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
OUTPUT_PATH=/tmp/op_names_workspace

model_list="$GRAPH_NET_ROOT/graph_net/config/small10_torch_samples_list.txt"

python3 -m graph_net.model_path_handler \
    --model-path-list $model_list \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/op_names_extractor.py",
    "handler_class_name": "OpNamesExtractor",
    "handler_config": {
        "resume": false,
        "model_path_prefix": "$GRAPH_NET_ROOT",
        "output_dir": "$OUTPUT_PATH",
        "limits_handled_models": 1
    }
}
EOF
)

# Fix all op_names.txt to append a newline character at the end.
find ${OUTPUT_PATH} -name op_names.txt -type f -exec sh -c '
  tail -c1 "$1" | read -r _ || echo >> "$1"
' sh {} \;

find ${OUTPUT_PATH} -name op_names.txt -type f -exec cat {} + | sort | uniq -c | sort -nr > ${OUTPUT_PATH}/op_names_freq.txt
