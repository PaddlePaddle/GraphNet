#!/usr/bin/env bash

GRAPH_NET_ROOT=$(python -c "import graph_net, os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
MODEL_PATH_PREFIX=$GRAPH_NET_ROOT
OUTPUT_DIR=/tmp/backward_graph_workspace
FRAMEWORK="torch"
HANDLER_CONFIG=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/backward_graph_extractor.py",
    "handler_class_name": "BackwardGraphExtractorPass",
    "handler_config": {
        "model_path_prefix": "$MODEL_PATH_PREFIX",
        "output_dir": "$OUTPUT_DIR",
        "device": "cuda",
        "resume": false
    }
}
EOF
)

run_case() {
    local rel_sample_path="$1"
    python -m graph_net.model_path_handler \
        --model-path "$rel_sample_path" \
        --handler-config "$HANDLER_CONFIG"
}

run_case "samples/torchvision/resnet18"
#run_case "samples/transformers-auto-model/albert-base-v2"
