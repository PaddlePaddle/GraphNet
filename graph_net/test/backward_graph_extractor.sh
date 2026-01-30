#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")
GRAPHNET_ROOT="$GRAPH_NET_ROOT/../"

# Device rewrite pass
DEVICE_REWRITE_OUTPUT_DIR="/tmp/device_rewrited_samples"
mkdir -p "$DEVICE_REWRITE_OUTPUT_DIR"

python3 -m graph_net.model_path_handler \
    --model-path-list "graph_net/config/small100_torch_samples_list.txt" \
    --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPHNET_ROOT/graph_net/torch/sample_pass/device_rewrite_sample_pass.py",
    "handler_class_name": "DeviceRewriteSamplePass",
    "handler_config": {
        "device": "cuda",
        "resume": false,
        "model_path_prefix": "$GRAPHNET_ROOT",
        "output_dir": "$DEVICE_REWRITE_OUTPUT_DIR"
    }
}
EOF
)

echo "Device rewrite pass completed!"

# Backward graph extraction
OUTPUT_DIR="/tmp/backward_graph_samples"
mkdir -p "$OUTPUT_DIR"

python3 -m graph_net.apply_sample_pass \
    --model-path-list "graph_net/config/small100_torch_samples_list.txt" \
    --sample-pass-file-path "graph_net/torch/sample_pass/backward_graph_extractor.py" \
    --sample-pass-class-name "BackwardGraphExtractorPass" \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "model_path_prefix": "$DEVICE_REWRITE_OUTPUT_DIR",
    "output_dir": "$OUTPUT_DIR",
    "device": "cuda"
}
EOF
)

echo "Backward graph extraction completed!"