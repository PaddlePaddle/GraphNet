#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")
GRAPHNET_ROOT="$GRAPH_NET_ROOT/../"
OUTPUT_DIR="/tmp/dtype_gen_samples"
mkdir -p "$OUTPUT_DIR"

# Step 1: Initialize dtype generalization passes (samples of torchvision)
python3 -m graph_net.apply_sample_pass \
    --model-path-list "graph_net/config/small100_torch_samples_list.txt" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/torch/sample_pass/dtype_generalizer.py" \
    --sample-pass-class-name InitDataTypeGeneralizationPasses \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "dtype_list": ["float16", "bfloat16"],
    "model_path_prefix": "$GRAPHNET_ROOT",
    "output_dir": "$OUTPUT_DIR",
    "resume": true,
    "limits_handled_models": null
}
EOF
) 

# Step 2: Apply passes to generate samples
python3 -m graph_net.apply_sample_pass \
    --model-path-list "graph_net/config/small100_torch_samples_list.txt" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/torch/sample_pass/dtype_generalizer.py" \
    --sample-pass-class-name ApplyDataTypeGeneralizationPasses \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "output_dir": "$OUTPUT_DIR",
    "model_path_prefix": "$GRAPHNET_ROOT",
    "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/torch/constraint_util.py",
    "model_runnable_predicator_class_name": "RunModelPredicator",
    "model_runnable_predicator_config": {
        "use_dummy_inputs": true
    },
    "resume": true,
    "limits_handled_models": null
}
EOF
)


