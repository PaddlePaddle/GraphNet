#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
OUTPUT_DIR="/tmp/dtype_gen_samples"

mkdir -p $OUTPUT_DIR

model_list=${GRAPH_NET_ROOT}/graph_net/config/small10_torch_samples_list.txt
model_path_prefix="${GRAPH_NET_ROOT}"

# Step 1: Initialize dtype generalization passes (samples of torchvision)
python3 -m graph_net.apply_sample_pass \
    --use-subprocess \
    --model-path-list $model_list \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/dtype_generalizer.py" \
    --sample-pass-class-name InitDataTypeGeneralizationPasses \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "dtype_list": ["float16", "bfloat16"],
    "model_path_prefix": "$model_path_prefix",
    "output_dir": "$OUTPUT_DIR",
    "resume": true,
    "limits_handled_models": 10
}
EOF
)

# Step 2: Apply passes to generate samples
python3 -m graph_net.apply_sample_pass \
    --use-subprocess \
    --model-path-list $model_list \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/dtype_generalizer.py" \
    --sample-pass-class-name ApplyDataTypeGeneralizationPasses \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "output_dir": "$OUTPUT_DIR",
    "model_path_prefix": "$model_path_prefix",
    "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/torch/constraint_util.py",
    "device": "cuda",
    "resume": false,
    "limits_handled_models": 10,
    "try_run": true
}
EOF
)