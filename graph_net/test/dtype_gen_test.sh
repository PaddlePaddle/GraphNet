#!/bin/bash

# Data Type Generalization Test Script
# This script demonstrates how to use dtype generalization in two steps:
# 1. Initialize passes: Test which passes work and save to graph_net.json
# 2. Apply passes: Generate low-precision samples

set -e

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")
SAMPLES_ROOT="$GRAPH_NET_ROOT/../samples"

# Configuration
OUTPUT_DIR="/tmp/dtype_gen_samples"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Data Type Generalization Test"
echo "=========================================="
echo ""

# ============================================
# Step 1: Initialize dtype generalization passes
# ============================================
echo "Step 1: Initialize dtype generalization passes"
echo "------------------------------------------"
echo "Testing which passes work for each model..."
echo ""

# Test on a CV model (ResNet)
echo "[1/2] Testing CV model: timm/resnet18"
config_json_str_init=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/torch/dtype_generalizer.py",
    "handler_class_name": "InitDataTypeGeneralizationPasses",
    "handler_config": {
        "dtype_list": ["float16", "bfloat16"],
        "model_path_prefix": "$SAMPLES_ROOT"
    }
}
EOF
)
CONFIG_INIT=$(echo "$config_json_str_init" | base64 -w 0)

python3 -m graph_net.model_path_handler \
    --model-path "timm/resnet18" \
    --handler-config="$CONFIG_INIT" || echo "Warning: CV model test failed"

echo ""

# Test on an NLP model (BERT-like)
echo "[2/2] Testing NLP model: transformers-auto-model/opus-mt-en-gmw"
python3 -m graph_net.model_path_handler \
    --model-path "transformers-auto-model/opus-mt-en-gmw" \
    --handler-config="$CONFIG_INIT" || echo "Warning: NLP model test failed"

echo ""
echo "Step 1 completed. Pass names written to graph_net.json"
echo ""

# ============================================
# Step 2: Apply passes to generate samples
# ============================================
echo "Step 2: Apply dtype generalization passes"
echo "------------------------------------------"
echo "Generating low-precision samples..."
echo ""

config_json_str_apply=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/torch/dtype_generalizer.py",
    "handler_class_name": "ApplyDataTypeGeneralizationPasses",
    "handler_config": {
        "output_dir": "$OUTPUT_DIR",
        "model_path_prefix": "$SAMPLES_ROOT",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "RunModelPredicator",
        "model_runnable_predicator_config": {
            "use_dummy_inputs": true
        }
    }
}
EOF
)
CONFIG_APPLY=$(echo "$config_json_str_apply" | base64 -w 0)

echo "[1/2] Generating CV samples..."
python3 -m graph_net.model_path_handler \
    --model-path "timm/resnet18" \
    --handler-config="$CONFIG_APPLY" || echo "Warning: CV generation failed"

echo ""

echo "[2/2] Generating NLP samples..."
python3 -m graph_net.model_path_handler \
    --model-path "transformers-auto-model/opus-mt-en-gmw" \
    --handler-config="$CONFIG_APPLY" || echo "Warning: NLP generation failed"

echo ""
echo "Step 2 completed. Generated samples in: $OUTPUT_DIR"
echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
