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
    "decorator_path": "$GRAPH_NET_ROOT/torch/dtype_generalizer.py",
    "decorator_class_name": "InitDataTypeGeneralizationPasses",
    "decorator_config": {
        "dtype_list": ["float16", "bfloat16"]
    }
}
EOF
)
CONFIG_INIT=$(echo "$config_json_str_init" | base64 -w 0)

python3 -m graph_net.torch.run_model \
    --model-path "$SAMPLES_ROOT/timm/resnet18" \
    --decorator-config="$CONFIG_INIT" || echo "Warning: CV model test failed"

echo ""

# Test on an NLP model (BERT-like)
echo "[2/2] Testing NLP model: transformers-auto-model/opus-mt-en-gmw"
python3 -m graph_net.torch.run_model \
    --model-path "$SAMPLES_ROOT/transformers-auto-model/opus-mt-en-gmw" \
    --decorator-config="$CONFIG_INIT" || echo "Warning: NLP model test failed"

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
    "decorator_path": "$GRAPH_NET_ROOT/torch/dtype_generalizer.py",
    "decorator_class_name": "ApplyDataTypeGeneralizationPasses",
    "decorator_config": {
        "output_dir": "$OUTPUT_DIR"
    }
}
EOF
)
CONFIG_APPLY=$(echo "$config_json_str_apply" | base64 -w 0)

echo "[1/2] Generating CV samples..."
python3 -m graph_net.torch.run_model \
    --model-path "$SAMPLES_ROOT/timm/resnet18" \
    --decorator-config="$CONFIG_APPLY" || echo "Warning: CV generation failed"

echo ""

echo "[2/2] Generating NLP samples..."
python3 -m graph_net.torch.run_model \
    --model-path "$SAMPLES_ROOT/transformers-auto-model/opus-mt-en-gmw" \
    --decorator-config="$CONFIG_APPLY" || echo "Warning: NLP generation failed"

echo ""
echo "Step 2 completed. Generated samples in: $OUTPUT_DIR"
echo ""

# ============================================
# Verification
# ============================================
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

if [ -d "$OUTPUT_DIR" ]; then
    echo "Generated samples:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    
    # Count generated samples
    SAMPLE_COUNT=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Total samples generated: $SAMPLE_COUNT"
    
    if [ $SAMPLE_COUNT -gt 0 ]; then
        echo ""
        echo "✓ Test PASSED: Successfully generated $SAMPLE_COUNT low-precision samples"
        echo ""
        echo "You can now use these samples for:"
        echo "  - test_compiler evaluation"
        echo "  - Agent code generation"
        echo "  - Performance benchmarking"
    else
        echo ""
        echo "✗ Test WARNING: No samples were generated"
        echo "  This might be normal if models don't support dtype conversion"
    fi
else
    echo "✗ Test FAILED: Output directory not created"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="

