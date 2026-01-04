#!/bin/bash

# GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
# os.path.dirname(graph_net.__file__))")
# SAMPLES_ROOT="$GRAPH_NET_ROOT/../samples"
# OUTPUT_DIR="/tmp/dtype_gen_samples"
# mkdir -p "$OUTPUT_DIR"

# # Step 1: Initialize dtype generalization passes
# config_json_str_init=$(cat <<EOF
# {
#     "handler_path": "$GRAPH_NET_ROOT/torch/dtype_generalizer.py",
#     "handler_class_name": "InitDataTypeGeneralizationPasses",
#     "handler_config": {
#         "dtype_list": ["float16", "bfloat16"],
#         "model_path_prefix": "$SAMPLES_ROOT"
#     }
# }
# EOF
# )
# CONFIG_INIT=$(echo "$config_json_str_init" | base64 -w 0)

# # python3 -m pdb -m graph_net.model_path_handler --model-path "timm/resnet18" --handler-config=$CONFIG_INIT
# python3 -m graph_net.model_path_handler --model-path "timm/resnet18" --handler-config=$CONFIG_INIT
# python3 -m graph_net.model_path_handler --model-path "transformers-auto-model/opus-mt-en-gmw" --handler-config=$CONFIG_INIT

# # Step 2: Apply passes to generate samples
# config_json_str_apply=$(cat <<EOF
# {
#     "handler_path": "$GRAPH_NET_ROOT/torch/dtype_generalizer.py",
#     "handler_class_name": "ApplyDataTypeGeneralizationPasses",
#     "handler_config": {
#         "output_dir": "$OUTPUT_DIR",
#         "model_path_prefix": "$SAMPLES_ROOT",
#         "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/torch/constraint_util.py",
#         "model_runnable_predicator_class_name": "RunModelPredicator",
#         "model_runnable_predicator_config": {
#             "use_dummy_inputs": true
#         }
#     }
# }
# EOF
# )
# CONFIG_APPLY=$(echo "$config_json_str_apply" | base64 -w 0)

# python3 -m graph_net.model_path_handler --model-path "timm/resnet18" --handler-config=$CONFIG_APPLY
# python3 -m graph_net.model_path_handler --model-path "transformers-auto-model/opus-mt-en-gmw" --handler-config=$CONFIG_APPLY

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")
SAMPLES_ROOT="$GRAPH_NET_ROOT/../samples"
TORCHVISION_ROOT="$SAMPLES_ROOT/torchvision"
OUTPUT_DIR="/tmp/dtype_gen_samples"
mkdir -p "$OUTPUT_DIR"

echo "GRAPH_NET_ROOT = $GRAPH_NET_ROOT"
echo "Testing samples under: $TORCHVISION_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo

# Step 1: Initialize dtype generalization passes
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

# Step 2: Apply passes
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

# Collect all torchvision model paths
MODEL_LIST=()
while IFS= read -r -d '' d; do
    rel_path="${d#$SAMPLES_ROOT/}"
    MODEL_LIST+=("$rel_path")
done < <(find "$TORCHVISION_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)

TOTAL=${#MODEL_LIST[@]}
SUCCESS=0
FAIL=0

echo "Found $TOTAL torchvision models"
echo "----------------------------------------"

# Run dtype generalization
for model_path in "${MODEL_LIST[@]}"; do
    echo "[INIT ] $model_path"
    python3 -m graph_net.model_path_handler \
        --model-path "$model_path" \
        --handler-config="$CONFIG_INIT" \
        >/dev/null 2>&1

    echo "[APPLY] $model_path"
    if python3 -m graph_net.model_path_handler \
        --model-path "$model_path" \
        --handler-config="$CONFIG_APPLY" \
        >/dev/null 2>&1
    then
        echo "SUCCESS"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
    fi
    echo
done

# Statistics
echo "========================================"
echo "Torchvision dtype generalization summary"
echo "----------------------------------------"
echo "Total models   : $TOTAL"
echo "Success        : $SUCCESS"
echo "Failed         : $FAIL"

if [ "$TOTAL" -gt 0 ]; then
    RATE=$(awk "BEGIN { printf \"%.2f\", ($SUCCESS / $TOTAL) * 100 }")
    echo "Success rate   : $RATE %"
fi
echo "========================================"


# validation
SAMPLES_DIR="/tmp/dtype_gen_samples"

TOTAL=0
SUCCESS=0
FAIL=0

echo "Validating samples under: $SAMPLES_DIR"
echo "----------------------------------------"

for model_path in "$SAMPLES_DIR"/*; do
    if [ ! -d "$model_path" ]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))
    echo "[VALIDATE] $model_path"

    output=$(python -m graph_net.torch.validate \
        --model-path "$model_path" 2>&1)

    if echo "$output" | grep -q "Validation success, model_path="; then
        echo "SUCCESS"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
        echo "  ---- error output ----"
        echo "$output" | sed 's/^/  /'
        echo "  ----------------------"
    fi

    echo
done

echo "========================================"
echo "Validation summary"
echo "----------------------------------------"
echo "Total models : $TOTAL"
echo "Success      : $SUCCESS"
echo "Failed       : $FAIL"

if [ "$TOTAL" -gt 0 ]; then
    RATE=$(awk "BEGIN { printf \"%.2f\", ($SUCCESS / $TOTAL) * 100 }")
    echo "Success rate : $RATE %"
fi
echo "========================================"

