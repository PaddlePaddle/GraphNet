# !/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")
SAMPLES_ROOT="$GRAPH_NET_ROOT/../samples"
OUTPUT_DIR="/tmp/dtype_gen_samples"
mkdir -p "$OUTPUT_DIR"

# Step 1: Initialize dtype generalization passes (samples of torchvision)
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

torch_samples="graph_net/config/small100_torch_samples_list.txt"
# python3 -m graph_net.model_path_handler --model-path "timm/resnet18" --handler-config=$CONFIG_INIT
# python3 -m graph_net.model_path_handler --model-path "transformers-auto-model/opus-mt-en-gmw" --handler-config=$CONFIG_INIT

tail -n +2 $torch_samples | while read line 
do
    sample_path="${line#samples/}"
    python3 -m graph_net.model_path_handler --model-path $sample_path --handler-config=$CONFIG_INIT
done

# Step 2: Apply passes to generate samples
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

# python3 -m graph_net.model_path_handler --model-path "timm/resnet18" --handler-config=$CONFIG_APPLY
# python3 -m graph_net.model_path_handler --model-path "transformers-auto-model/opus-mt-en-gmw" --handler-config=$CONFIG_APPLY

tail -n +2 $torch_samples | while read line 
do
    sample_path="${line#samples/}"
    python3 -m graph_net.model_path_handler --model-path $sample_path --handler-config=$CONFIG_APPLY
done

# validation
for model_path in "$OUTPUT_DIR"/*; do
    echo "[VALIDATE] $model_path"

    output=$(python -m graph_net.torch.validate \
        --model-path "$model_path" 2>&1)

    if echo "$output" | grep -q "Validation success, model_path="; then
        echo "SUCCESS"
    else
        echo "FAIL"
    fi
done | tee >(grep SUCCESS | wc -l | xargs -I{} echo SUCCESS {}) | tee >(grep FAIL | wc -l | xargs -I{} echo FAIL {}); wait




