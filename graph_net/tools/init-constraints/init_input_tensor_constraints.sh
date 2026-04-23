#!/bin/bash
set -euo pipefail

# Parse --fast flag
FAST=false
args=()
for arg in "$@"; do
    if [ "$arg" = "--fast" ]; then
        FAST=true
    else
        args+=("$arg")
    fi
done
set -- "${args[@]}"

MODEL_PATH_PREFIX="${1:?Usage: $0 [--fast] <model_path_prefix> <cuda_device> [suffix]}"
CUDA_DEVICE="${2:?Usage: $0 [--fast] <model_path_prefix> <cuda_device> [suffix]}"
SUFFIX="${3:-}"

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

if [ "$FAST" = true ]; then
    # ShapePropagatablePredicator: shape-propagation only, no actual model run.
    model_runnable_predicator=ShapePropagatablePredicator
else
    # ModelRunnablePredicator: actually runs the model (slower, more accurate).
    model_runnable_predicator=ModelRunnablePredicator
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH_PREFIX=$MODEL_PATH_PREFIX"
echo "SUFFIX=$SUFFIX"
echo "predicator=$model_runnable_predicator"

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

MODEL_PATH_LIST="${MODEL_PATH_PREFIX}/device_rewrited_sample_list_successed.txt${SUFFIX}"

python3 -m graph_net.model_path_handler \
    --use-subprocess                    \
    --model-path-list "$MODEL_PATH_LIST" \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/constraint_util.py",
    "handler_class_name": "UpdateInputTensorConstraints",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$MODEL_PATH_PREFIX/",
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/torch/constraint_util.py",
        "data_input_predicator_class_name": "NaiveDataInputPredicator",
        "model_runnable_predicator_filepath": "$GRAPH_NET_ROOT/torch/constraint_util.py",
        "model_runnable_predicator_class_name": "$model_runnable_predicator",
        "dimension_generalizer_filepath": "$GRAPH_NET_ROOT/torch/static_to_dynamic.py",
        "dimension_generalizer_class_name": "StaticToDynamic",
        "dimension_generalizer_config": {
            "pass_names": [
                "batch_call_method_view_pass",
                "tuple_arg_call_method_view_pass",
                "naive_call_method_reshape_pass",
                "naive_call_method_expand_pass",
                "non_batch_call_method_expand_pass",
                "non_batch_call_method_view_pass",
                "non_batch_call_function_arange_pass",
                "non_batch_call_function_getitem_slice_pass",
                "non_batch_call_function_full_pass",
                "non_batch_call_function_full_plus_one_pass",
                "non_batch_call_function_zeros_pass",
                "non_batch_call_function_arange_plus_one_pass"
            ]
        },
        "limits_handled_models": 999999,
        "last_model_log_file": "/dev/null"
    }
}
EOF
)
