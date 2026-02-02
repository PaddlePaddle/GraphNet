#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
# model_runnable_predicator=ShapePropagatablePredicator
model_runnable_predicator=ModelRunnablePredicator

python3 -m graph_net.model_path_handler \
    --use-subprocess                    \
    --model-path-list $GRAPH_NET_ROOT/config/small100_torch_samples_list.txt \
    --handler-config=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/constraint_util.py",
    "handler_class_name": "UpdateInputTensorConstraints",
    "handler_config": {
        "resume": true,
        "model_path_prefix": "$GRAPH_NET_ROOT/../",
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
        "last_model_log_file": "/tmp/a.py"
    }
}
EOF
)
