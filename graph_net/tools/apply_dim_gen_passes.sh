#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

python3 -m graph_net.apply_sample_pass \
    --model-path-list $GRAPH_NET_ROOT/config/torch_samples_list.txt \
    --sample-pass-file-path "$GRAPH_NET_ROOT/dimension_generalizer.py" \
    --sample-pass-class-name "ApplyDimGenPasses" \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "resume": false,
    "output_dir": "/tmp/dimension_generalized_samples",
    "model_path_prefix": "$GRAPH_NET_ROOT/../",
    "dimension_generalizer_filepath": "$GRAPH_NET_ROOT/torch/static_to_dynamic.py",
    "dimension_generalizer_class_name": "StaticToDynamic",
    "limits_handled_models": 40,
    "last_model_log_file": "/tmp/a.py"
}
EOF
)
