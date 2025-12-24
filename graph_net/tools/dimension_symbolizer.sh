#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(os.path.dirname(graph_net.__file__)))")

python3 -m graph_net.apply_sample_pass \
    --model-path-list $GRAPH_NET_ROOT/graph_net/config/small_sample_list_for_get_fusible_subgraph.txt \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/dimension_symbolizer.py" \
    --sample-pass-class-name "DimensionSymbolizer" \
    --sample-pass-config "$(cat <<EOF
{
    "resume": false,
    "output_dir": "/tmp/workspace_dimension_symbolizer",
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "limits_handled_models": 10,
    "last_model_log_file": "/tmp/a.py"
}
EOF
)"