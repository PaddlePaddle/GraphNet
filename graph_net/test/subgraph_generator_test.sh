#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

python3 -m graph_net.apply_sample_pass \
    --model-path-list "$GRAPH_NET_ROOT/graph_net/test/workspace_subgraph_generator/sample_list.txt" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_generator.py" \
    --sample-pass-class-name SubgraphGenerator \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "output_dir": "/tmp/workspace_subgraph_generator",
    "subgraph_ranges_json_root": "$GRAPH_NET_ROOT/graph_net/test/workspace_subgraph_generator/",
    "use_all_inputs": true,
    "resume": false
}
EOF
)
