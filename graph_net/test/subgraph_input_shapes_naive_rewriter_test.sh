#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

python3 -m graph_net.apply_sample_pass \
    --model-path-list "$GRAPH_NET_ROOT/graph_net/test/workspace_group_ranges_from_subgraph_sources/sample_list.txt" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/sample_pass/subgraph_input_shapes_naive_rewriter.py" \
    --sample-pass-class-name SubgraphInputShapesNaiveRewriter \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "resume": false,
    "model_path_prefix": "$GRAPH_NET_ROOT/graph_net/test/workspace_group_ranges_from_subgraph_sources",
    "subgraph_sources_json_root": "$GRAPH_NET_ROOT/graph_net/test/workspace_group_ranges_from_subgraph_sources",
    "shape_propagate_json_root": "$GRAPH_NET_ROOT/graph_net/test/workspace_shape_propagator",
    "subgraph_input_producer_indexes_json_root": "$GRAPH_NET_ROOT/graph_net/test/workspace_subgraph_input_producer_indexes_generator",
    "output_dir": "/tmp/workspace_subgraph_input_shapes_naive_rewriter"
}
EOF
)
