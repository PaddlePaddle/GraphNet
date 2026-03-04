#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

python3 -m graph_net.apply_sample_pass \
    --model-path-list "$GRAPH_NET_ROOT/graph_net/test/workspace_group_ranges_from_subgraph_sources/sample_list.txt" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/sample_pass/group_ranges_from_subgraph_sources.py" \
    --sample-pass-class-name GroupRangesFromSubgraphSources \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "subgraph_model_path_prefix": "$GRAPH_NET_ROOT/graph_net/test/workspace_group_ranges_from_subgraph_sources",
    "output_dir": "/tmp/workspace_group_ranges_from_subgraph_sources"
}
EOF
)
