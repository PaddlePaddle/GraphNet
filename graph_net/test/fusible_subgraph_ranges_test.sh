#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

python3 -m graph_net.model_path_handler \
    --model-path-list "$GRAPH_NET_ROOT/graph_net/test/dev_model_list/cumsum_num_kernels_sample_list.txt" \
    --handler-config $(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/fusible_subgraph_ranges_generator.py",
    "handler_class_name": "FusibleSubgraphRangesGenerator",
    "handler_config": {
        "resume": false,
        "model_path_prefix": "$GRAPH_NET_ROOT/graph_net/test/workspace_cumsum_num_kernels",
        "input_json_file_name": "cumsum_num_kernels.json",
        "output_json_file_name": "fusible_subgraph_ranges.json",
        "output_dir": "/tmp/workspace_fusible_subgraph_ranges"
    }
}
EOF
)
