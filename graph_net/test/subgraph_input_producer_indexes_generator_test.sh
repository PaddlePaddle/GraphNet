#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

python3 -m graph_net.apply_sample_pass \
    --model-path-list "$GRAPH_NET_ROOT/graph_net/test/workspace_subgraph_input_producer_indexes_generator/sample_list.txt" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/torch/sample_pass/subgraph_input_producer_indexes_generator.py" \
    --sample-pass-class-name SubgraphInputProducerIndexesGenerator \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "resume": false,
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "subgraph_ranges_json_root": "$GRAPH_NET_ROOT/graph_net/test/workspace_subgraph_input_producer_indexes_generator",
    "subgraph_ranges_json_file_name": "grouped_ranges_from_subgraph_sources.json",
    "subgraph_ranges_json_key": "grouped_ranges_from_subgraph_sources",
    "output_dir": "/tmp/workspace_subgraph_input_producer_indexes_generator"
}
EOF
)
