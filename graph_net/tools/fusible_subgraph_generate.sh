#!/bin/bash

# GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

# model_path_list=$GRAPH_NET_ROOT/graph_net/test/dev_model_list/cumsum_num_kernels_sample_list.txt
# WORKSPACE_ROOT=/tmp/fusible_subgraphs
# CUMSUM_NUM_KERNELS_WORKSPACE=$WORKSPACE_ROOT/workspace_cumsum_num_kernels
# FUSIBLE_SUBGRAPH_RANGES_WORKSPACE=$WORKSPACE_ROOT/workspace_fusible_subgraph_ranges
# FUSIBLE_SUBGRAPH_SAMPLES_WORKSPACE=$WORKSPACE_ROOT/workspace_fusible_subgraph_samples
# resume=false

# python3 -m graph_net.model_path_handler \
#     --model-path-list "$model_path_list" \
#     --handler-config $(base64 -w 0 <<EOF
# {
#     "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_passes/cumsum_num_kernels_generator.py",
#     "handler_class_name": "CumSumNumKernelsGenerator",
#     "handler_config": {
#         "output_json_file_name": "cumsum_num_kernels.json",
#         "model_path_prefix": "$GRAPH_NET_ROOT",
#         "output_dir": "$CUMSUM_NUM_KERNELS_WORKSPACE",
#         "resume": $resume
#     }
# }
# EOF
# )

# python3 -m graph_net.model_path_handler \
#     --model-path-list "$model_path_list" \
#     --handler-config $(base64 -w 0 <<EOF
# {
#     "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/fusible_subgraph_ranges_generator.py",
#     "handler_class_name": "FusibleSubgraphRangesGenerator",
#     "handler_config": {
#         "model_path_prefix": "$CUMSUM_NUM_KERNELS_WORKSPACE",
#         "input_json_file_name": "cumsum_num_kernels.json",
#         "output_json_file_name": "fusible_subgraph_ranges.json",
#         "output_dir": "$FUSIBLE_SUBGRAPH_RANGES_WORKSPACE",
#         "resume": $resume
#     }
# }
# EOF
# )

# python3 -m graph_net.model_path_handler \
#     --model-path-list "$model_path_list" \
#     --handler-config $(base64 -w 0 <<EOF
# {
#     "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_passes/subgraph_generator.py",
#     "handler_class_name": "SubgraphGenerator",
#     "handler_config": {
#         "model_path_prefix": "$GRAPH_NET_ROOT",
#         "output_dir": "$FUSIBLE_SUBGRAPH_SAMPLES_WORKSPACE",
#         "subgraph_ranges_json_root": "$FUSIBLE_SUBGRAPH_RANGES_WORKSPACE",
#         "resume": $resume
#     }
# }
# EOF
# )

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

model_path_list=$GRAPH_NET_ROOT/graph_net/test/dev_model_list/cumsum_num_kernels_sample_list.txt
WORKSPACE_ROOT=/tmp/fusible_subgraphs
CUMSUM_NUM_KERNELS_WORKSPACE=$WORKSPACE_ROOT/workspace_cumsum_num_kernels
FUSIBLE_SUBGRAPH_RANGES_WORKSPACE=$WORKSPACE_ROOT/workspace_fusible_subgraph_ranges
FUSIBLE_SUBGRAPH_SAMPLES_WORKSPACE=$WORKSPACE_ROOT/workspace_fusible_subgraph_samples
resume=false

python3 -m graph_net.apply_sample_pass \
    --model-path-list "$model_path_list" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/torch/sample_passes/cumsum_num_kernels_generator.py" \
    --sample-pass-class-name "CumSumNumKernelsGenerator" \
    --sample-pass-config "$(cat <<EOF
{
    "output_json_file_name": "cumsum_num_kernels.json",
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "output_dir": "$CUMSUM_NUM_KERNELS_WORKSPACE",
    "resume": $resume
}
EOF
)"

python3 -m graph_net.apply_sample_pass \
    --model-path-list "$model_path_list" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/sample_pass/fusible_subgraph_ranges_generator.py" \
    --sample-pass-class-name "FusibleSubgraphRangesGenerator" \
    --sample-pass-config "$(cat <<EOF
{
    "model_path_prefix": "$CUMSUM_NUM_KERNELS_WORKSPACE",
    "input_json_file_name": "cumsum_num_kernels.json",
    "output_json_file_name": "fusible_subgraph_ranges.json",
    "output_dir": "$FUSIBLE_SUBGRAPH_RANGES_WORKSPACE",
    "resume": $resume
}
EOF
)"

python3 -m graph_net.apply_sample_pass \
    --model-path-list "$model_path_list" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/sample_pass/subgraph_generator.py" \
    --sample-pass-class-name "SubgraphGenerator" \
    --sample-pass-config "$(cat <<EOF
{
    "model_path_prefix": "$GRAPH_NET_ROOT",
    "output_dir": "$FUSIBLE_SUBGRAPH_SAMPLES_WORKSPACE",
    "subgraph_ranges_json_root": "$FUSIBLE_SUBGRAPH_RANGES_WORKSPACE",
    "resume": $resume
}
EOF
)"