#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
MODEL_NAME=resnet18
MODEL_PATH_IN_SAMPLES=/timm/$MODEL_NAME
INPUT_MODEL_LIST=$GRAPH_NET_ROOT/test/dev_model_list/get_fusible_subgraph_sample_list.txt
# INPUT_MODEL_LIST=$GRAPH_NET_ROOT/test/dev_model_list/small_sample_list_for_get_fusible_subgraph.txt

OUTPUT_DIR="/tmp/find_fully_fusible_output_test5"
config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/torch/fully_fusible_subgraph_extractor.py",
    "handler_class_name":"FullyFusibleSubgraphExtractor",
    "handler_config": {
        "resume": false,
        "model_path_prefix": "$GRAPH_NET_ROOT/../",
        "output_dir": "$OUTPUT_DIR",
        "nn_module_fully_fusible_decorator_path": "$GRAPH_NET_ROOT/torch/count_kernels_util.py",
        "nn_module_fully_fusible_decorator_class_name": "TorchSubModuleFullyFusibleDecorator",
        "subgraph_max_size": 10,
        "subgraph_min_size": 1,
        "graph_max_search_nodes": 20
    }
}
EOF
)
CONFIG=$(echo $config_json_str | base64 -w 0)

# python3 -m graph_net.model_path_handler --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --handler-config=$CONFIG
python3 -m graph_net.model_path_handler --model-path-list $INPUT_MODEL_LIST --handler-config=$CONFIG &> /work/.BCloud/cmds/fusibletest.log
#"model_path_prefix": "$GRAPH_NET_ROOT/../",
        # "model_path_prefix": "/tmp/decompose_workspace/",