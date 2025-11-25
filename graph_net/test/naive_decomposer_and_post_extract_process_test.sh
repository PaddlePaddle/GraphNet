#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GRAPH_NET_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_ROOT=$(dirname "$GRAPH_NET_DIR")

# 将项目根目录加入Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
MODEL_NAME=resnet18
MODEL_PATH_IN_SAMPLES=/timm/$MODEL_NAME
decorator_config_json_str=$(cat <<EOF
{
    "decorator_path": "$GRAPH_NET_ROOT/torch/extractor.py",
    "decorator_config": {
        "name": "$MODEL_NAME",
        "custom_extractor_path": "$GRAPH_NET_ROOT/torch/naive_graph_decomposer.py",
        "custom_extractor_config": {
            "output_dir": "/work/.BCloud/countkernels/",
            "split_positions": [8, 16, 32],
            "group_head_and_tail": true,
            "filter_path":"$GRAPH_NET_ROOT/torch/naive_subgraph_filter.py",
            "filter_config": {},
            "post_extract_process_path":"$GRAPH_NET_ROOT/torch/post_extract_process.py",
            "post_extract_process_config": {
                "decorator_path": "$GRAPH_NET_ROOT/torch/shape_prop.py",
                "decorator_class_name": "ShapePropagate"
            }
        }
    }
}
EOF
)
DECORATOR_CONFIG=$(echo $decorator_config_json_str | base64 -w 0)

python3 -m graph_net.torch.run_model --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --decorator-config=$DECORATOR_CONFIG
