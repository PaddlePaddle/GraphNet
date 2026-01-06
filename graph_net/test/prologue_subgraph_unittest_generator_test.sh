#!/usr/bin/env bash

GRAPH_NET_ROOT=$(python -c "import graph_net, os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

MODEL_PATH=paddle_samples/PaddleX/ResNet18
decorator_config_json_str=$(cat <<EOF
{
    "decorator_path": "$GRAPH_NET_ROOT/graph_net/paddle/extractor.py",
    "decorator_config": {
        "name": "ResNet18",
        "custom_extractor_path": "$GRAPH_NET_ROOT/graph_net/paddle/prologue_subgraph_unittest_generator.py",
        "custom_extractor_config": {
            "output_dir": "/tmp/prologue_unittests",
            "subgraph_range": [0, 6],
            "device": "auto",
            "tolerance": 0,
            "try_run": true
        }
    }
}
EOF
)
DECORATOR_CONFIG=$(echo $decorator_config_json_str | base64 -w 0)

python3 -m graph_net.paddle.run_model --model-path $GRAPH_NET_ROOT/$MODEL_PATH --decorator-config=$DECORATOR_CONFIG 