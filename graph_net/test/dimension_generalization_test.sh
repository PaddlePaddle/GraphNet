#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
config_json_str=$(cat <<EOF
{
    "decorator_path": "$GRAPH_NET_ROOT/torch/static_to_dynamic.py",
    "decorator_class_name": "StaticToDynamic",
    "decorator_config": {
        "output_dir": ""
    }
}
EOF
)
CONFIG=$(echo $config_json_str | base64 -w 0)

python3 -m graph_net.torch.run_model --model-path $GRAPH_NET_ROOT/../samples/transformers-auto-model/opus-mt-en-gmw --decorator-config=$CONFIG
