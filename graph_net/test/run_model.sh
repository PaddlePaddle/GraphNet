#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
# MODEL_NAME=opus-mt-en-gv
# MODEL_PATH_PREFIX=$GRAPH_NET_ROOT/../samples/transformers-auto-model/
MODEL_NAME=opus-mt-en-gv__S0_128_S1_64
MODEL_PATH_PREFIX=/tmp/dimension_generalized_samples/samples/transformers-auto-model/opus-mt-en-gv
MODEL_PATH=$MODEL_PATH_PREFIX/$MODEL_NAME

DECORATOR_CONFIG=$(echo $decorator_config_json_str | base64 -w 0)

python3 -m graph_net.torch.run_model --model-path $MODEL_PATH
