#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")

OUTPUT_DIR=/tmp/workspace_local_graph_decomposer_wrapper_test

python3 -m graph_net.local_graph_decomposer_wrapper \
    --framework torch \
    --model-name hf-tiny-model-private_tiny-random-BlipModel \
    --model-path $GRAPH_NET_ROOT/../samples/transformers-auto-model/hf-tiny-model-private_tiny-random-BlipModel \
    --output-dir $OUTPUT_DIR \
    --split-positions-json WzAsIDUxMl0=
