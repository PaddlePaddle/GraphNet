#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

MODEL_LIST_FILE="$GRAPH_NET_ROOT/test/dev_model_list/run_model_use_small_sample_list.log"
RUN_LOG="$GRAPH_NET_ROOT/test/run_model.log"

while IFS= read -r FULL_PATH || [[ -n "$FULL_PATH" ]]
do
    python3 -m graph_net.torch.run_model --model-path "$FULL_PATH" >> "$RUN_LOG" 2>&1
done < "$MODEL_LIST_FILE"
