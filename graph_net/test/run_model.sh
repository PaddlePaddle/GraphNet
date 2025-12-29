#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

MODEL_LIST_FILE="/work/.BCloud/run_model_use_small_sample_list.log"
RUN_LOG="/work/.BCloud/run_model_samll.log"

while IFS= read -r FULL_PATH || [[ -n "$FULL_PATH" ]]
do
    python3 -m graph_net.torch.run_model --model-path "$FULL_PATH" >> "$RUN_LOG" 2>&1
done < "$MODEL_LIST_FILE"
