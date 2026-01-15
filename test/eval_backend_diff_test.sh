#!/bin/bash

AI4C_ROOT=$(python3 -c "import graph_net_bench; import os; print(os.path.dirname(os.path.dirname(graph_net_bench.__file__)))")
OUTPUT_PATH=/tmp/workspace_eval_backend_diff_test

mkdir -p "$OUTPUT_PATH"
model_list="$AI4C_ROOT/test/workspace_eval_backend_diff/sample_list.txt"

python3 -m graph_net_bench.torch.eval_backend_diff \
    --model-path-prefix $AI4C_ROOT/ \
    --allow-list $model_list \
    --compiler nope \
    --device cuda \
    --config $(base64 -w 0 <<EOF
{
}
EOF
) 2>&1 | tee "$OUTPUT_PATH/validation.log"
