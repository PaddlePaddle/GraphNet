#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")

LOG_FILE="$GRAPH_NET_ROOT/test/log_file_for_subgraph_decompose_and_evaluation_step.log"
OUTPUT_DIR="$GRAPH_NET_ROOT/../decomposed_samples"
TOLERANCE=3
INITIAL_MAX_SIZE=4096 

test_config_json_str=$(cat <<EOF
{ 
    "module_name": "graph_net.torch.test_compiler",
    "arguments": {
        "compiler": "nope",
        "device": "cuda",
        "warmup": 5,
        "trials": 20
    }
}
EOF
)

extractor_config_json_str=$(cat <<EOF
{
    "decorator_path": "$GRAPH_NET_ROOT/torch/extractor.py",
    "decorator_config": {
        "name": "PLACEHOLDER_NAME",
        "custom_extractor_path": "$GRAPH_NET_ROOT/torch/naive_graph_decomposer.py",
        "custom_extractor_config": {
            "output_dir": "PLACEHOLDER_DIR",
            "split_positions": [],
            "group_head_and_tail": false,
            "chain_style": false,
            "filter_path": "$GRAPH_NET_ROOT/torch/naive_subgraph_filter.py",
            "filter_config": {}
        }
    }
}
EOF
)

TEST_CONFIG_B64=$(echo "$test_config_json_str" | base64 -w 0)
EXTRACTOR_CONFIG_B64=$(echo "$extractor_config_json_str" | base64 -w 0)

echo "Starting GraphNet Auto-Debugger"
echo "--------------------------------------------------------"
echo "Log File:    $LOG_FILE"
echo "Output Dir:  $OUTPUT_DIR"
echo "Init Size:   $INITIAL_MAX_SIZE"
echo "--------------------------------------------------------"

python3 -m graph_net.torch.subgraph_decompose_and_evaluation_step \
    --log-file="$LOG_FILE" \
    --output-dir="$OUTPUT_DIR" \
    --test-config="$TEST_CONFIG_B64" \
    --decorator-config="$EXTRACTOR_CONFIG_B64" \
    --tolerance="$TOLERANCE" \
    --max-subgraph-size="$INITIAL_MAX_SIZE"

echo ""
echo ">>> Pass execution finished."
echo ">>> Run this script again to execute the NEXT pass if needed."