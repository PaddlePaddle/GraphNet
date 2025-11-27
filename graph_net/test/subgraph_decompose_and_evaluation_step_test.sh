#!/bin/bash

export PYTHONPATH=/work/GraphNet:/work/abstract_pass/Athena:$PYTHONPATH

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")

FRAMEWORK="paddle"
LOG_FILE="$GRAPH_NET_ROOT/test/log_xpu.txt"
#LOG_FILE="/work/GraphNet/benchmark_results/log_test_target_device-xpu_p800_nope-pd_20251111_2.txt"
OUTPUT_DIR="outputs"
TOLERANCE=3
INITIAL_MAX_SIZE=4096

test_config_json_str=$(cat <<EOF
{ 
    "module_name": "graph_net.${FRAMEWORK}.test_compiler",
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
    "decorator_path": "$GRAPH_NET_ROOT/${FRAMEWORK}/extractor.py",
    "decorator_config": {
        "name": "PLACEHOLDER_NAME",
        "custom_extractor_path": "$GRAPH_NET_ROOT/${FRAMEWORK}/naive_graph_decomposer.py",
        "custom_extractor_config": {
            "output_dir": "PLACEHOLDER_DIR",
            "split_positions": [],
            "group_head_and_tail": true,
            "chain_style": false
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

python3 -m graph_net.subgraph_decompose_and_evaluation_step \
    --log-file="$LOG_FILE" \
    --output-dir="$OUTPUT_DIR" \
    --framework="${FRAMEWORK}" \
    --test-config="$TEST_CONFIG_B64" \
    --decorator-config="$EXTRACTOR_CONFIG_B64" \
    --tolerance="$TOLERANCE" \
    --max-subgraph-size="$INITIAL_MAX_SIZE"

echo ""
echo ">>> Pass execution finished."
echo ">>> Run this script again to execute the NEXT pass if needed."