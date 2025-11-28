#!/bin/bash

export PYTHONPATH=/work/GraphNet:/work/abstract_pass/Athena:$PYTHONPATH

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")

FRAMEWORK="paddle"
LOG_FILE="$GRAPH_NET_ROOT/test/log_xpu.txt"
#LOG_FILE="/work/GraphNet/benchmark_results/log_test_target_device-xpu_p800_nope-pd_20251111_2.txt"
OUTPUT_DIR="outputs"
TOLERANCE=0
INITIAL_MAX_SIZE=4096

#rm -rf ${OUTPUT_DIR}

test_compiler_config_str=$(cat <<EOF
{ 
    "test_module_name": "test_compiler",
    "test_compiler_arguments": {
        "model-path": null,
        "compiler": "nope",
        "device": "cuda",
        "warmup": 5,
        "trials": 20
    }
}
EOF
)

test_reference_device_config_str=$(cat <<EOF
{
    "test_module_name": "test_reference_device",
    "test_reference_device_arguments": {
        "model-path": null,
        "reference-dir": null,
        "compiler": "nope",
        "device": "cuda",
        "warmup": 5,
        "trials": 20
    }
}
EOF
)

test_target_device_config_str=$(cat <<EOF
{
    "test_module_name": "test_target_device",
    "test_target_device_arguments": {
        "model-path": null,
        "reference-dir": null,
        "device": "xpu"
    }
}
EOF
)

test_module_name="test_reference_device"
if [ "${test_module_name}" = "test_compiler" ]; then
    TEST_CONFIG_B64=$(echo "$test_compiler_config_str" | base64 -w 0)
elif [ "${test_module_name}" = "test_reference_device" ]; then
    TEST_CONFIG_B64=$(echo "$test_reference_device_config_str" | base64 -w 0)
elif [ "${test_module_name}" = "test_target_device" ]; then
    TEST_CONFIG_B64=$(echo "$test_reference_device_config_str" | base64 -w 0)
else
    echo "test_module_name (${test_module_name}) is unsupported!"
    exit
fi

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
    --tolerance="$TOLERANCE" \
    --max-subgraph-size="$INITIAL_MAX_SIZE"

echo ""
echo ">>> Pass execution finished."
echo ">>> Run this script again to execute the NEXT pass if needed."