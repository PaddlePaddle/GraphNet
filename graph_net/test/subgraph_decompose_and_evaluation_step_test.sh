#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")

FRAMEWORK="torch"
LOG_FILE="$GRAPH_NET_ROOT/test/log_file_for_subgraph_decompose_and_evaluation_step.log"
OUTPUT_DIR="/tmp/workspace_subgraph_decompose_and_evaluation_step_test"
TOLERANCE="0 2"
INITIAL_MAX_SIZE=2048

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

test_remote_reference_device_config_str=$(cat <<EOF
{
    "test_module_name": "test_remote_reference_device",
    "test_remote_reference_device_arguments": {
        "model-path": null,
        "reference-dir": null,
        "compiler": "nope",
        "device": "cuda",
        "op-lib": "default",
        "warmup": 5,
        "trials": 20,
        "seed": 123,
        "machine": "localhost",
        "port": 50052,
        "rpc-cmd": "python3 -m graph_net.torch.test_reference_device"
    }
}
EOF
)

test_module_name="test_compiler"
if [ "${test_module_name}" = "test_compiler" ]; then
    TEST_CONFIG_B64=$(echo "$test_compiler_config_str" | base64 -w 0)
elif [ "${test_module_name}" = "test_reference_device" ]; then
    TEST_CONFIG_B64=$(echo "$test_reference_device_config_str" | base64 -w 0)
elif [ "${test_module_name}" = "test_target_device" ]; then
    TEST_CONFIG_B64=$(echo "$test_target_device_config_str" | base64 -w 0)
elif [ "${test_module_name}" = "test_remote_reference_device" ]; then
    TEST_CONFIG_B64=$(echo "$test_remote_reference_device_config_str" | base64 -w 0)
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
    --log-file "$LOG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --framework "${FRAMEWORK}" \
    --test-config "$TEST_CONFIG_B64" \
    --decompose-method "fixed-start" \
    --tolerance $TOLERANCE \
    --max-subgraph-size "$INITIAL_MAX_SIZE"

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Task failed! Please check logs and fix bugs before proceeding."
else
    echo ""
    echo ">>> Pass execution finished."
    echo ">>> Run this script again to execute the NEXT pass if needed."
fi
