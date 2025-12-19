#!/usr/bin/env bash

GRAPH_NET_ROOT=$(python -c "import graph_net, os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
OUTPUT_DIR="/tmp/agent_unittests"
FRAMEWORK="torch"
HANDLER_CONFIG=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/sample_pass/agent_unittest_generator.py",
    "handler_class_name": "AgentUnittestGeneratorPass",
    "handler_config": {
        "framework": "${FRAMEWORK}",
        "model_path_prefix": "${GRAPH_NET_ROOT}",
        "output_dir": "$OUTPUT_DIR",
        "device": "auto",
        "generate_main": true,
        "try_run": true,
        "resume": false,
        "data_input_predicator_filepath": "$GRAPH_NET_ROOT/graph_net/${FRAMEWORK}/constraint_util.py",                                                                                     
        "data_input_predicator_class_name": "NaiveDataInputPredicator"
    }
}
EOF
)

run_case() {
  local rel_sample_path="$1"
  echo "[AgentTest] running $rel_sample_path"
  python -m graph_net.model_path_handler \
    --model-path "$rel_sample_path" \
    --handler-config "$HANDLER_CONFIG"
}

run_case "samples/torchvision/resnet18"
run_case "samples/transformers-auto-model/albert-base-v2"

echo "[AgentTest] done. Generated *_test.py files should now exist beside the samples."
