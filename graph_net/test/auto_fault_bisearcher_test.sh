#!/bin/bash

# Resolve the root directory of the project
GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

# Test Environment Setup
MODEL_LIST="$GRAPH_NET_ROOT/graph_net/test/small10_torch_samples_list.txt"
MODEL_PREFIX="$GRAPH_NET_ROOT"
OUTPUT_DIR="/tmp/workspace_auto_fault_bisearcher"

# Execute the SamplePass via the standard CLI entry point
python3 -m graph_net.apply_sample_pass \
    --model-path-list "$MODEL_LIST" \
    --sample-pass-file-path "$GRAPH_NET_ROOT/graph_net/sample_pass/auto_fault_bisearcher.py" \
    --sample-pass-class-name AutoFaultBisearcher \
    --sample-pass-config $(base64 -w 0 <<EOF
{
    "model_path_prefix": "$MODEL_PREFIX",
    "output_dir": "$OUTPUT_DIR",
    "output_file_name": "truncate_size_has_fault.txt",
    
    "truncator_config": {
        "model_path_prefix": "$MODEL_PREFIX",
        "output_dir": "$OUTPUT_DIR/workspace_truncator/"
    },

    "evaluator_file_path": "$GRAPH_NET_ROOT/graph_net/fault_locator/torch/compiler_evaluator.py",
    "evaluator_class_name": "CompilerEvaluator",
    "evaluator_config": {
        "model_path_prefix": "$OUTPUT_DIR/workspace_truncator/",
        "output_dir": "$OUTPUT_DIR/torch_compiler_eval",
        "compiler": "nope",
        "device": "cuda"
    },

    "tolerance": 0
}
EOF
)
