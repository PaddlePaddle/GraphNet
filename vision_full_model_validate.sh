#!/usr/bin/env bash
# Batch-validate all extracted models under $GRAPH_NET_EXTRACT_WORKSPACE,
# and move any that fail validation into the fail_samples directory.

ws="${GRAPH_NET_EXTRACT_WORKSPACE:?GRAPH_NET_EXTRACT_WORKSPACE not set}"
fail_dir="/root/paddlejob/workspace/env_run/output/zhenghuaijin/GraphNet/fail_samples"

# ensure fail_samples exists
mkdir -p "$fail_dir"

for dir in "$ws"/*; do
    if [ -d "$dir" ]; then
        name="$(basename "$dir")"
        echo "Validating ${name}..."
        # run validation
        python -m graph_net.torch.validate --model-path "$dir"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "❌ Validation failed for ${name}, moving to ${fail_dir}"
            mv "$dir" "$fail_dir/"
        else
            echo "✅ Validation succeeded for ${name}"
        fi
    fi
done
