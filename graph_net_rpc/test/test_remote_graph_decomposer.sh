#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64/:/usr/local/lib/:$LD_LIBRARY_PATH
set -e  
GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(graph_net.__file__))")
# Test configuration
MODEL_NAME="resnet18"
MODEL_PATH="$GRAPH_NET_ROOT/../samples/timm/$MODEL_NAME"
OUTPUT_DIR="/tmp/remote_graph_decomposer"
MACHINE="localhost"
PORT=50052

# Create decorator config (similar to naive_graph_decomposer_test.sh)
config_json_str=$(cat <<EOF
{
    "decorator_path": "$GRAPH_NET_ROOT/torch/extractor.py",
    "decorator_config": {
        "name": "$MODEL_NAME",
        "custom_extractor_path": "$GRAPH_NET_ROOT/torch/graph_decomposer.py",
        "custom_extractor_config": {
            "output_dir": "$OUTPUT_DIR/samples",
            "split_positions": [8, 16],
            "group_head_and_tail": true,
            "use_all_inputs": true,
            "chain_style": false
        }
    }
}
EOF
)
# Encode config to base64
DECORATOR_CONFIG=$(echo "$config_json_str" | base64 -w 0)
# Clean up previous test output
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning up previous test output..."
    rm -rf "$OUTPUT_DIR"
fi
# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the remote graph decomposer
echo "Running remote_graph_decomposer.py..."
python3 -m graph_net.torch.remote_graph_decomposer \
    --model-path "$MODEL_PATH" \
    --decorator-config "$DECORATOR_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --machine "$MACHINE" \
    --port "$PORT" \
    --rpc-cmd "python3 -m graph_net.torch.remote_graph_decomposer_entry"

