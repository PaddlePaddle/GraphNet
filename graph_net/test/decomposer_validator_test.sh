#!/bin/bash

if [ -z "$GRAPH_NET_BENCHMARK_PATH" ]; then
    GRAPH_NET_BENCHMARK_PATH="$(pwd)/graphnet_benchmark"
fi

FILE_PATH=$GRAPH_NET_BENCHMARK_PATH/decomposer
mkdir -p "$(dirname "$FILE_PATH/log.log")"

MODEL_PATH="./todo_works/range_decomposer_validator/test/resnet18"

python -m graph_net.torch.test_compiler \
    --model-path $MODEL_PATH \
    --compiler range_decomposer_validator \
    --device cuda > "$FILE_PATH/log.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: decomposer_validator execution failed"
    echo "Please check the log file: $FILE_PATH/log.log"
    exit 1
fi

python -m graph_net.log2json \
    --log-file "$FILE_PATH/log.log" \
    --output-dir "$FILE_PATH/JSON_results/"

python -m graph_net.plot_ESt \
    --benchmark-path "$FILE_PATH/JSON_results/" \
    --output-dir "$FILE_PATH"

echo "=================================================="
echo "Results saved in: $FILE_PATH/ES_result.png"
echo ""
echo "IMPORTANT: Please verify if the curve in ES_result.png is a straight line"
echo "If the curve is NOT a straight line, please check the log file: $FILE_PATH/log.log"
echo "=================================================="