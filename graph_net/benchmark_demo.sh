#!/bin/bash

for i in /work/GraphNet/samples/torchvision/*/; do
    dir_name=$(basename "${i%/}")
    
    echo "Processing ${dir_name}"
    
    python -m graph_net.torch.test_compiler \
        --model-path "${i}" \
        --compiler "inductor" \
        --warmup 3 \
        --trials 10 \
        --device cuda \
        --output-dir "/work/GraphNet/benchmark_logs"
       
done  > torchvision_cuda.log 2>&1

# python3 -m graph_net.analysis --test-compiler-log-file /work/GraphNet/torchvision_cuda.log

python3 -m graph_net.analysis --benchmark-path /work/GraphNet/benchmark