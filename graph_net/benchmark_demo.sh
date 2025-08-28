#!/bin/bash

for i in /work/GraphNet/samples/torchvision/*/; do
    dir_name=$(basename "${i%/}")
    
    echo "Processing ${dir_name}"
    
    python -m graph_net.torch.test_compiler \
        --model-path "${i}" \
        --device cuda \
       
done  > torchvision_cuda.log 2>&1

python3 -m graph_net.analysis --benchmark-log-file /work/GraphNet/torchvision_cuda.log