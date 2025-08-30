#!/bin/bash
benchmark_dir="/work/GraphNet/benchmark_logs"
mkdir -p "${benchmark_dir}"

global_log="${benchmark_dir}/global_0830.log"
> "$global_log"

echo "[$(date)] Script started in background (PID: $$)" | tee -a "$global_log"
{
    valid_packages=("timm" "torchaudio" "torchgeometric" "torchvision" "transformers-auto-model" "ultralytics")
    for i in /work/GraphNet/samples/*/; do
        package_name=$(basename "${i%/}")
        if [[ " ${valid_packages[*]} " == *" ${package_name} "* ]]; then
            echo "[$(date)] Processing package: ${package_name}" | tee -a "$global_log"
            for j in "$i"*/; do
                model_name=$(basename "${j%/}")
                
                echo "Processing model: ${model_name}"
                
                python -m graph_net.torch.test_compiler \
                    --model-path "${j}" \
                    --compiler "inductor" \
                    --warmup 3 \
                    --trials 10 \
                    --device cuda \
                    --output-dir "${benchmark_dir}/${package_name}"
                
            done
        else
            echo "[$(date)] Skipping package (not in valid list): ${package_name}" | tee -a "$global_log"
        fi
    done
    echo "[$(date)] Script completed" | tee -a "$global_log"
} >> "$global_log" 2>&1


# nohup bash /work/GraphNet/graph_net/benchmark_demo.sh &

# python3 -m graph_net.analysis --test-compiler-log-file /work/GraphNet/global.log

# python3 -m graph_net.analysis --benchmark-path "${benchmark_dir}"