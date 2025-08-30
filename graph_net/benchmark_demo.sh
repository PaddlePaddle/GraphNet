#!/bin/bash
# A benchmark script for GraphNet models.

benchmark_dir="/work/GraphNet/benchmark_logs"
samples_dir="/work/GraphNet/samples"
global_log="${benchmark_dir}/global.log"

mkdir -p "${benchmark_dir}"
> "$global_log"

for package_path in "${samples_dir}"/*/; do
    package_name=$(basename "${package_path%/}")
    output_dir="${benchmark_dir}/${package_name}"
    mkdir -p "${output_dir}"

    for model_path in "${package_path}"*/; do
        model_name=$(basename "${model_path%/}")
        {
            if ls "${output_dir}"/*"${model_name}"*.json > /dev/null 2>&1; then
                echo "[$(date)] SKIPPING: ${package_name}/${model_name} (JSON result already exists)"
            else
                echo "[$(date)] STARTING: ${package_name}/${model_name}"

                python -m graph_net.torch.test_compiler \
                    --model-path "${model_path}" \
                    --compiler "inductor" \
                    --warmup 3 \
                    --trials 10 \
                    --device "cuda" \
                    --output-dir "${output_dir}"

                echo "[$(date)] FINISHED: ${package_name}/${model_name}"
            fi
        } >> "$global_log" 2>&1 &
    done
done

echo "[$(date)] All tasks launched. Waiting for remaining background jobs to complete..." | tee -a "$global_log"
wait
echo "[$(date)] All jobs finished. Script completed." | tee -a "$global_log"
