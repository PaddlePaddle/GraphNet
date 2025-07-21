#!/usr/bin/env bash
set -euo pipefail

# 全局变量
model_name="Qwen/Qwen3-0.6B"
# model_name="bert-base-uncased"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_path_base="$script_dir/samples/torch"

text="Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal."

model_path="${model_path_base}/${model_name}"

# 调用 extractor
python -m graph_net.torch.extractor.LLM_extractor \
    --model-name "$model_name" \
    --model-path "$model_path" \
    --text "$text"
