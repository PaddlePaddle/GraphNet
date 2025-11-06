#!/bin/bash

export CUDA_VISIBLE_DEVICES="5"

WORK_ROOT=/work

export PYTHONPATH=${WORK_ROOT}/GraphNet:${WORK_ROOT}/abstract_pass/Athena:$PYTHONPATH
export GRAPH_NET_EXTRACT_WORKSPACE=${WORK_ROOT}/GraphNet/tests/tmp

MODEL_NAMES=(
    "PP-OCRv3_mobile_rec"
    "PP-YOLOE-R-L"
    "PP-YOLOE_plus-M"
    "PP-YOLOE-S_human"
    "TimesNet_ad"
    "PP-YOLOE_plus-S"
    "PP-YOLOE-L_human"
    "PP-YOLOE_plus_SOD-L"
    "PP-YOLOE_plus_SOD-largesize-L"
    "Nonstationary"
    "PP-YOLOE-S_vehicle"
    "ch_SVTRv2_rec"
    "PP-YOLOE_plus-L"
    "PP-DocLayout-M"
    "PP-YOLOE_plus_SOD-S"
    "TimesNet"
    "PP-YOLOE-L_vehicle"
)

for model_name in "${MODEL_NAMES[@]}"; do
  echo "git add ${model_name}"
  git add ${model_name}
  
  for ((i=0;i<0;i++)); do
    python -m graph_net.paddle.validate \
      --dump-graph-hash-key \
      --graph-net-samples-path /work/GraphNet/paddle_samples/PaddleX \
      --model-path /work/GraphNet/paddle_samples/PaddleX/${model_name}/subgraph_${i}
  done
done
