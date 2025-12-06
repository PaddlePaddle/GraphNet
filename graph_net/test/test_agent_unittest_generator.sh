#!/usr/bin/env bash
set -euo pipefail

# Smoke tests for AgentUnittestGenerator on one CV and one NLP sample (Torch side).
# It runs run_model with the decorator, which will drop a *_test.py under each sample directory.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TORCH_RUN="python -m graph_net.torch.run_model"

CV_SAMPLE="$ROOT_DIR/samples/torchvision/resnet18"
NLP_SAMPLE="$ROOT_DIR/samples/transformers-auto-model/albert-base-v2"

encode_cfg() {
  MODEL_PATH="$1" python - <<'PY'
import base64, json, os
cfg = {
    "decorator_class_name": "AgentUnittestGenerator",
    "decorator_config": {
        "model_path": os.environ["MODEL_PATH"],
        "force_device": "auto",
        "output_path": None,
        "use_dummy_inputs": False,
    },
}
print(base64.b64encode(json.dumps(cfg).encode()).decode())
PY
}

run_case() {
  local sample_path="$1"
  local name="$2"
  echo "[AgentTest] running $name sample at $sample_path"
  cfg_b64="$(encode_cfg "$sample_path")"
  $TORCH_RUN --model-path "$sample_path" --decorator-config "$cfg_b64"
}

run_case "$CV_SAMPLE" "CV (torchvision/resnet18)"
run_case "$NLP_SAMPLE" "NLP (transformers-auto-model/albert-base-v2)"

echo "[AgentTest] done. Generated *_test.py files should now exist beside the samples."
