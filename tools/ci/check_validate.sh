#!/bin/bash

set +ex

function LOG {
  echo [$0:${BASH_LINENO[0]}] $* >&2
}

LOG "[INFO] Start validate samples changed by pull request ..."

export GRAPH_NET_ROOT=$(cd $(dirname $0)/../.. && pwd)
export GRAPH_NET_EXTRACT_WORKSPACE="${GRAPH_NET_ROOT}/samples"
export PYTHONPATH=${GRAPH_NET_EXTRACT_WORKSPACE}:$PYTHONPATH

[ -z "$CUDA_VISIBLE_DEVICES" ] && CUDA_VISIBLE_DEVICES="0"

function prepare_env() {
  git config --global --add safe.directory "*"
  num_changed_samples=$(git diff --name-only develop | grep -E "samples/(.*\.py|.*\.json)" | wc -l)
  if [ ${num_changed_samples} -eq 0 ]; then
    python ${GRAPH_NET_ROOT}/tools/count_sample.py
    LOG "[INFO] This pull request doesn't change any samples, skip the CI."
    exit 0
  fi

  LOG "[INFO] Device Id: ${CUDA_VISIBLE_DEVICES}"
  # Update pip
  LOG "[INFO] Update pip ..."
  env http_proxy="" https_proxy="" pip install -U pip > /dev/null
  [ $? -ne 0 ] && LOG "[FATAL] Update pip failed!" && exit -1
  # install torch 
  pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
}

function check_validation() {
  LOG "[INFO] Start run validate for changed samples ..."
  MODIFIED_MODEL_PATHS=()
  for file in $(git diff --name-only develop | grep -E "samples/(.*\.py|.*\.json)")
  do
    LOG "[INFO] Found ${file} modified."
    model_path=$(dirname ${file})
    MODIFIED_MODEL_PATHS[${#MODIFIED_MODEL_PATHS[@]}]=$model_path
  done
  MODIFIED_MODEL_PATHS=($(echo ${MODIFIED_MODEL_PATHS[@]} | tr ' ' '\n' | sort | uniq))
  LOG "[INFO] Validation of these models will run: ${MODIFIED_MODEL_PATHS[@]}"
  fail_name=()
  for model_path in ${MODIFIED_MODEL_PATHS[@]}
  do
    python -m graph_net.torch.validate --model-path ${GRAPH_NET_EXTRACT_WORKSPACE}/${model_path} --graph-net-samples-path ${GRAPH_NET_EXTRACT_WORKSPACE}/samples >&2
    [ $? -ne 0 ] && fail_name[${#fail_name[@]}]="${model_path}"
  done
  if [ ${#fail_name[@]} -ne 0 ]
  then
    LOG "[FATAL] Failed tests: ${fail_name[@]}"
    echo ${fail_name[@]}
    exit -1
  fi
}

function summary_problems() {
  local check_validation_code=$1
  local check_validation_info=$2
  if [ $check_validation_code -ne 0 ]
  then
    LOG "[FATAL] ============================================"
    LOG "[FATAL] Summary problems:"
    LOG "[FATAL] === API test error - Please fix the failed API tests accroding to fatal log:"
    LOG "[FATAL] $check_validation_info"
    exit $check_validation_code
  fi
}

function main() {
  prepare_env
  check_validation_info=$(check_validation)
  check_validation_code=$?
  summary_problems $check_validation_code "$check_validation_info"
  python ${GRAPH_NET_ROOT}/tools/count_sample.py
  LOG "[INFO] check_validation run success and no error!"
}

main
