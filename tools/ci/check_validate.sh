#!/bin/bash

set +ex

function LOG {
  echo [$0:${BASH_LINENO[0]}] $* >&2
}

LOG "[INFO] Start validate samples changed by pull request ..."

export GRAPH_NET_EXTRACT_WORKSPACE=$(cd $(dirname $0)/../.. && pwd)
export PYTHONPATH=${GRAPH_NET_EXTRACT_WORKSPACE}:$PYTHONPATH

[ -z "$CUDA_VISIBLE_DEVICES" ] && CUDA_VISIBLE_DEVICES="0"

function prepare_torch_env() {
  git config --global --add safe.directory "*"
  num_changed_samples=$(git diff --name-only develop | grep -E "\bsamples\b/(.*\.py|.*\.json)" | wc -l)
  if [ ${num_changed_samples} -ne 0 ]; then
    LOG "[INFO] Device Id: ${CUDA_VISIBLE_DEVICES}"
    # Update pip
    LOG "[INFO] Update pip ..."
    env http_proxy="" https_proxy="" pip install -U pip > /dev/null
    [ $? -ne 0 ] && LOG "[FATAL] Update pip failed!" && exit -1
    # install torch
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118 > /dev/null
    [ $? -ne 0 ] && LOG "[FATAL] Install torch2.6.0 failed!" && exit -1
  else
    python ${GRAPH_NET_EXTRACT_WORKSPACE}/tools/count_sample.py
    LOG "[INFO] This pull request doesn't change any torch samples, skip the CI."
  fi
}

function prepare_paddle_env() {
  git config --global --add safe.directory "*"
  num_changed_paddle_samples=$(git diff --name-only develop | grep -E "\bpaddle_samples\b/(.*\.py|.*\.json)" | wc -l)
  if [ ${num_changed_paddle_samples} -ne 0 ]; then
    LOG "[INFO] Device Id: ${CUDA_VISIBLE_DEVICES}"
    # Update pip
    LOG "[INFO] Update pip ..."
    env http_proxy="" https_proxy="" pip install -U pip > /dev/null
    [ $? -ne 0 ] && LOG "[FATAL] Update pip failed!" && exit -1
    # install paddle
    pip uninstall torch==2.7.0 --yes
    LOG "[INFO] Install paddlepaddle-develop ..."
    python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/ > /dev/null
    [ $? -ne 0 ] && LOG "[FATAL] Install paddlepaddle-develop failed!" && exit -1
    python -c "import paddle; print('[PaddlePaddle Commit]', paddle.version.commit)"
  else
    python ${GRAPH_NET_EXTRACT_WORKSPACE}/tools/count_sample.py
    LOG "[INFO] This pull request doesn't change any paddle samples, skip the CI."
  fi
}

function check_torch_validation() {
  LOG "[INFO] Start run validate for changed torch samples ..."
  MODIFIED_MODEL_PATHS=()
  for file in $(git diff --name-only develop | grep -E "\bsamples\b/(.*\.py|.*\.json)")
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

function check_paddle_validation() {
  LOG "[INFO] Start run validate for changed paddle samples ..."
  MODIFIED_MODEL_PATHS=()
  for file in $(git diff --name-only develop | grep -E "\bpaddle_samples\b/(.*\.py|.*\.json)")
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
    python -m graph_net.paddle.validate --model-path ${GRAPH_NET_EXTRACT_WORKSPACE}/${model_path} --graph-net-samples-path ${GRAPH_NET_EXTRACT_WORKSPACE}/samples >&2
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
  prepare_torch_env
  check_validation_info=$(check_torch_validation)
  check_validation_code=$?
  summary_problems $check_validation_code "$check_validation_info"
  prepare_paddle_env
  check_validation_info=$(check_paddle_validation)
  check_validation_code=$?
  summary_problems $check_validation_code "$check_validation_info"
  python ${GRAPH_NET_EXTRACT_WORKSPACE}/tools/count_sample.py
  LOG "[INFO] check_validation run success and no error!"
}

main