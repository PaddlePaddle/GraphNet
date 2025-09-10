#!/bin/bash

set +ex

function LOG {
  echo [$0:${BASH_LINENO[0]}] $* >&2
}

LOG "[INFO] Start validate samples changed by pull request ..."

export GRAPH_NET_EXTRACT_WORKSPACE=$(cd $(dirname $0)/../.. && pwd)
export PYTHONPATH=${GRAPH_NET_EXTRACT_WORKSPACE}:$PYTHONPATH

[ -z "$CUDA_VISIBLE_DEVICES" ] && CUDA_VISIBLE_DEVICES="0"

function check_paths_without_spaces() {
  LOG "[INFO] Checking for spaces in modified file paths..."
  git config --global --add safe.directory "*"
  mapfile -t MODIFIED_FILES < <(git diff --diff-filter=ACM --name-only develop | grep -E "\bsamples\b/|\bpaddle_samples\b/")
  
  for file in "${MODIFIED_FILES[@]}"; do
    if [[ "${file}" == *" "* ]]; then
      LOG "[FATAL] File path contains spaces, which is not allowed. Please rename the file or directory."
      LOG "[FATAL] Offending path: '${file}'"
      exit 1
    fi
  done
  LOG "[INFO] No spaces found in file paths. Continuing..."
}

function prepare_torch_env() {
  git config --global --add safe.directory "*"
  num_changed_samples=$(git diff --diff-filter=ACM --name-only develop | grep -E "\bsamples\b/.*/(model\.py|input_meta\.py|weight_meta\.py)$" | wc -l)
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
  num_changed_paddle_samples=$(git diff --diff-filter=ACM --name-only develop | grep -E "\bpaddle_samples\b/.*/(model\.py|input_meta\.py|weight_meta\.py)$" | wc -l)
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
  for file in $(git diff --diff-filter=ACM --name-only develop | grep -E "\bsamples\b/.*/(model\.py|input_meta\.py|weight_meta\.py)$")
  do
    if [[ ${file} != *shape_patches_* ]]; then
      LOG "[INFO] Found ${file} modified."
      model_path=$(dirname ${file})
      MODIFIED_MODEL_PATHS[${#MODIFIED_MODEL_PATHS[@]}]=$model_path
    fi
  done
  MODIFIED_MODEL_PATHS=($(echo ${MODIFIED_MODEL_PATHS[@]} | tr ' ' '\n' | sort | uniq))
  local total_models=${#MODIFIED_MODEL_PATHS[@]}
  if [ ${total_models} -eq 0 ]; then
    LOG "[INFO] No changed torch sample models detected. Skip torch validation."
    return 0
  fi
  LOG "[INFO] Validation will run on ${total_models} torch model path(s):"
  printf "  - %s\n" "${MODIFIED_MODEL_PATHS[@]}" >&2
  fail_name=()
  for model_path in ${MODIFIED_MODEL_PATHS[@]}
  do
    python -m graph_net.torch.validate --model-path ${GRAPH_NET_EXTRACT_WORKSPACE}/${model_path} >&2
    [ $? -ne 0 ] && fail_name[${#fail_name[@]}]="${model_path}"
  done
  local failed_cnt=${#fail_name[@]}
  if [ ${failed_cnt} -ne 0 ]; then
    LOG "[FATAL] Failed tests (${failed_cnt}/${total_models}):"
    printf "  - %s\n" "${fail_name[@]}" >&2
    printf "%s\n" "${fail_name[@]}"
    exit -1
  else
    LOG "[INFO] All torch sample validations passed (0 failures / ${total_models} total)."
  fi
}

function check_paddle_validation() {
  LOG "[INFO] Start run validate for changed paddle samples ..."
  MODIFIED_MODEL_PATHS=()
  for file in $(git diff --diff-filter=ACM --name-only develop | grep -E "\bpaddle_samples\b/.*/(model\.py|input_meta\.py|weight_meta\.py)$")
  do
    if [[ ${file} != *shape_patches_* ]]; then
      LOG "[INFO] Found ${file} modified."
      model_path=$(dirname ${file})
      MODIFIED_MODEL_PATHS[${#MODIFIED_MODEL_PATHS[@]}]=$model_path
    fi
  done
  MODIFIED_MODEL_PATHS=($(echo ${MODIFIED_MODEL_PATHS[@]} | tr ' ' '\n' | sort | uniq))
  local total_models=${#MODIFIED_MODEL_PATHS[@]}
  if [ ${total_models} -eq 0 ]; then
    LOG "[INFO] No changed paddle sample models detected. Skip paddle validation."
    return 0
  fi
  LOG "[INFO] Validation will run on ${total_models} paddle model path(s):"
  printf "  - %s\n" "${MODIFIED_MODEL_PATHS[@]}" >&2
  fail_name=()
  for model_path in ${MODIFIED_MODEL_PATHS[@]}
  do
    python -m graph_net.paddle.validate --model-path ${GRAPH_NET_EXTRACT_WORKSPACE}/${model_path} >&2
    [ $? -ne 0 ] && fail_name[${#fail_name[@]}]="${model_path}"
  done
  local failed_cnt=${#fail_name[@]}
  if [ ${failed_cnt} -ne 0 ]; then
    LOG "[FATAL] Failed tests (${failed_cnt}/${total_models}):"
    printf "  - %s\n" "${fail_name[@]}" >&2
    printf "%s\n" "${fail_name[@]}"
    exit -1
  else
    LOG "[INFO] All paddle sample validations passed (0 failures / ${total_models} total)."
  fi
}

function summary_problems() {
  local check_validation_code=$1
  local check_validation_info=$2
  if [ $check_validation_code -ne 0 ]
  then
    LOG "[FATAL] ============================================"
    LOG "[FATAL] Summary problems:"
    local failed_list=($check_validation_info)
    local failed_count=${#failed_list[@]}
    LOG "[FATAL] === Sample test error (${failed_count} failure(s)) - Please fix the failed sample tests according to fatal log:"
    LOG "[FATAL] Failed model path(s):"
    printf "  - %s\n" "${failed_list[@]}" >&2
    exit $check_validation_code
  fi
}

function main() {
  check_paths_without_spaces
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
