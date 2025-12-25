#!/bin/bash
set +ex

function LOG {
  echo "[$0:${BASH_LINENO[0]}] $*" >&2
}

export GRAPH_NET_ROOT=$(cd $(dirname "$0")/../.. && pwd)
export PYTHONPATH=${GRAPH_NET_ROOT}:$PYTHONPATH

function prepare_torch_env() {
    LOG "[INFO] Initializing torch environment for Unit Tests..."
    git config --global --add safe.directory "*"

    if ! python3.10 -c "import torch" &> /dev/null; then
        LOG "[INFO] Device Id: ${CUDA_VISIBLE_DEVICES}"
        LOG "[INFO] Update pip ..."
        env http_proxy="" https_proxy="" python3.10 -m pip install -U pip > /dev/null
        [ $? -ne 0 ] && LOG "[FATAL] Update pip failed!" && exit -1
        python3.10 -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 > /dev/null
        [ $? -ne 0 ] && LOG "[FATAL] Install torch2.9.0 failed!" && exit -1
    else
        LOG "[INFO] Torch environment is already ready."
    fi
}
function run_unit_test() {
    UNITTEST_PATH="$GRAPH_NET_ROOT/graph_net/torch/unittest"

    python3.10 -m unittest discover \
        -s "$UNITTEST_PATH" \
        -p "test_*.py"
    RET=$?

    if [ $RET -eq 0 ]; then
        echo "All unit tests passed!"
    else
        echo "Unit tests failed with exit code $RET"
    fi
    return $RET
}

function main() {
    prepare_torch_env
    run_unit_test
}

main