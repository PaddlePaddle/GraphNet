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

    LOG "[INFO] Installing grpcio and protobuf..."
    python3.10 -m pip install -U grpcio "protobuf>=5.26.1" > /dev/null
    [ $? -ne 0 ] && LOG "[FATAL] Install grpcio or protobuf failed!" && exit -1
    LOG "[INFO] grpcio and protobuf installed successfully."
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

function run_shell_tests() {
    TEST_DIR="$GRAPH_NET_ROOT/graph_net/test"
    SKIP_LIST=("cumsum_num_kernels_test.sh" "prologue_subgraph_unittest_generator_test.sh")
    LOG "[INFO] Looking for shell scripts in: $TEST_DIR"

    mkdir -p .tmp_bin
    PY310_PATH=$(which python3.10)
    ln -sf "$PY310_PATH" .tmp_bin/python
    ln -sf "$PY310_PATH" .tmp_bin/python3
    export PATH="$(pwd)/.tmp_bin:$PATH"

    SCRIPTS=("$TEST_DIR"/*.sh)
    for script in "${SCRIPTS[@]}"; do
        script_name=$(basename "$script")
        if [[ " ${SKIP_LIST[*]} " =~ " $script_name " ]]; then
            continue
        fi

        LOG "[INFO] Running script: $script_name"        
        chmod +x "$script"
        "$script" || {
            EXIT_CODE=$?
            LOG "[ERROR] $script_name failed with exit code $EXIT_CODE"
            return $EXIT_CODE
        }
        LOG "[SUCCESS] $script_name finished successfully."
    done

    rm -rf .tmp_bin
    LOG "[INFO] All shell scripts passed successfully!"
    return 0
}

function main() {
    prepare_torch_env
    run_unit_test
    run_shell_tests
}

main