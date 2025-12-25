#!/bin/bash
set -e
export GRAPH_NET_ROOT=$(cd $(dirname "$0")/../.. && pwd)
export PYTHONPATH=${GRAPH_NET_ROOT}:$PYTHONPATH

UNITTEST_PATH="$GRAPH_NET_ROOT/graph_net/torch/unittest"

python3.10 -m unittest discover \
    -s "$UNITTEST_PATH" \
    -p "test_*.py"
RET=$?

if [ $RET -eq 0 ]; then
    echo "✅ [SUCCESS] All unit tests passed!"
else
    echo "❌ [FAILURE] Unit tests failed with exit code $RET"
fi
exit $RET