#!/bin/bash
set -e

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

UNITTEST_PATH="$GRAPH_NET_ROOT/graph_net/torch/unittest"

python3 -m unittest discover \
    -s "$UNITTEST_PATH" \
    -p "test_*.py"
RET=$?

if [ $RET -eq 0 ]; then
    echo "✅ [SUCCESS] All unit tests passed!"
else
    echo "❌ [FAILURE] Unit tests failed with exit code $RET"
fi
exit $RET