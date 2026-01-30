#!/bin/bash
set -x

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DB_PATH="${GRAPH_NET_ROOT}/sqlite/GraphNet.db"
TORCH_MODEL_LIST="graph_net/config/test.txt"
PADDLE_MODEL_LIST="graph_net/config/small10_paddle_samples_list.txt"
TYPICAL_GRAPH_SAMPLES_LIST="tututu/range_decomposed_subgraph_sample_list.txt"
FUSIBLE_GRAPH_SAMPLES_LIST="tututu/fusible_subgraph_sample_list.txt"
SOLE_OP_GRAPH_SAMPLES_LIST="sole_graph/single_operator_sample_list.txt"
ORDER_VALUE=0


if [ ! -f "$DB_PATH" ]; then
    echo "Fail ! No Database ! : $DB_PATH"
    exit 1
fi

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "$GRAPH_NET_ROOT" \
        --relative_model_path "$model_rel_path" \
        --repo_uid "github_torch_samples" \
        --sample_type "full_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$TORCH_MODEL_LIST"

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "$GRAPH_NET_ROOT" \
        --relative_model_path "$model_rel_path" \
        --repo_uid "github_paddle_samples" \
        --sample_type "full_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$PADDLE_MODEL_LIST"

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "${GRAPH_NET_ROOT}/tututu/range_decompose" \
        --relative_model_path "$model_rel_path" \
        --repo_uid "github_torch_samples" \
        --sample_type "typical_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$TYPICAL_GRAPH_SAMPLES_LIST"

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "${GRAPH_NET_ROOT}/tututu/fusible_subgraph_samples" \
        --relative_model_path "$model_rel_path" \
        --repo_uid "github_torch_samples" \
        --sample_type "fusible_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$FUSIBLE_GRAPH_SAMPLES_LIST"

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "${GRAPH_NET_ROOT}/sole_graph" \
        --relative_model_path "$model_rel_path" \
        --repo_uid "github_torch_samples" \
        --sample_type "sole_op_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$SOLE_OP_GRAPH_SAMPLES_LIST"

echo "all done"
