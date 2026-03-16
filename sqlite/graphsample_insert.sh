#!/bin/bash
set -x

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DB_PATH="${1:-${GRAPH_NET_ROOT}/sqlite/GraphNet.db}"
TORCH_MODEL_LIST="graph_net/config/torch_samples_list.txt"
PADDLE_MODEL_LIST="graph_net/config/small10_paddle_samples_list.txt"
TYPICAL_GRAPH_SAMPLES_LIST="subgraph_dataset_20260203/deduplicated_subgraph_sample_list.txt"
FUSIBLE_GRAPH_SAMPLES_LIST="subgraph_dataset_20260203/deduplicated_dimension_generalized_subgraph_sample_list.txt"
SOLE_OP_GRAPH_SAMPLES_LIST="subgraph_dataset_20260203/solo_sample_list.txt"
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
        --repo_uid "hf_torch_samples" \
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
        --repo_uid "hf_paddle_samples" \
        --sample_type "full_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$PADDLE_MODEL_LIST"

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "${GRAPH_NET_ROOT}/subgraph_dataset_20260203/typical_graph" \
        --relative_model_path "$model_rel_path" \
        --op_names_path_prefix "${GRAPH_NET_ROOT}/subgraph_dataset_20260203/03_sample_op_names" \
        --repo_uid "hf_torch_samples" \
        --sample_type "typical_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$TYPICAL_GRAPH_SAMPLES_LIST"

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "${GRAPH_NET_ROOT}/subgraph_dataset_20260203/fusible_graph" \
        --relative_model_path "$model_rel_path" \
        --op_names_path_prefix "${GRAPH_NET_ROOT}/subgraph_dataset_20260203/03_sample_op_names" \
        --repo_uid "hf_torch_samples" \
        --sample_type "fusible_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$FUSIBLE_GRAPH_SAMPLES_LIST"

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "${GRAPH_NET_ROOT}/subgraph_dataset_20260203/sole_op_graph" \
        --relative_model_path "$model_rel_path" \
        --op_names_path_prefix "${GRAPH_NET_ROOT}/subgraph_dataset_20260203/03_sample_op_names" \
        --repo_uid "hf_torch_samples" \
        --sample_type "sole_op_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$SOLE_OP_GRAPH_SAMPLES_LIST"

echo "all done"
