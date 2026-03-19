#!/bin/bash
set -x

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DB_PATH="${1:-${GRAPH_NET_ROOT}/sqlite/GraphNet.db}"
DATASET_ROOT="${GRAPH_NET_ROOT}/20260317"
TORCH_MODEL_LIST="${DATASET_ROOT}/full_graph.txt"
TYPICAL_GRAPH_SAMPLES_LIST="${DATASET_ROOT}/typical_graph.txt"
FUSIBLE_GRAPH_SAMPLES_LIST="${DATASET_ROOT}/fusible_graph.txt"
SOLE_OP_GRAPH_SAMPLES_LIST="${DATASET_ROOT}/sole_op_graph.txt"
ORDER_VALUE=0

if [ ! -f "$DB_PATH" ]; then
    echo "Fail ! No Database ! : $DB_PATH"
    exit 1
fi

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "$DATASET_ROOT/full_graph" \
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
        --model_path_prefix "${DATASET_ROOT}/typical_graph" \
        --relative_model_path "$model_rel_path" \
        --op_names_path_prefix "${DATASET_ROOT}/03_sample_op_names" \
        --repo_uid "hf_torch_samples" \
        --sample_type "typical_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$TYPICAL_GRAPH_SAMPLES_LIST"

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "${DATASET_ROOT}/fusible_graph" \
        --relative_model_path "$model_rel_path" \
        --op_names_path_prefix "${DATASET_ROOT}/03_sample_op_names" \
        --repo_uid "hf_torch_samples" \
        --sample_type "fusible_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$FUSIBLE_GRAPH_SAMPLES_LIST"

while IFS= read -r model_rel_path; do
    echo "insert : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_insert.py" \
        --model_path_prefix "${DATASET_ROOT}/sole_op_graph" \
        --relative_model_path "$model_rel_path" \
        --op_names_path_prefix "${DATASET_ROOT}/03_sample_op_names" \
        --repo_uid "hf_torch_samples" \
        --sample_type "sole_op_graph" \
        --order_value "$ORDER_VALUE" \
        --db_path "$DB_PATH"

    ((ORDER_VALUE++))
    
done < "$SOLE_OP_GRAPH_SAMPLES_LIST"

echo "all done"
