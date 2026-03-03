#!/bin/bash
set -x

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")
DB_PATH="${1:-${GRAPH_NET_ROOT}/sqlite/GraphNet.db}"
DELETE_GRAPH_SAMPLES_LIST="graph_net/config/delete_list.txt"

if [ ! -f "$DB_PATH" ]; then
    echo "Fail ! No Database ! : $DB_PATH"
    exit 1
fi

while IFS= read -r model_rel_path; do
    echo "delete : $model_rel_path"
    python3 "${GRAPH_NET_ROOT}/sqlite/graphsample_delete.py" \
        --relative_model_path "$model_rel_path" \
        --repo_uid "github_torch_samples" \
        --db_path "$DB_PATH"
    
done < "$DELETE_GRAPH_SAMPLES_LIST"

echo "all done"
