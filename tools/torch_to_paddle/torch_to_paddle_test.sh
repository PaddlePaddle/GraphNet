graph_net=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")
GraphNet="$graph_net/.."
output_dir="$GraphNet/torch_to_paddle_samples"
mkdir -p "$output_dir"
log_dir="$GraphNet/torch_to_paddle_samples"
result_file="$GraphNet/torch_to_paddle_samples/result.json"
touch "$result_file"

python3 -m tools.torch_to_paddle.convert \
--model-path-prefix "$GraphNet" \
--model-path-list "graph_net/config/small100_torch_samples_list.txt" \
--output-dir "$output_dir" \
--result-file "$result_file" 