#!/bin/bash
# 批量运行 GraphNet benchmark for unstable_to_stable (_add_batch_dim)
# 从文件列表中读取模型路径执行编译测试，并将 log 转换为 JSON
if [ -z "$DISALLOWED_UNSTABLE_API" ]; then
  echo "❌ 环境变量 DISALLOWED_UNSTABLE_API 未设置！"
  echo "请使用： export DISALLOWED_UNSTABLE_API=<target_unstable_api>"
  exit 1
fi

# === 配置区 ===
root_dir="todo_works/unstable_api_to_stable_api/${DISALLOWED_UNSTABLE_API}"
file_list="${root_dir}/${DISALLOWED_UNSTABLE_API}_files.txt"
log_file="${root_dir}/log.log"
json_output_dir="${root_dir}/JSON_results"

# 设置环境变量（benchmark 路径）
export GRAPH_NET_BENCHMARK_PATH="$root_dir"

# === 检查输入文件 ===
if [ ! -f "$file_list" ]; then
  echo "❌ 文件不存在: $file_list"
  exit 1
fi

# === 执行 benchmark ===
echo "🚀 开始执行 benchmark..."
echo "日志将写入: $log_file"
# echo "--------------------------------------" > "$log_file"

if [ -f "$log_file" ]; then
  echo "🧹 删除旧的日志文件: $log_file"
  rm "$log_file"
fi

while IFS= read -r model_path; do
  [ -z "$model_path" ] && continue

  echo "▶️ 运行模型: $model_path"
  echo ">>> Running model: $model_path" 

  python -m graph_net.torch.test_compiler \
    --model-path "${model_path}/" \
    --compiler unstable_to_stable \
    >> "$log_file" 2>&1

  echo "✅ 完成: $model_path" 
  echo "--------------------------------------" 
done < "$file_list"

echo "🎯 所有模型运行完成，日志保存在: $log_file"

# === 转换 log 为 JSON ===
echo "📦 正在将日志转换为 JSON..."
if [ -d "$json_output_dir" ]; then
  echo "🧹 删除旧的 JSON 输出目录: $json_output_dir"
  rm -rf "$json_output_dir"
fi
mkdir -p "$json_output_dir"

python -m graph_net.log2json \
  --log-file "$log_file" \
  --output-dir "$json_output_dir"

if [ $? -eq 0 ]; then
  echo "✅ JSON 文件已生成: $json_output_dir"
else
  echo "⚠️ log2json 执行失败，请检查 log.log"
fi

echo "📦 正在将JSON转换为结果图"
python -m graph_net.plot_ESt \
  --benchmark-path $GRAPH_NET_BENCHMARK_PATH/JSON_results/ \
  --output-dir $GRAPH_NET_BENCHMARK_PATH \

if [ $? -eq 0 ]; then
  echo "✅ 结果图 文件已生成: $GRAPH_NET_BENCHMARK_PATH"
else
  echo "❌结果图生成失败"
fi
