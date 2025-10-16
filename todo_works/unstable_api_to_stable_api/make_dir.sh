#!/bin/bash
# 文件名：collect_unstable_paths.sh
# 功能：统计 samples/ 目录下所有包含各类 torch._C.* 调用的文件路径（去掉根目录和 model.py）

# 定义项目根目录
root_dir="/root/GraphNet"

# 定义不稳定 API 列表
unstable_apis=(
  "torch._C._nn.linear"
  "torch._C._nn.scaled_dot_product_attention"
  "torch._C._nn.gelu"
  "torch._C._nn.pad"
  "torch._C._nn.avg_pool2d"
  "torch._C._log_api_usage_once"
  "torch._C._functorch._add_batch_dim"
  "torch._C._functorch._remove_batch_dim"
  "torch._C._functorch._vmap_decrement_nesting"
  "torch._C._functorch._vmap_increment_nesting"
  "torch._C._linalg.linalg_norm"
  "torch._C._fft.fft_fftn"
  "torch._C._set_grad_enabled"
  "torch._C._nn.softplus"
  "torch._C._nn.one_hot"
  "torch._C._special.special_logit"
  "torch._C._fft.fft_rfft"
  "torch._C._linalg.linalg_vector_norm"
  "torch._C._fft.fft_irfft"
)

# 遍历每个 API
for api in "${unstable_apis[@]}"; do
  dir_name=$(echo "$api" | awk -F '.' '{print $NF}')
  mkdir -p "$dir_name"
  output_file="${dir_name}/${dir_name}_files.txt"

  echo "Searching for $api ..."

  # 查找并处理路径：去掉 root_dir 前缀和 /model.py 后缀
  find "$root_dir/samples/" -name "model.py" \
    | xargs grep "$api" 2>/dev/null \
    | tr ':' ' ' \
    | awk '{print $1}' \
    | sort | uniq \
    | sed "s|^${root_dir}/||" \
    | sed 's|/model.py$||' \
    > "$output_file"

  echo "  → Results saved to $output_file"
done

echo "✅ 所有不稳定 API 文件路径已收集完成（路径已去掉根目录与 model.py）！"
