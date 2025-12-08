#!/bin/bash

GRAPH_NET_ROOT=$(python3 -c "import graph_net; import os; print(
os.path.dirname(graph_net.__file__))")

# input model path
MODEL_NAME=resnet18
MODEL_PATH_IN_SAMPLES=/timm/$MODEL_NAME
# INPUT_MODEL_LIST=$GRAPH_NET_ROOT/config/get_fusible_subgraph_sample_list.txt
INPUT_MODEL_LIST=$GRAPH_NET_ROOT/config/small_sample_list_for_get_fusible_subgraph.txt

OUTPUT_DIR="/tmp/find_fully_fusible_output"
config_json_str=$(cat <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/torch/fully_fusible_subgraph_extractor.py",
    "handler_class_name":"FullyFusibleSubgraphExtractor",
    "handler_config": {
        "model_path_prefix": "$GRAPH_NET_ROOT/../",
        "output_dir": "$OUTPUT_DIR",
        "split_positions": [],
        "group_head_and_tail": false,
        "chain_style": false,
        "max_step": 3,
        "min_step": 2,
        "max_nodes": 4
    }
}
EOF
)
CONFIG=$(echo $config_json_str | base64 -w 0)

# python3 -m graph_net.model_path_handler --model-path $GRAPH_NET_ROOT/../samples/$MODEL_PATH_IN_SAMPLES --handler-config=$CONFIG
python3 -m graph_net.model_path_handler --model-path-list $INPUT_MODEL_LIST --handler-config=$CONFIG


# 1. 读取总输入数量 (通过计算输入模型列表文件中的行数)
if [ ! -f "$INPUT_MODEL_LIST" ]; then
    echo "input dir $INPUT_MODEL_LIST not exist."
    TOTAL_INPUT_COUNT=0
else
    # wc -l 计算行数, awk '{print $1}' 移除文件名
    TOTAL_INPUT_COUNT=$(wc -l < "$INPUT_MODEL_LIST")
fi

# 2. 统计已产出文件夹数量
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "output dir $OUTPUT_DIR not exist."
    PRODUCED_COUNT=0
else
    # 查找 OUTPUT_DIR 下的一级子目录数量
    # -mindepth 1: 忽略父目录本身
    # -maxdepth 1: 只查找一级目录
    # -type d: 只查找目录 (文件夹)
    PRODUCED_COUNT=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
fi


echo "Total Input = $TOTAL_INPUT_COUNT"
echo "Total Output = $PRODUCED_COUNT"

# 3. 计算产出率 (使用 bc 进行浮点数运算)
if [ "$TOTAL_INPUT_COUNT" -gt 0 ]; then
    # scale=2 设置小数点后两位精度
    PRODUCTION_RATE=$(echo "scale=2; ($PRODUCED_COUNT * 100) / $TOTAL_INPUT_COUNT" | bc)
    echo "Production Rate = ${PRODUCTION_RATE}%"
fi
