#!/bin/bash
# =============================================================================
# 模型抽取日志分析脚本
# =============================================================================
# 用途：分析 GraphNet Agent 批量模型抽取日志，输出成功/失败分布、异常类型、
#       模型大小分布、HTTP 状态码等统计信息，便于快速定位批量失败根因。
#
# 适用场景：
#   - 某批次跑完后，快速统计成功率和失败分布
#   - 对比多轮抽取的失败模式变化
#   - 生成已处理/成功模型列表，用于生成下一轮剩余列表
#
# 用法：
#   bash analyze_extraction_log.sh <log_file>
#
# 示例：
#   bash analyze_extraction_log.sh $HOME/workspace/logs_and_lists/batch7_safe_run.log
#
# 输出内容（10 个部分）：
#   一、总体统计 —— 总尝试数、成功数、失败数、成功率、PROGRESS 行数
#   二、失败原因一级分布 —— Script execution failed / Failed to analyze model /
#                           Failed to download / timeout / 401 / ducc 等
#   三、Failed to analyze model 细分 —— Model too large / Unsupported model_type 等
#   四、Model too large 分布（Top 10） —— 按参数量统计被过滤的大模型
#   五、Script execution failed 异常类型分布 —— ValueError / Dynamo / IndexError 等
#   六、ValueError 详细分布（Top 10） —— 最常见的 ValueError 具体信息
#   七、KeyError 详细分布（Top 10） —— 最常见的 KeyError 具体 key
#   八、Download failure HTTP 状态码分布 —— 404 / 403 等
#   九、其他异常 —— ducc 超时、ducc 空输出、脚本执行超时
#   十、辅助文件 —— 生成 /tmp/<log_name>_processed.txt 和 _success.txt
#
# 示例输出片段：
#   --- 一、总体统计 ---
#   总尝试数: 4973
#   成功数:   244
#   失败数:   1791
#   成功率:   4.9%
#   PROGRESS 行数: 2925
#
#   --- 二、失败原因一级分布 ---
#   类型 | 数量
#   ---|---
#   Script execution failed | 1003
#   Failed to analyze model | 718
#   timeout | 2
#   401 | 3
#   ducc | 66
#
#   --- 五、Script execution failed 异常类型分布 ---
#   异常类型 | 数量
#   ---|---
#   ValueError | 1368
#   torch._dynamo.exc.InternalTorchDynamoError | 940
#   IndexError | 936
#   ...
#
# 依赖：bash, grep, sed, awk, bc, sort, uniq
# =============================================================================

set -e

LOG_FILE="${1:-}"
if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
    echo "用法: bash $0 <log_file>"
    echo "示例: bash $0 \$HOME/workspace/logs_and_lists/batch7_safe_run.log"
    exit 1
fi

echo "======================================"
echo "批次日志分析报告"
echo "日志文件: $LOG_FILE"
echo "分析时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"
echo ""

# 1. 总体统计
echo "--- 一、总体统计 ---"
# 使用日志中实际的标记格式统计：
# 成功: "Graph extracted to:"（主要成功路径）
# 失败: "Extraction failed for"
# 注：部分旧日志可能用 "Successfully extracted"，但当前代码主要输出 "Graph extracted to"
# 部分日志含二进制数据，用 grep -a 强制按文本处理
TOTAL=$(grep -ac "Starting extraction for model:" "$LOG_FILE" 2>/dev/null || echo 0)
SUCCESS=$(grep -ac "Graph extracted to" "$LOG_FILE" 2>/dev/null || echo 0)
FAILED=$(grep -ac "Extraction failed for" "$LOG_FILE" 2>/dev/null || echo 0)
PROGRESS_LINES=$(grep -c "PROGRESS" "$LOG_FILE" 2>/dev/null || echo 0)

echo "总尝试数: $TOTAL"
echo "成功数:   $SUCCESS"
echo "失败数:   $FAILED"
if [ "$TOTAL" -gt 0 ]; then
    printf "成功率:   %.1f%%\n" $(echo "scale=2; $SUCCESS * 100 / $TOTAL" | bc 2>/dev/null || echo 0)
fi
echo "PROGRESS 行数: $PROGRESS_LINES"
echo ""

# 2. 失败原因一级分布
# 注意：不同错误类型在日志中的格式不同
# - Script execution failed / Failed to analyze model: "Extraction failed for ...: $reason"
# - Failed to download / 401: "Unexpected error for ...: $reason"
# - timeout: 可能出现在两种格式中
echo "--- 二、失败原因一级分布 ---"
echo "类型 | 数量"
echo "---|---"

# Script execution failed
scount=$(grep -c "Extraction failed for.*Script execution failed" "$LOG_FILE" 2>/dev/null || echo 0)
scount=$(echo "$scount" | tr -d '\n')
if [ "$scount" -gt 0 ]; then echo "Script execution failed | $scount"; fi

# Failed to analyze model (包含 Model too large / Unsupported model_type)
acount=$(grep -c "Extraction failed for.*Failed to analyze model" "$LOG_FILE" 2>/dev/null || echo 0)
acount=$(echo "$acount" | tr -d '\n')
if [ "$acount" -gt 0 ]; then echo "Failed to analyze model | $acount"; fi

# Download failure（包括超时、404、403、401 等）
dcount=$(grep -c "Unexpected error for.*Failed to download" "$LOG_FILE" 2>/dev/null || echo 0)
dcount=$(echo "$dcount" | tr -d '\n')
if [ "$dcount" -gt 0 ]; then echo "Failed to download | $dcount"; fi

# Sample verification failed
vcount=$(grep -c "Extraction failed for.*Sample verification failed" "$LOG_FILE" 2>/dev/null || echo 0)
vcount=$(echo "$vcount" | tr -d '\n')
if [ "$vcount" -gt 0 ]; then echo "Sample verification failed | $vcount"; fi

# 401（未授权 / 私有模型）
# 401 可能在 download 失败中，也可能单独出现
count401=$(grep -c "401 Client Error" "$LOG_FILE" 2>/dev/null || echo 0)
count401=$(echo "$count401" | tr -d '\n')
if [ "$count401" -gt 0 ]; then echo "401 Unauthorized | $count401"; fi

# ducc
ducccount=$(grep -c "ducc -p" "$LOG_FILE" 2>/dev/null || echo 0)
ducccount=$(echo "$ducccount" | tr -d '\n')
if [ "$ducccount" -gt 0 ]; then echo "ducc retry | $ducccount"; fi
echo ""

# 3. Failed to analyze model 细分
echo "--- 三、Failed to analyze model 细分 ---"
echo "子类别 | 数量"
echo "---|---"
for sub in "Model too large" "Unsupported model_type" "not supported between instances"; do
    count=$(grep "Extraction failed for.*Failed to analyze model" "$LOG_FILE" | grep -c "$sub" 2>/dev/null || echo 0)
    count=$(echo "$count" | tr -d '\n')
    if [ "$count" -gt 0 ]; then
        echo "$sub | $count"
    fi
done
echo ""

# 4. Model too large 的参数量分布（Top 10）
echo "--- 四、Model too large 分布（Top 10） ---"
grep "Extraction failed for.*Failed to analyze model.*Model too large" "$LOG_FILE" 2>/dev/null | \
    sed 's/.*estimated //;s/ parameters.*//' | \
    sort | uniq -c | sort -rn | head -10 | \
    awk '{print $2 " | " $1}'
echo ""

# 5. Script execution failed 的异常类型分布
echo "--- 五、Script execution failed 异常类型分布 ---"
TEMP_ERR=$(mktemp)
grep -A 30 "Extraction failed.*Script execution failed" "$LOG_FILE" 2>/dev/null | \
    grep -oE "(torch\._dynamo\.exc\.InternalTorchDynamoError|ValueError|AttributeError|TypeError|KeyError|ImportError|ModuleNotFoundError|RuntimeError|IndexError|NameError|OSError|FileNotFoundError)" | \
    sort | uniq -c | sort -rn > "$TEMP_ERR"
if [ -s "$TEMP_ERR" ]; then
    echo "异常类型 | 数量"
    echo "---|---"
    cat "$TEMP_ERR" | awk '{print $2 " | " $1}'
else
    echo "未提取到异常类型（可能日志中无详细 traceback）"
fi
rm -f "$TEMP_ERR"
echo ""

# 6. ValueError 详细分布
echo "--- 六、ValueError 详细分布（Top 10） ---"
grep -A 30 "Extraction failed.*Script execution failed" "$LOG_FILE" 2>/dev/null | \
    grep "ValueError" | sed 's/.*ValueError: //' | \
    sort | uniq -c | sort -rn | head -10 | \
    awk '{first=$1; $1=""; print substr($0,2) " | " first}'
echo ""

# 7. KeyError 详细分布
echo "--- 七、KeyError 详细分布（Top 10） ---"
grep -A 30 "Extraction failed.*Script execution failed" "$LOG_FILE" 2>/dev/null | \
    grep "KeyError" | sed "s/.*KeyError: //;s/'//g" | \
    sort | uniq -c | sort -rn | head -10 | \
    awk '{first=$1; $1=""; print substr($0,2) " | " first}'
echo ""

# 8. Download failure 的 HTTP 状态码分布
echo "--- 八、Download failure HTTP 状态码分布 ---"
grep "Unexpected error.*Failed to download" "$LOG_FILE" 2>/dev/null | \
    grep -oP '\d{3} Client Error' | sort | uniq -c | sort -rn | \
    awk '{print $2 " | " $1}'
echo ""

# 9. 超时/ducc 相关
echo "--- 九、其他异常 ---"
DUCC_TIMEOUT=$(grep -c "ducc -p timed out" "$LOG_FILE" 2>/dev/null || echo 0)
DUCC_TIMEOUT=$(echo "$DUCC_TIMEOUT" | tr -d '\n')
DUCC_EMPTY=$(grep -c "ducc -p returned empty" "$LOG_FILE" 2>/dev/null || echo 0)
DUCC_EMPTY=$(echo "$DUCC_EMPTY" | tr -d '\n')
SCRIPT_TIMEOUT=$(grep -c "Script execution timed out" "$LOG_FILE" 2>/dev/null || echo 0)
SCRIPT_TIMEOUT=$(echo "$SCRIPT_TIMEOUT" | tr -d '\n')
if [ "$DUCC_TIMEOUT" -gt 0 ]; then echo "ducc 超时: $DUCC_TIMEOUT"; fi
if [ "$DUCC_EMPTY" -gt 0 ]; then echo "ducc 空输出: $DUCC_EMPTY"; fi
if [ "$SCRIPT_TIMEOUT" -gt 0 ]; then echo "脚本执行超时: $SCRIPT_TIMEOUT"; fi
echo ""

# 10. 生成已处理模型列表（可选）
echo "--- 十、辅助文件 ---"
BASE_NAME=$(basename "$LOG_FILE" .log)
PROCESSED="/tmp/${BASE_NAME}_processed.txt"
SUCCESS_LIST="/tmp/${BASE_NAME}_success.txt"

grep "Starting extraction for model:" "$LOG_FILE" 2>/dev/null | \
    sed 's/.*Starting extraction for model: //' > "$PROCESSED"

grep -a "Graph extracted to" "$LOG_FILE" 2>/dev/null | \
    sed 's/.*Graph extracted to: .*\//\//' | \
    sed 's/.*samples\///' | sort -u > "$SUCCESS_LIST"

echo "已处理模型列表: $PROCESSED ($(wc -l < "$PROCESSED" 2>/dev/null || echo 0) 个)"
echo "成功模型列表:   $SUCCESS_LIST ($(wc -l < "$SUCCESS_LIST" 2>/dev/null || echo 0) 个)"
echo ""

echo "======================================"
echo "分析完成"
echo "======================================"
