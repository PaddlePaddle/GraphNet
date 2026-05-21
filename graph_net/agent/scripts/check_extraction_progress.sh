#!/bin/bash
# =============================================================================
# 模型抽取进度检查脚本
# =============================================================================
# 用途：一键查看当前模型抽取任务的运行状态，包括进程存活、日志进度、
#       成功/失败统计、处理速度估算、磁盘空间等。
#
# 用法：
#   bash check_extraction_progress.sh [log_file]
#
#   不带参数：自动查找 $HOME/workspace/logs_and_lists/ 下最新的 .log
#   带参数：  指定日志文件路径
#
# 示例：
#   bash check_extraction_progress.sh
#   bash check_extraction_progress.sh $HOME/workspace/logs_and_lists/batch7_safe_run.log
#
# 输出示例：
#   ====== 进程状态 ======
#   主进程 PID: 24925 (运行时长: 00:24)
#   CPU 使用率: 132%
#   内存使用率: 1.0%
#   Worker 数量: 7
#
#   ====== 日志进度 ======
#   日志文件: batch7_safe_run_no_llm.log
#   最新进度: [PROGRESS] 1608/63555 done, success rate so far: 24.5%
#   日志时间: 2026-05-18 16:57:48
#   最后更新: 0 秒前
#
#   ====== 统计汇总 ======
#   已处理: 1608 / 63555
#   剩余:   61947
#   成功率: 24.5%
#   成功:   37
#   失败:   1195
#   速度:   ~1800 个/小时 (最近 50 条 PROGRESS)
#   预计剩余: ~34.4 小时 (~1.4 天)
#
#   ====== 磁盘空间 ======
#   /home: 可用 67G / 总 3.6T (99% 已用)
#   /tmp:  可用 5.8G / 总 6.5G (6% 已用)
#
#   ====== 样本目录 ======
#   成功样本: 2025 个 (目录: $HOME/workspace/success)
#   样本文件: 3382 个 (目录: $HOME/workspace/samples)
#
# =============================================================================

# 默认路径，可通过环境变量覆盖
LOG_DIR="${GRAPHNET_LOG_DIR:-$HOME/workspace/logs_and_lists}"
SUCCESS_DIR="${GRAPHNET_SUCCESS_DIR:-$HOME/workspace/success}"
SAMPLES_DIR="${GRAPHNET_SAMPLES_DIR:-$HOME/workspace/samples}"

# ---------------------------------------------------------------------------
# 1. 确定日志文件
# ---------------------------------------------------------------------------
if [ -n "$1" ]; then
    LOG_FILE="$1"
else
    LOG_FILE=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
fi

if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
    echo "错误: 未找到日志文件"
    echo "用法: bash $0 [log_file]"
    exit 1
fi

LOG_BASENAME=$(basename "$LOG_FILE")

# ---------------------------------------------------------------------------
# 2. 进程状态
# ---------------------------------------------------------------------------
echo "====== 进程状态 ======"

# 查找 graph_net_agent / run_model 相关进程
PIDS=$(pgrep -f "graph_net_agent|run_model\.py|parallel_extract" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    MAIN_PID=$(echo "$PIDS" | head -1)
    # 计算运行时长
    ELAPSED=$(ps -o etime= -p "$MAIN_PID" 2>/dev/null | tr -d ' ' || echo "未知")
    # CPU 和内存
    CPU_MEM=$(ps -o %cpu=,%mem= -p "$MAIN_PID" 2>/dev/null | tr -s ' ' | sed 's/^ //' || echo "? ?")
    CPU=$(echo "$CPU_MEM" | awk '{print $1}')
    MEM=$(echo "$CPU_MEM" | awk '{print $2}')
    # Worker 数量（run_model.py 子进程数）
    WORKERS=$(pgrep -f "run_model\.py" 2>/dev/null | wc -l)

    echo "主进程 PID: $MAIN_PID (运行时长: $ELAPSED)"
    echo "CPU 使用率: ${CPU}%"
    echo "内存使用率: ${MEM}%"
    echo "Worker 数量: $WORKERS"
else
    echo "状态: 无运行中的抽取进程"
fi
echo ""

# ---------------------------------------------------------------------------
# 3. 日志进度
# ---------------------------------------------------------------------------
echo "====== 日志进度 ======"
echo "日志文件: $LOG_BASENAME"

LATEST_PROGRESS=$(grep "PROGRESS" "$LOG_FILE" 2>/dev/null | tail -1)
if [ -n "$LATEST_PROGRESS" ]; then
    echo "最新进度: $LATEST_PROGRESS"
else
    echo "最新进度: (日志中无 PROGRESS 记录)"
fi

# 最后一条日志的时间戳（从末尾 20 行里找）
LAST_LOG_TIME=$(tail -20 "$LOG_FILE" 2>/dev/null | grep -oP '^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}' | tail -1)
if [ -n "$LAST_LOG_TIME" ]; then
    LAST_TS=$(date -d "$LAST_LOG_TIME" +%s 2>/dev/null || echo 0)
    NOW_TS=$(date +%s)
    DELTA=$((NOW_TS - LAST_TS))
    if [ $DELTA -lt 60 ]; then
        AGO="${DELTA} 秒前"
    elif [ $DELTA -lt 3600 ]; then
        AGO="$((DELTA / 60)) 分钟前"
    elif [ $DELTA -lt 86400 ]; then
        AGO="$((DELTA / 3600)) 小时前"
    else
        AGO="$((DELTA / 86400)) 天前"
    fi
    echo "日志时间: $LAST_LOG_TIME"
    echo "最后更新: $AGO"
else
    echo "日志时间: (无法解析)"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. 统计汇总
# ---------------------------------------------------------------------------
echo "====== 统计汇总 ======"

# 从日志统计
TOTAL_IN_LOG=$(grep -cE "Starting extraction for|Successfully extracted|Extraction failed for" "$LOG_FILE" 2>/dev/null || echo 0)
SUCCESS_IN_LOG=$(grep -c "Successfully extracted" "$LOG_FILE" 2>/dev/null || echo 0)
FAILED_IN_LOG=$(grep -c "Extraction failed for" "$LOG_FILE" 2>/dev/null || echo 0)

# 从 PROGRESS 行提取进度
if [ -n "$LATEST_PROGRESS" ]; then
    DONE=$(echo "$LATEST_PROGRESS" | grep -oP '\d+(?=/)' | head -1)
    TOTAL=$(echo "$LATEST_PROGRESS" | grep -oP '(?<=/)\d+' | head -1)
    RATE=$(echo "$LATEST_PROGRESS" | grep -oP '\d+\.\d+(?=%)' | head -1)

    if [ -n "$DONE" ] && [ -n "$TOTAL" ]; then
        echo "已处理: $DONE / $TOTAL"
        REMAIN=$((TOTAL - DONE))
        echo "剩余:   $REMAIN"
    fi
    if [ -n "$RATE" ]; then
        echo "成功率: ${RATE}%"
    fi
fi

echo "成功:   $SUCCESS_IN_LOG"
echo "失败:   $FAILED_IN_LOG"

# 速度估算：取最近 50 条 PROGRESS，计算时间差
PROGRESS_TIMES=$(grep "PROGRESS" "$LOG_FILE" 2>/dev/null | tail -50 | grep -oP '^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}')
PROGRESS_COUNT=$(echo "$PROGRESS_TIMES" | wc -l)
if [ "$PROGRESS_COUNT" -ge 2 ]; then
    FIRST_TIME=$(echo "$PROGRESS_TIMES" | head -1)
    LAST_TIME=$(echo "$PROGRESS_TIMES" | tail -1)
    FIRST_TS=$(date -d "$FIRST_TIME" +%s 2>/dev/null || echo 0)
    LAST_TS=$(date -d "$LAST_TIME" +%s 2>/dev/null || echo 0)
    TIME_DIFF=$((LAST_TS - FIRST_TS))
    if [ "$TIME_DIFF" -gt 0 ]; then
        SPEED_PER_HOUR=$(echo "scale=0; $PROGRESS_COUNT * 3600 / $TIME_DIFF" | bc 2>/dev/null || echo 0)
        echo "速度:   ~$SPEED_PER_HOUR 个/小时 (最近 $PROGRESS_COUNT 条 PROGRESS)"
        if [ -n "$REMAIN" ] && [ "$SPEED_PER_HOUR" -gt 0 ] 2>/dev/null; then
            HOURS_LEFT=$(echo "scale=1; $REMAIN / $SPEED_PER_HOUR" | bc 2>/dev/null || echo 0)
            DAYS_LEFT=$(echo "scale=1; $HOURS_LEFT / 24" | bc 2>/dev/null || echo 0)
            echo "预计剩余: ~${HOURS_LEFT} 小时 (~${DAYS_LEFT} 天)"
        fi
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 5. 磁盘空间
# ---------------------------------------------------------------------------
echo "====== 磁盘空间 ======"
df -h /home /tmp 2>/dev/null | awk 'NR==1 {next} {printf "%s: 可用 %s / 总 %s (%s 已用)\n", $6, $4, $2, $5}'
echo ""

# ---------------------------------------------------------------------------
# 6. 样本目录统计（可选）
# ---------------------------------------------------------------------------
echo "====== 样本目录 ======"
if [ -d "$SUCCESS_DIR" ]; then
    SUCCESS_COUNT=$(ls -1 "$SUCCESS_DIR" 2>/dev/null | wc -l)
    echo "成功样本: $SUCCESS_COUNT 个 (目录: $SUCCESS_DIR)"
fi
if [ -d "$SAMPLES_DIR" ]; then
    SAMPLE_COUNT=$(ls -1 "$SAMPLES_DIR" 2>/dev/null | wc -l)
    echo "样本文件: $SAMPLE_COUNT 个 (目录: $SAMPLES_DIR)"
fi
echo ""

echo "====== 检查完成 ======"
