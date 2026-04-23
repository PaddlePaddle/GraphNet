#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH_PREFIX="${1:?Usage: $0 <model_path_prefix> [monitor|status|restart] [num_gpus] [log_dir]}"
MODE="${2:-monitor}"
NUM_GPUS="${3:-8}"
LOG_DIR="${4:-/tmp/init-constraints-logs}"

mkdir -p "$LOG_DIR"

# Check if a task for a specific GPU is running
check_task() {
    local gpu_id=$1
    local suffix=$(printf "%02d" $gpu_id)
    local pid=$(pgrep -f "model_path_handler.*txt$suffix" | head -1)

    if [ -n "$pid" ]; then
        if kill -0 "$pid" 2>/dev/null; then
            echo "running:$pid"
            return
        fi
    fi
    echo "stopped"
}

# Start a task on a specific GPU
start_task() {
    local gpu_id=$1
    local suffix=$(printf "%02d" $gpu_id)
    local log_file="$LOG_DIR/gpu_${gpu_id}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting GPU $gpu_id task..."
    bash "$SCRIPT_DIR/init_input_tensor_constraints.sh" "$MODEL_PATH_PREFIX" "$gpu_id" "$suffix" \
        > "$log_file" 2>&1 &
    local new_pid=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id started, PID: $new_pid, log: $log_file"
}

# Check if task failed due to OOM
check_oom() {
    local gpu_id=$1
    local log_file="$LOG_DIR/gpu_${gpu_id}.log"

    if [ -f "$log_file" ]; then
        if grep -q "out of memory\|CUDA out of memory\|OOM" "$log_file" 2>/dev/null; then
            echo "oom"
            return
        fi
    fi
    echo "normal"
}

# Show status of all tasks
show_status() {
    echo ""
    echo "========== Task Status [$(date '+%Y-%m-%d %H:%M:%S')] =========="
    echo ""

    local running_count=0
    local stopped_count=0
    local oom_count=0

    for ((gpu_id = 0; gpu_id < NUM_GPUS; gpu_id++)); do
        local status=$(check_task $gpu_id)

        if [ "$status" = "stopped" ]; then
            local oom_status=$(check_oom $gpu_id)
            if [ "$oom_status" = "oom" ]; then
                if [ $gpu_id -eq 0 ]; then
                    echo "GPU $gpu_id: OOM (expected on GPU 0, ignored)"
                    ((oom_count++))
                else
                    echo "GPU $gpu_id: STOPPED (OOM) - needs restart"
                    ((stopped_count++))
                fi
            else
                echo "GPU $gpu_id: STOPPED - needs restart"
                ((stopped_count++))
            fi
        else
            local pid=${status#running:}
            echo "GPU $gpu_id: RUNNING (PID: $pid)"
            ((running_count++))
        fi
    done

    echo ""
    echo "Summary: $running_count running, $stopped_count stopped, $oom_count OOM(GPU0 ignored)"
    echo "======================================================"
}

# Monitor and restart stopped tasks
monitor_and_restart() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring tasks... (Ctrl+C to stop)"
    echo ""

    local iteration=0

    while true; do
        ((iteration++))

        for ((gpu_id = 0; gpu_id < NUM_GPUS; gpu_id++)); do
            local status=$(check_task $gpu_id)

            if [ "$status" = "stopped" ]; then
                if [ $gpu_id -eq 0 ]; then
                    local oom_status=$(check_oom $gpu_id)
                    if [ "$oom_status" = "oom" ]; then
                        continue
                    fi
                fi

                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id stopped, restarting..."
                start_task $gpu_id
                sleep 2
            fi
        done

        if [ $((iteration % 5)) -eq 0 ]; then
            show_status
        fi

        sleep 3
    done
}

# One-shot restart all stopped tasks
restart_all_stopped() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting all stopped tasks..."

    for ((gpu_id = 0; gpu_id < NUM_GPUS; gpu_id++)); do
        local status=$(check_task $gpu_id)

        if [ "$status" = "stopped" ]; then
            if [ $gpu_id -eq 0 ]; then
                local oom_status=$(check_oom $gpu_id)
                if [ "$oom_status" = "oom" ]; then
                    echo "GPU $gpu_id: OOM (expected, skipping)"
                    continue
                fi
            fi

            start_task $gpu_id
        else
            echo "GPU $gpu_id: already running, skipping"
        fi
    done
}

case "$MODE" in
    "monitor")
        monitor_and_restart
        ;;
    "status")
        show_status
        ;;
    "restart")
        restart_all_stopped
        ;;
    *)
        echo "Usage: $0 <model_path_prefix> [monitor|status|restart] [num_gpus] [log_dir]"
        echo "  monitor - Continuously monitor and auto-restart stopped tasks (default)"
        echo "  status  - Show current task status"
        echo "  restart - One-shot restart all stopped tasks"
        exit 1
        ;;
esac
