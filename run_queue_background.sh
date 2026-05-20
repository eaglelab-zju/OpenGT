#!/usr/bin/env bash
# 在 OpenGT_eval 目录下后台跑队列；标准输出/错误追加到日志文件。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs
LOG="${ROOT}/logs/queue_worker_$(date +%Y%m%d_%H%M%S).log"
REGISTRY="${ROOT}/logs/queue_workers.registry"
# 将剩余参数原样传给 run_queue_experiment.py
nohup python3 "${ROOT}/run_queue_experiment.py" "$@" >>"$LOG" 2>&1 &
WPID=$!
# 仅保留「最后一次」启动的 PID，便于单进程场景下 ps/kill；多进程请用 registry
echo "$WPID" >"${ROOT}/logs/queue_worker.pid"
# 每次启动追加一行：时间、PID、日志路径、启动参数（制表符分隔）
printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds 2>/dev/null || date)" "$WPID" "$LOG" "$*" >>"$REGISTRY"
echo "已后台启动 PID=$WPID（已写入 logs/queue_worker.pid，仅记录最后一次启动）"
echo "多 worker 登记文件（追加，不会覆盖）: $REGISTRY"
echo "主进程日志（含子进程 print）: $LOG"
echo "若使用了 --task-log-dir，单任务 tune 详细日志在该目录下。"
echo ""
echo "=== 监控进度（另开终端，在 OpenGT_eval 下执行）==="
echo "  tail -f \"$LOG\""
echo "  tail -f \"${ROOT}/logs/queue_worker_\"*.log"
echo "  watch -n 30 \"cd \\\"$ROOT\\\" && grep -c ',done,' experiment_tasks.csv; tail -1 experiment_results.csv\""
echo "  ps -p \"$WPID\" -o pid,etime,cmd"
echo "  cut -f2 \"$REGISTRY\"                        # 本机登记过的所有 worker PID（第2列）"
echo "  nvidia-smi -l 5"
