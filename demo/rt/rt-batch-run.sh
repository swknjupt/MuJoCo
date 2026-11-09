#!/usr/bin/env bash
# rt-batch-run.sh — RT 会话批处理执行器（部署到笔记本 /usr/local/bin/）
#
# 机制：generic（有网络）时把任务脚本丢进队列目录 → kexec 进 RT →
# 本脚本由 systemd 在 RT 启动时自动执行队列 → 结果写盘 → 自动 reboot
# 回 generic（网络恢复）→ 远程取结果。
#
# 队列目录:  /var/local/rt-batch/queue/     （任务脚本，按文件名排序执行）
# 结果目录:  /var/local/rt-batch/results/   （<时间戳>_<任务名>.log）
# 队列为空时本脚本不做任何事（不影响手工 kexec 进 RT 的交互会话）。
set -u

QUEUE_DIR=/var/local/rt-batch/queue
RESULT_DIR=/var/local/rt-batch/results
JOB_TIMEOUT=7200   # 单任务最长 2 小时，防挂死

mkdir -p "$QUEUE_DIR" "$RESULT_DIR"

# 只在 RT 内核下执行（generic 会话中该服务虽启动但应跳过，防止误触发重启）
# 注意：/sys/kernel/realtime 是"文件"不是目录——曾用 -d 判断导致任务永远不执行（已修复）
if [[ ! -e /sys/kernel/realtime ]]; then
    exit 0
fi

shopt -s nullglob
JOBS=("$QUEUE_DIR"/*.sh)
if [[ ${#JOBS[@]} -eq 0 ]]; then
    exit 0    # 队列空：不动作、不重启（交互式 RT 会话不受影响）
fi

# 等系统基本就绪
sleep 5

for job in "${JOBS[@]}"; do
    name=$(basename "$job" .sh)
    out="$RESULT_DIR/$(date +%Y%m%d_%H%M%S)_${name}.log"
    {
        echo "==== rt-batch: $name ===="
        echo "kernel: $(uname -r)  boot: $(uptime -s)"
        echo "start : $(date -Is)"
    } > "$out"
    timeout "$JOB_TIMEOUT" bash "$job" >> "$out" 2>&1
    echo "end   : $(date -Is)  exit=$?" >> "$out"
    mv "$job" "$job.done"   # 任务出队（保留 .done 便于追溯）
done

sync
# 任务全部完成 → 回到 generic 恢复网络（GRUB 默认项 = generic，安全）
systemctl reboot
