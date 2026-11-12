#!/usr/bin/env bash
# kexec-rt.sh — 从 generic 内核热切换到 PREEMPT_RT 内核（绕开 GRUB 引导 RT 的已知缺陷）
#
# 背景：本机（ThinkPad T14, 20S0A001CD）GRUB 引导 6.8.1-1015-realtime 会反复
# 卡死在 "loading initial ramdisk"（4 次 3 卡，参数/磁盘/冷热启动均已排除），
# 而 kexec 热切换同一内核+initrd 稳定成功。故本机的标准进 RT 流程为：
#   上电（GRUB 默认进 generic，网络正常）→ 本脚本 → RT 内核（WiFi 会失效，属预期）
#
# 用法：sudo ./kexec-rt.sh                         # 交互式进入 RT
#       sudo ./kexec-rt.sh --dry-run               # 只打印交互式命令行
#       sudo ./kexec-rt.sh --batch                 # 无头启动 rt-batch.service
#       sudo ./kexec-rt.sh --batch --dry-run       # 只打印批处理命令行
#       sudo ./kexec-rt.sh --batch --isolate       # 无头批处理 + CPU2/3 隔离
#
# 注意：
# 1. 切换是瞬时的，本 SSH 会话会立即断开（正常现象）；
# 2. RT 会话中 WiFi 不可用（kexec 跳过固件自检，网卡固件留脏状态）——
#    需要远程时请使用网线/USB 网卡；本地键盘可直接操作；
# 3. 回到 generic：正常退出/重启优先由批处理脚本执行；人工恢复使用
#    poweroff + 重新上电，禁止热重启累积 CNVi/ME 状态。
set -euo pipefail

KERNEL=/boot/vmlinuz-6.8.1-1015-realtime
INITRD=/boot/initrd.img-6.8.1-1015-realtime
# root UUID 与 grub.cfg 一致（已验证可启动的参数组合）
CMDLINE="BOOT_IMAGE=${KERNEL} root=UUID=154dea20-3faf-4946-8db6-95f864532397 ro quiet splash"
BATCH_MODE=0
ISOLATE_MODE=0
DRY_RUN=0
QUEUE_DIR=/var/local/rt-batch/queue

for arg in "$@"; do
    case "$arg" in
        --batch) BATCH_MODE=1 ;;
        --isolate) ISOLATE_MODE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        *)
            echo "未知参数：$arg"
            exit 2
            ;;
    esac
done

[[ $EUID -eq 0 ]] || { echo "请用 sudo 运行"; exit 1; }
[[ -f $KERNEL && -f $INITRD ]] || { echo "错误：RT 内核/initrd 文件缺失"; exit 1; }

if [[ $BATCH_MODE -eq 1 ]]; then
    shopt -s nullglob
    jobs=("$QUEUE_DIR"/*.sh)
    [[ ${#jobs[@]} -gt 0 ]] || {
        echo "错误：批处理队列为空，拒绝进入无人值守 RT"
        exit 1
    }
    systemctl is-enabled --quiet rt-batch.service || {
        echo "错误：rt-batch.service 未启用"
        exit 1
    }
    [[ -x /usr/local/bin/rt-batch-run.sh ]] || {
        echo "错误：/usr/local/bin/rt-batch-run.sh 缺失或不可执行"
        exit 1
    }

    # RT 会话不加载桌面 GPU/音频驱动；上次这些 udev 探测触发 RCU stall。
    # systemd.unit 直接启动批处理服务，不等待 graphical/multi-user target。
    CMDLINE+=" nomodeset module_blacklist=nouveau,i915,snd_hda_intel,snd_sof_pci_intel_cnl systemd.unit=rt-batch.service"
    echo "批处理任务：${#jobs[@]} 个（${jobs[*]}）"
fi

if [[ $ISOLATE_MODE -eq 1 ]]; then
    # CPU2 给 1kHz 控制、CPU3 给推理；普通任务和 IRQ 留在 CPU0/1。
    CMDLINE+=" isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3 irqaffinity=0,1"
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] kexec -l $KERNEL --initrd=$INITRD --command-line=\"$CMDLINE\""
    echo "[dry-run] sync && kexec -e"
    exit 0
fi

echo "[1/3] 同步磁盘..."
sync
echo "[2/3] 装载 RT 内核到内存..."
kexec -l "$KERNEL" --initrd="$INITRD" --command-line="$CMDLINE"
echo "[3/3] 热切换（本会话即将断开，属预期）..."
sleep 1
exec kexec -e
