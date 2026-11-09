#!/usr/bin/env bash
# kexec-rt.sh — 从 generic 内核热切换到 PREEMPT_RT 内核（绕开 GRUB 引导 RT 的已知缺陷）
#
# 背景：本机（ThinkPad T14, 20S0A001CD）GRUB 引导 6.8.1-1015-realtime 会反复
# 卡死在 "loading initial ramdisk"（4 次 3 卡，参数/磁盘/冷热启动均已排除），
# 而 kexec 热切换同一内核+initrd 稳定成功。故本机的标准进 RT 流程为：
#   上电（GRUB 默认进 generic，网络正常）→ 本脚本 → RT 内核（WiFi 会失效，属预期）
#
# 用法：sudo ./kexec-rt.sh          # 切换并重启进 RT
#       sudo ./kexec-rt.sh --dry-run  # 只打印将执行的命令，不实际切换
#
# 注意：
# 1. 切换是瞬时的，本 SSH 会话会立即断开（正常现象）；
# 2. RT 会话中 WiFi 不可用（kexec 跳过固件自检，网卡固件留脏状态）——
#    需要远程时请使用网线/USB 网卡；本地键盘可直接操作；
# 3. 回到 generic：直接 sudo reboot（GRUB 默认项 = generic，安全）。
set -euo pipefail

KERNEL=/boot/vmlinuz-6.8.1-1015-realtime
INITRD=/boot/initrd.img-6.8.1-1015-realtime
# root UUID 与 grub.cfg 一致（已验证可启动的参数组合）
CMDLINE="BOOT_IMAGE=${KERNEL} root=UUID=154dea20-3faf-4946-8db6-95f864532397 ro quiet splash"

[[ $EUID -eq 0 ]] || { echo "请用 sudo 运行"; exit 1; }
[[ -f $KERNEL && -f $INITRD ]] || { echo "错误：RT 内核/initrd 文件缺失"; exit 1; }

if [[ "${1:-}" == "--dry-run" ]]; then
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
