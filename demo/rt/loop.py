"""PREEMPT_RT 1kHz 周期循环骨架（Task D / D1.4）。

设计目标：
  - SCHED_FIFO 高优先级、单核绑定、mlockall，避免迁移与缺页；
  - CLOCK_MONOTONIC 绝对时基，周期不累计睡眠误差；
  - 每拍记录启动抖动；回调跨过下一拍起点时累计 overrun；
  - 纯标准库实现，可直接复制到 RT 盒运行。

标准验收（RT 盒、sudo）：
    sudo python3 demo/rt/loop.py --duration 1800

开发机功能冒烟（不具实时验收效力）：
    .venv/bin/python demo/rt/loop.py --duration 5 --allow-unprivileged --allow-non-rt

默认验收口径沿用 D2.1 修订：P99.99 < 50µs、Max < 5ms、overrun = 0，
且必须运行在 PREEMPT_RT 内核、SCHED_FIFO 和成功 mlockall 的条件下。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import socket
import sys
import time
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "examples" / "output" / "rt_loop_report.json"

MCL_CURRENT = 0x01
MCL_FUTURE = 0x02


@dataclass(frozen=True)
class LoopConfig:
    """周期循环参数。"""

    period_us: int = 1000
    duration_s: float = 30.0
    priority: int = 99
    cpu: int = 1
    sleep_margin_us: int = 200


@dataclass(frozen=True)
class LoopResult:
    """一次周期循环的验收结果。"""

    passed: bool
    ticks: int
    duration_s: float
    overrun_count: int
    jitter_mean_us: float
    jitter_p50_us: float
    jitter_p99_us: float
    jitter_p99_99_us: float
    jitter_max_us: float
    callback_max_us: float
    rt_kernel: bool
    scheduler: str
    actual_priority: int
    cpu_affinity: list[int]
    mlockall_ok: bool


def _percentile_rank(sorted_values: list[float], percentile: float) -> float:
    """最近秩百分位数，避免依赖 numpy。"""
    if not sorted_values:
        raise ValueError("空样本无法计算百分位")
    index = max(0, min(len(sorted_values) - 1, int(percentile * len(sorted_values)) - 1))
    return sorted_values[index]


def _is_rt_kernel() -> bool:
    """PREEMPT_RT 内核会暴露 /sys/kernel/realtime。"""
    return Path("/sys/kernel/realtime").exists()


def _scheduler_name(scheduler: int) -> str:
    """把 sched_getscheduler 返回值转换成可读名称。"""
    names = {
        os.SCHED_OTHER: "SCHED_OTHER",
        os.SCHED_FIFO: "SCHED_FIFO",
        os.SCHED_RR: "SCHED_RR",
    }
    return names.get(scheduler, f"SCHED_{scheduler}")


def configure_realtime(
    priority: int,
    cpu: int,
    *,
    allow_unprivileged: bool,
) -> tuple[bool, list[int]]:
    """配置调度策略、CPU 亲和性与内存锁定。

    返回 (mlockall_ok, 实际 CPU 亲和性)。调度失败时默认终止；仅在开发机
    显式传入 allow_unprivileged 时降级为 SCHED_OTHER。
    """
    # isolcpus 会把隔离核从默认 affinity 集合中移除，但显式绑定仍然合法。
    try:
        os.sched_setaffinity(0, {cpu})
    except OSError as exc:
        raise RuntimeError(
            f"无法绑定 CPU {cpu}；当前 affinity={sorted(os.sched_getaffinity(0))}"
        ) from exc

    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except PermissionError as exc:
        if not allow_unprivileged:
            raise PermissionError(
                "SCHED_FIFO 配置失败；RT 验收必须用 sudo 或授予 CAP_SYS_NICE"
            ) from exc
        print(f"警告：SCHED_FIFO 配置失败，开发机降级运行：{exc}", file=sys.stderr)

    libc = ctypes.CDLL(None, use_errno=True)
    mlockall_ok = libc.mlockall(MCL_CURRENT | MCL_FUTURE) == 0
    if not mlockall_ok:
        error_no = ctypes.get_errno()
        message = f"mlockall 失败：{os.strerror(error_no)}"
        if not allow_unprivileged:
            raise RuntimeError(message + "；请提高 RLIMIT_MEMLOCK 或用 root 运行")
        print(f"警告：{message}；开发机继续运行", file=sys.stderr)

    return mlockall_ok, sorted(os.sched_getaffinity(0))


def _wait_until(
    target_ns: int,
    sleep_margin_ns: int,
) -> int:
    """混合 sleep + 忙等到绝对目标时刻，返回实际唤醒时刻。"""
    while True:
        remaining_ns = target_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return time.monotonic_ns()
        if remaining_ns > sleep_margin_ns:
            time.sleep((remaining_ns - sleep_margin_ns) / 1_000_000_000)
        else:
            while time.monotonic_ns() < target_ns:
                pass
            return time.monotonic_ns()


def run_realtime_loop(
    config: LoopConfig,
    step_fn: Callable[[int, int], None] | None = None,
    *,
    allow_unprivileged: bool = False,
    allow_non_rt: bool = False,
) -> LoopResult:
    """运行周期循环并返回验收统计。

    step_fn 接收 (tick_index, deadline_ns)，只在两次取时之间执行，
    不应主动睡眠或分配大对象。
    """
    if config.period_us <= 0:
        raise ValueError("period_us 必须 > 0")
    if config.duration_s <= 0:
        raise ValueError("duration_s 必须 > 0")
    if not 0 <= config.sleep_margin_us < config.period_us:
        raise ValueError("sleep_margin_us 必须满足 0 <= margin < period")

    rt_kernel = _is_rt_kernel()
    if not rt_kernel and not allow_non_rt:
        raise RuntimeError(
            "当前不是 PREEMPT_RT 内核（缺少 /sys/kernel/realtime）；"
            "开发机请显式加 --allow-non-rt"
        )

    mlockall_ok, cpu_affinity = configure_realtime(
        config.priority, config.cpu, allow_unprivileged=allow_unprivileged
    )
    scheduler = _scheduler_name(os.sched_getscheduler(0))
    actual_priority = os.sched_getparam(0).sched_priority

    period_ns = config.period_us * 1_000
    sleep_margin_ns = config.sleep_margin_us * 1_000
    total_ticks = max(1, int(config.duration_s * 1_000_000 / config.period_us))
    jitter_ns = array("q", [0]) * total_ticks
    callback_ns = array("q", [0]) * total_ticks
    overrun_count = 0

    first_deadline_ns = time.monotonic_ns() + period_ns
    run_start_ns = time.monotonic_ns()
    deadline_ns = first_deadline_ns

    for tick in range(total_ticks):
        actual_start_ns = _wait_until(deadline_ns, sleep_margin_ns)
        jitter_ns[tick] = actual_start_ns - deadline_ns

        if step_fn is not None:
            step_fn(tick, deadline_ns)

        finish_ns = time.monotonic_ns()
        callback_ns[tick] = max(0, finish_ns - actual_start_ns)
        next_deadline_ns = deadline_ns + period_ns
        if finish_ns > next_deadline_ns:
            overrun_count += 1
        deadline_ns = next_deadline_ns

    run_elapsed_s = (time.monotonic_ns() - run_start_ns) / 1_000_000_000
    jitter_us = [value / 1_000.0 for value in jitter_ns]
    callback_us = [value / 1_000.0 for value in callback_ns]
    sorted_jitter = sorted(jitter_us)
    p99_99_us = _percentile_rank(sorted_jitter, 0.9999)
    max_us = max(sorted_jitter)
    passed = (
        rt_kernel
        and scheduler == "SCHED_FIFO"
        and mlockall_ok
        and overrun_count == 0
        and p99_99_us < 50.0
        and max_us < 5_000.0
    )

    return LoopResult(
        passed=passed,
        ticks=total_ticks,
        duration_s=run_elapsed_s,
        overrun_count=overrun_count,
        jitter_mean_us=sum(jitter_us) / len(jitter_us),
        jitter_p50_us=_percentile_rank(sorted_jitter, 0.50),
        jitter_p99_us=_percentile_rank(sorted_jitter, 0.99),
        jitter_p99_99_us=p99_99_us,
        jitter_max_us=max_us,
        callback_max_us=max(callback_us),
        rt_kernel=rt_kernel,
        scheduler=scheduler,
        actual_priority=actual_priority,
        cpu_affinity=cpu_affinity,
        mlockall_ok=mlockall_ok,
    )


def save_report(report_path: Path, config: LoopConfig, result: LoopResult) -> None:
    """保存机器可读的循环验收报告。"""
    report = {
        "config": asdict(config),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "python": platform.python_version(),
        },
        "result": asdict(result),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PREEMPT_RT 1kHz 周期循环骨架")
    parser.add_argument("--duration", type=float, default=30.0, help="运行秒数")
    parser.add_argument("--period-us", type=int, default=1000)
    parser.add_argument("--priority", type=int, default=99)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--sleep-margin-us", type=int, default=200)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--allow-unprivileged", action="store_true")
    parser.add_argument("--allow-non-rt", action="store_true")
    args = parser.parse_args()

    config = LoopConfig(
        period_us=args.period_us,
        duration_s=args.duration,
        priority=args.priority,
        cpu=args.cpu,
        sleep_margin_us=args.sleep_margin_us,
    )
    result = run_realtime_loop(
        config,
        allow_unprivileged=args.allow_unprivileged,
        allow_non_rt=args.allow_non_rt,
    )
    save_report(args.report, config, result)

    print(
        f"ticks={result.ticks} duration={result.duration_s:.3f}s "
        f"scheduler={result.scheduler}/{result.actual_priority} "
        f"cpu={result.cpu_affinity} rt_kernel={result.rt_kernel} "
        f"mlockall={result.mlockall_ok}"
    )
    print(
        f"jitter mean={result.jitter_mean_us:.3f}us P50={result.jitter_p50_us:.3f}us "
        f"P99={result.jitter_p99_us:.3f}us P99.99={result.jitter_p99_99_us:.3f}us "
        f"Max={result.jitter_max_us:.3f}us"
    )
    print(
        f"callback_max={result.callback_max_us:.3f}us "
        f"overrun={result.overrun_count}"
    )
    print(f"报告：{args.report}")
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
