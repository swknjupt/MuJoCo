"""确定性电机 I/O 抽象与 loopback backend（Task D / D2.2）。

本模块先固定真机驱动所需接口、超时语义和统计口径。真实 moteus/RS485 的帧格式
由官方库处理，待硬件到货后实现 `MoteusBackend`；当前 `LoopbackBackend` 可在
无硬件条件下验证上层调用、超时、丢包和延迟报告。

运行（仓库根目录）：
    .venv/bin/python demo/rt/motor_io.py --samples 10000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "examples" / "output" / "motor_io_loopback.json"


class IoStatus(str, Enum):
    """调用结果。"""

    OK = "ok"
    TIMEOUT = "timeout"
    CHECKSUM = "checksum_error"
    BUSY = "busy"
    IO_ERROR = "io_error"


class MotorIoError(RuntimeError):
    """后端传输错误基类。"""


class MotorTimeout(MotorIoError):
    """在调用截止时间前未收到合法响应。"""


class MotorChecksum(MotorIoError):
    """RS485 响应校验失败。"""


class MotorBusy(MotorIoError):
    """后端仍在处理上一请求。"""


@dataclass(frozen=True)
class MotorCommand:
    """一次力矩命令。"""

    timestamp_ns: int
    sequence: int
    torque_nm: float


@dataclass(frozen=True)
class MotorSample:
    """电机编码器与状态快照。"""

    timestamp_ns: int
    sequence: int
    position_rad: float
    velocity_rad_s: float
    torque_nm: float
    temperature_c: float
    fault_code: int
    roundtrip_ns: int


@dataclass(frozen=True)
class MotorIoResult:
    """上层调用结果。"""

    status: IoStatus
    sequence: int
    sample: MotorSample | None
    stale: bool
    error: str | None = None


class MotorBackend(Protocol):
    """真实或模拟 motor backend。"""

    def exchange(self, command: MotorCommand, deadline_ns: int) -> MotorSample:
        """发送命令并返回同拍状态；超时/校验失败必须抛出对应异常。"""


class LoopbackBackend:
    """确定性单摆负载 backend，用于无硬件接口验收。"""

    def __init__(
        self,
        *,
        latency_us: int = 50,
        jitter_us: int = 10,
        timeout_p: float = 0.002,
        checksum_p: float = 0.001,
        seed: int = 0,
    ) -> None:
        if latency_us <= 0 or jitter_us < 0:
            raise ValueError("latency_us 必须 > 0，jitter_us 必须 >= 0")
        if not 0.0 <= timeout_p <= 1.0 or not 0.0 <= checksum_p <= 1.0:
            raise ValueError("错误概率必须在 [0, 1]")
        self.latency_us = latency_us
        self.jitter_us = jitter_us
        self.timeout_p = timeout_p
        self.checksum_p = checksum_p
        self.seed = seed
        self._rng = random.Random(seed)
        self.position_rad = 0.0
        self.velocity_rad_s = 0.0
        self.last_torque_nm = 0.0

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self.position_rad = 0.0
        self.velocity_rad_s = 0.0
        self.last_torque_nm = 0.0

    def exchange(self, command: MotorCommand, deadline_ns: int) -> MotorSample:
        if command.timestamp_ns > deadline_ns:
            raise MotorTimeout("命令时间戳已超过截止时间")
        roll = self._rng.random()
        if roll < self.timeout_p:
            raise MotorTimeout("模拟响应超时")
        if roll < self.timeout_p + self.checksum_p:
            raise MotorChecksum("模拟 CRC 错误")

        jitter = self._rng.randint(-self.jitter_us, self.jitter_us)
        roundtrip_ns = max(1_000, (self.latency_us + jitter) * 1_000)
        if command.timestamp_ns + roundtrip_ns > deadline_ns:
            raise MotorTimeout("模拟往返时间超过 deadline")

        # 简单 1ms 负载积分只用于让 loopback 状态连续变化，不代表真机动力学。
        inertia = 0.2133
        damping = 0.05
        gravity_torque = 9.81 * 0.4 * math.sin(self.position_rad)
        acceleration = (
            command.torque_nm - damping * self.velocity_rad_s - gravity_torque
        ) / inertia
        self.velocity_rad_s += acceleration * 0.001
        self.position_rad += self.velocity_rad_s * 0.001
        self.last_torque_nm = command.torque_nm
        return MotorSample(
            timestamp_ns=command.timestamp_ns + roundtrip_ns,
            sequence=command.sequence,
            position_rad=self.position_rad,
            velocity_rad_s=self.velocity_rad_s,
            torque_nm=command.torque_nm,
            temperature_c=31.0,
            fault_code=0,
            roundtrip_ns=roundtrip_ns,
        )


class LatencyStats:
    """记录调用状态与往返延迟分布。"""

    def __init__(self) -> None:
        self.status_counts = {status.value: 0 for status in IoStatus}
        self.latencies_ns: list[int] = []

    def record(self, status: IoStatus, latency_ns: int) -> None:
        self.status_counts[status.value] += 1
        if status is IoStatus.OK:
            self.latencies_ns.append(int(latency_ns))

    def summary(self) -> dict[str, object]:
        values = sorted(self.latencies_ns)
        if not values:
            latency = {}
        else:
            latency = {
                "samples": len(values),
                "mean_us": sum(values) / len(values) / 1_000,
                "p50_us": _percentile(values, 0.50) / 1_000,
                "p99_us": _percentile(values, 0.99) / 1_000,
                "max_us": values[-1] / 1_000,
                "over_1ms": sum(value > 1_000_000 for value in values),
            }
        return {
            "status_counts": dict(self.status_counts),
            "latency": latency,
        }


def _percentile(sorted_values: list[int], fraction: float) -> int:
    index = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * fraction) - 1))
    return sorted_values[index]


class MotorIO:
    """带截止时间、最近有效样本和统计的电机 I/O。"""

    def __init__(
        self,
        backend: MotorBackend,
        *,
        timeout_us: int = 500,
        stale_after_us: int = 2_000,
    ) -> None:
        if timeout_us <= 0 or stale_after_us <= 0:
            raise ValueError("timeout/stale 必须 > 0")
        self.backend = backend
        self.timeout_us = timeout_us
        self.stale_after_ns = stale_after_us * 1_000
        self.stats = LatencyStats()
        self._sequence = 0
        self._last_sample: MotorSample | None = None

    @property
    def last_sample(self) -> MotorSample | None:
        return self._last_sample

    def command(
        self,
        torque_nm: float,
        *,
        now_ns: int | None = None,
        timeout_us: int | None = None,
    ) -> MotorIoResult:
        """发送力矩命令；任何错误都转成显式 IoStatus。"""
        timestamp_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        timeout = self.timeout_us if timeout_us is None else int(timeout_us)
        deadline_ns = timestamp_ns + timeout * 1_000
        self._sequence += 1
        command = MotorCommand(
            timestamp_ns=timestamp_ns,
            sequence=self._sequence,
            torque_nm=float(torque_nm),
        )
        wall_start_ns = time.perf_counter_ns()
        try:
            sample = self.backend.exchange(command, deadline_ns)
            wall_latency_ns = time.perf_counter_ns() - wall_start_ns
            latency_ns = sample.roundtrip_ns or wall_latency_ns
            status = IoStatus.OK
            error = None
            self._last_sample = sample
        except MotorTimeout as exc:
            sample, latency_ns, status, error = None, timeout * 1_000, IoStatus.TIMEOUT, str(exc)
        except MotorChecksum as exc:
            sample, latency_ns, status, error = None, timeout * 1_000, IoStatus.CHECKSUM, str(exc)
        except MotorBusy as exc:
            sample, latency_ns, status, error = None, timeout * 1_000, IoStatus.BUSY, str(exc)
        except Exception as exc:
            sample, latency_ns, status, error = None, timeout * 1_000, IoStatus.IO_ERROR, str(exc)

        self.stats.record(status, latency_ns)
        stale = (
            self._last_sample is None
            or timestamp_ns - self._last_sample.timestamp_ns > self.stale_after_ns
        )
        return MotorIoResult(
            status=status,
            sequence=command.sequence,
            sample=sample,
            stale=stale,
            error=error,
        )


def run_loopback_benchmark(
    samples: int,
    *,
    seed: int = 0,
    timeout_p: float = 0.002,
    checksum_p: float = 0.001,
) -> dict[str, object]:
    """运行固定 seed 的 loopback 功能验收。"""
    if samples <= 0:
        raise ValueError("samples 必须 > 0")
    backend = LoopbackBackend(timeout_p=timeout_p, checksum_p=checksum_p, seed=seed)
    io = MotorIO(backend)
    statuses: dict[str, int] = {}
    now_ns = 10_000_000
    for _ in range(samples):
        result = io.command(2.0, now_ns=now_ns)
        statuses[result.status.value] = statuses.get(result.status.value, 0) + 1
        now_ns += 1_000_000
    summary = io.stats.summary()
    summary.update(
        {
            "samples": samples,
            "seed": seed,
            "timeout_p": timeout_p,
            "checksum_p": checksum_p,
            "functional_evidence_only": True,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="电机 I/O loopback 功能验收")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-p", type=float, default=0.002)
    parser.add_argument("--checksum-p", type=float, default=0.001)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    result = run_loopback_benchmark(
        args.samples,
        seed=args.seed,
        timeout_p=args.timeout_p,
        checksum_p=args.checksum_p,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"报告：{args.report}")


if __name__ == "__main__":
    main()
