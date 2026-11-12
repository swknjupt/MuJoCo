"""D2.2 硬件无关部分：loopback 确定性、超时和延迟统计。

运行（仓库根目录）：
    .venv/bin/python -m demo.rt.test_motor_io
"""

from __future__ import annotations

from demo.rt.motor_io import (
    IoStatus,
    LoopbackBackend,
    MotorIO,
    run_loopback_benchmark,
)


def test_deterministic_loopback() -> None:
    first = run_loopback_benchmark(1_000, seed=42)
    second = run_loopback_benchmark(1_000, seed=42)
    assert first == second
    assert first["functional_evidence_only"] is True
    print("[验收1] 同种子 loopback 统计完全一致 ✔")


def test_timeout_and_checksum_paths() -> None:
    backend = LoopbackBackend(
        timeout_p=0.3,
        checksum_p=0.3,
        seed=7,
    )
    io = MotorIO(backend, timeout_us=500)
    statuses = set()
    for index in range(100):
        result = io.command(1.0, now_ns=1_000_000 + index * 1_000_000)
        statuses.add(result.status)
    assert IoStatus.OK in statuses
    assert IoStatus.TIMEOUT in statuses
    assert IoStatus.CHECKSUM in statuses
    summary = io.stats.summary()
    assert summary["status_counts"]["timeout"] > 0
    assert summary["status_counts"]["checksum_error"] > 0
    print("[验收2] timeout / checksum 错误路径可观测 ✔")


def test_latency_report() -> None:
    result = run_loopback_benchmark(
        10_000,
        seed=0,
        timeout_p=0.001,
        checksum_p=0.001,
    )
    latency = result["latency"]
    assert latency["samples"] > 9_000
    assert latency["p99_us"] < 500.0
    assert latency["over_1ms"] == 0
    print(
        f"[验收3] 10k loopback: P99={latency['p99_us']:.1f}us "
        f"Max={latency['max_us']:.1f}us ✔"
    )


def main() -> None:
    test_deterministic_loopback()
    test_timeout_and_checksum_paths()
    test_latency_report()
    print("\n=== D2.2 电机 I/O 软件侧验收通过（真机待硬件） ===")


if __name__ == "__main__":
    main()
