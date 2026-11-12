"""D1.6 验收：正常推理、超时保持、计数与主循环非阻塞。

运行（仓库根目录）：
    .venv/bin/python -m demo.rt.test_inference
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from demo.rt.inference import OnnxCpuBackend, TimeboxedInferenceShell

ROOT = Path(__file__).resolve().parents[2]


class ControllableBackend:
    """在真实 ONNX 推理前按开关注入阻塞。"""

    def __init__(self) -> None:
        self.backend = OnnxCpuBackend()
        self.block_next = threading.Event()
        self.blocking_started = threading.Event()

    def infer(self, observation: np.ndarray) -> np.ndarray:
        if self.block_next.is_set():
            self.block_next.clear()
            self.blocking_started.set()
            time.sleep(0.050)
            self.blocking_started.clear()
        return self.backend.infer(observation)


def wait_until(predicate, timeout_s: float) -> bool:
    """轮询等待测试条件。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


def main() -> None:
    backend = ControllableBackend()
    shell = TimeboxedInferenceShell(backend)
    shell.start()
    try:
        observation = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
        shell.update(observation)
        assert wait_until(lambda: shell.get_latest().valid, 0.2), "正常推理未产生输出"
        baseline = shell.get_latest()
        assert baseline.age_s < 0.03, f"正常输出年龄异常：{baseline.age_s:.4f}s"
        print(
            f"[验收1] 正常推理输出有效 ✔ action={baseline.action[0]:.4f} "
            f"age={baseline.age_s * 1000:.2f}ms"
        )

        backend.block_next.set()
        assert backend.blocking_started.wait(timeout=0.2), "阻塞后端未被触发"
        baseline = shell.get_latest()
        shell.update(np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32))

        # 主循环调用 update 不得等待推理。集中调用 1000 次，避免阻塞后端
        # 在测量期间已经超时恢复，掩盖“旧值保持”的检查窗口。
        update_durations_us: list[float] = []
        for _ in range(1000):
            t0 = time.perf_counter_ns()
            shell.update(observation)
            update_durations_us.append((time.perf_counter_ns() - t0) / 1_000.0)
        max_update_us = max(update_durations_us)
        p99_update_us = float(np.percentile(update_durations_us, 99))
        assert max_update_us < 1_000.0, f"update 阻塞达 {max_update_us:.1f}us"
        print(
            f"[验收2] update 非阻塞 ✔ P99={p99_update_us:.2f}us "
            f"Max={max_update_us:.2f}us"
        )

        assert wait_until(lambda: shell.timeout_count >= 1, 0.2), "未检测到推理超时"
        during_timeout = shell.get_latest()
        assert during_timeout.timestamp_ns == baseline.timestamp_ns, (
            "超时结果不应覆盖最近有效值"
        )
        assert shell.timeout_streak >= 1
        print(
            f"[验收3] 超时保持旧值 ✔ timeout_count={shell.timeout_count} "
            f"streak={shell.timeout_streak}"
        )

        assert wait_until(
            lambda: shell.get_latest().timestamp_ns > baseline.timestamp_ns, 0.3
        ), "阻塞恢复后未发布新输出"
        assert wait_until(lambda: shell.timeout_streak == 0, 0.2), "成功输出未清零 streak"
        print("[验收4] 阻塞恢复 + 连续超时计数清零 ✔")
    finally:
        shell.stop()

    print("\n=== D1.6 推理壳全部验收通过 ===")


if __name__ == "__main__":
    main()
