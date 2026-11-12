"""Timeboxed ONNX 推理壳与无锁 latest-value 输出（Task D / D1.6）。

部署约束：
  - 推理在独立线程运行，主实时循环只调用 update()/get_latest()；
  - 单次推理超过 timeout 后丢弃其结果，继续保留最近一次有效输出；
  - 连续超时由 timeout_streak 记录，下一条成功输出后清零；
  - 输出缓冲采用双缓冲 + seqlock 风格索引，读取路径不使用 Python Lock。

运行（仓库根目录）：
    .venv/bin/python -m demo.rt.test_inference
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "examples" / "output" / "policy.onnx"


class InferenceBackend(Protocol):
    """推理后端最小接口。"""

    def infer(self, observation: np.ndarray) -> np.ndarray:
        """输入单个 observation，返回单个 action。"""


class OnnxCpuBackend:
    """onnxruntime CPU 单样本推理后端。"""

    def __init__(self, model_path: Path = DEFAULT_ONNX) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = self._session.run(
            None, {self._input_name: observation[None, :].astype(np.float32, copy=False)}
        )[0]
        # 策略动作空间是目标角 [-π, π]；导出图保留原始 mean action，在运行边界裁剪。
        return np.clip(np.asarray(value, dtype=np.float32).reshape(-1), -math.pi, math.pi)


class LatestValueBuffer:
    """双缓冲 latest-value 缓冲；读取路径无锁，写者发布后原子切换索引。"""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._buffers = [np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)]
        self._active_index = -1
        self._sequence = 0
        self._timestamp_ns = 0

    @property
    def valid(self) -> bool:
        return self._active_index >= 0

    @property
    def timestamp_ns(self) -> int:
        return self._timestamp_ns

    def publish(self, value: np.ndarray, timestamp_ns: int) -> None:
        """写者：先写空闲槽，再递增序号切换 active index。"""
        next_index = 1 - self._active_index if self._active_index >= 0 else 0
        np.copyto(self._buffers[next_index], np.asarray(value, dtype=np.float32))
        self._sequence += 1
        self._timestamp_ns = int(timestamp_ns)
        self._active_index = next_index
        self._sequence += 1

    def read(self) -> tuple[np.ndarray, int] | None:
        """读者：seqlock 重试，避开正在发布的槽。"""
        for _ in range(8):
            sequence_before = self._sequence
            if sequence_before & 1:
                continue
            active_index = self._active_index
            if active_index < 0:
                return None
            value = self._buffers[active_index].copy()
            timestamp_ns = self._timestamp_ns
            if sequence_before == self._sequence and not (sequence_before & 1):
                return value, timestamp_ns
        # Python 的 GIL 使这个重试上限实际不可达；保留安全回退避免忙等。
        return None


@dataclass(frozen=True)
class LatestValue:
    """供主循环消费的最近有效策略输出。"""

    action: np.ndarray
    timestamp_ns: int
    age_s: float
    valid: bool

    def is_stale(self, stale_after_s: float) -> bool:
        return not self.valid or self.age_s > stale_after_s


@dataclass(frozen=True)
class _InferenceRequest:
    request_id: int
    observation: np.ndarray
    start_ns: int


@dataclass(frozen=True)
class _InferenceOutcome:
    request_id: int
    action: np.ndarray | None
    finish_ns: int
    error: str | None


class TimeboxedInferenceShell:
    """独立推理线程 + 10ms 调度线程 + 超时降级。"""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        observation_shape: tuple[int, ...] = (4,),
        period_s: float = 0.010,
        timeout_s: float = 0.005,
        stale_after_s: float = 0.030,
        worker_cpu: int | None = None,
    ) -> None:
        if period_s <= 0 or timeout_s <= 0 or stale_after_s <= 0:
            raise ValueError("period/timeout/stale_after 必须 > 0")
        if timeout_s >= stale_after_s:
            raise ValueError("timeout_s 必须小于 stale_after_s")

        self.backend = backend
        self.observation_shape = observation_shape
        self.period_s = float(period_s)
        self.timeout_s = float(timeout_s)
        self.stale_after_s = float(stale_after_s)
        self.worker_cpu = worker_cpu

        self._latest = LatestValueBuffer((1,))
        self._latest_observation: np.ndarray | None = None
        self._requests: queue.Queue[_InferenceRequest | None] = queue.Queue(maxsize=1)
        self._outcomes: queue.Queue[_InferenceOutcome] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, name="inference-worker", daemon=True
        )
        self._scheduler = threading.Thread(
            target=self._scheduler_loop, name="inference-scheduler", daemon=True
        )
        self._lock = threading.Lock()
        self._request_id = 0
        self._outstanding: _InferenceRequest | None = None
        self._timed_out = False
        self.timeout_count = 0
        self.timeout_streak = 0
        self.last_error: str | None = None

    def start(self) -> None:
        self._worker.start()
        self._scheduler.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=0.2)
        self._scheduler.join(timeout=0.2)

    def update(self, observation: np.ndarray) -> None:
        """主循环每拍调用；只保存引用拷贝，不执行推理。"""
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != self.observation_shape:
            raise ValueError(
                f"observation shape {value.shape} != {self.observation_shape}"
            )
        with self._lock:
            self._latest_observation = value.copy()

    def get_latest(self) -> LatestValue:
        """主循环读取最近有效 action；读取路径无锁。"""
        item = self._latest.read()
        now_ns = time.monotonic_ns()
        if item is None:
            return LatestValue(
                action=np.zeros(1, dtype=np.float32),
                timestamp_ns=0,
                age_s=math.inf,
                valid=False,
            )
        action, timestamp_ns = item
        return LatestValue(
            action=action,
            timestamp_ns=timestamp_ns,
            age_s=max(0.0, (now_ns - timestamp_ns) / 1_000_000_000),
            valid=True,
        )

    def _worker_loop(self) -> None:
        if self.worker_cpu is not None:
            os.sched_setaffinity(0, {self.worker_cpu})
        while not self._stop_event.is_set():
            try:
                request = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                break
            try:
                action = self.backend.infer(request.observation)
                outcome = _InferenceOutcome(
                    request_id=request.request_id,
                    action=np.asarray(action, dtype=np.float32).reshape(-1),
                    finish_ns=time.monotonic_ns(),
                    error=None,
                )
            except Exception as exc:  # 后端异常也必须让调度线程继续运行
                outcome = _InferenceOutcome(
                    request_id=request.request_id,
                    action=None,
                    finish_ns=time.monotonic_ns(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._outcomes.put(outcome)

    def _consume_outcomes(self, now_ns: int) -> None:
        while True:
            try:
                outcome = self._outcomes.get_nowait()
            except queue.Empty:
                return

            with self._lock:
                request = self._outstanding
                if request is None or outcome.request_id != request.request_id:
                    continue
                late = outcome.finish_ns - request.start_ns > self.timeout_s * 1e9
                if not late and not self._timed_out and outcome.error is None:
                    self._latest.publish(outcome.action, now_ns)
                    self.timeout_streak = 0
                    self.last_error = None
                else:
                    if not self._timed_out:
                        self.timeout_count += 1
                        self.timeout_streak += 1
                    if outcome.error is not None:
                        self.last_error = outcome.error
                self._outstanding = None
                self._timed_out = False

    def _scheduler_loop(self) -> None:
        if self.worker_cpu is not None:
            os.sched_setaffinity(0, {self.worker_cpu})
        next_deadline_ns = time.monotonic_ns()
        while not self._stop_event.is_set():
            now_ns = time.monotonic_ns()
            self._consume_outcomes(now_ns)

            with self._lock:
                request = self._outstanding
                if (
                    request is not None
                    and not self._timed_out
                    and now_ns - request.start_ns > self.timeout_s * 1e9
                ):
                    self._timed_out = True
                    self.timeout_count += 1
                    self.timeout_streak += 1

                if self._outstanding is None and self._latest_observation is not None:
                    self._request_id += 1
                    request = _InferenceRequest(
                        request_id=self._request_id,
                        observation=self._latest_observation.copy(),
                        start_ns=now_ns,
                    )
                    self._outstanding = request
                    self._timed_out = False
                    try:
                        self._requests.put_nowait(request)
                    except queue.Full:
                        self._outstanding = None

            next_deadline_ns += int(self.period_s * 1e9)
            remaining_s = (next_deadline_ns - time.monotonic_ns()) / 1e9
            if remaining_s > 0:
                self._stop_event.wait(remaining_s)
            else:
                next_deadline_ns = time.monotonic_ns()
