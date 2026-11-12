"""信号层故障注入框架（Task D / D1.7）。

注入位于传感器读数与控制输出之间，三类故障可独立配置：
  1. 观测延迟：回放 k ms 前的采样，零阶保持；
  2. 编码器丢帧：按种子控制的概率保持上一帧；
  3. 力矩衰减：最终力矩乘以 0~1 的系数。

所有随机决策来自单一 random.Random(seed)，同配置与种子逐拍可复现。

运行（仓库根目录）：
    .venv/bin/python -m demo.rt.test_inject
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum


class FaultId(str, Enum):
    """写入九信号时间线的故障类型。"""

    NONE = "none"
    OBS_DELAY = "obs_delay"
    ENCODER_DROPOUT = "encoder_dropout"
    TORQUE_ATTENUATION = "torque_attenuation"


@dataclass(frozen=True)
class InjectionConfig:
    """三类注入参数；默认值均表示关闭。"""

    delay_ms: int = 0
    encoder_dropout_p: float = 0.0
    torque_scale: float = 1.0
    sample_hz: int = 1000

    def __post_init__(self) -> None:
        if self.sample_hz <= 0:
            raise ValueError("sample_hz 必须 > 0")
        if self.delay_ms < 0:
            raise ValueError("delay_ms 必须 >= 0")
        if not 0.0 <= self.encoder_dropout_p <= 1.0:
            raise ValueError("encoder_dropout_p 必须在 [0, 1]")
        if not 0.0 <= self.torque_scale <= 1.0:
            raise ValueError("torque_scale 必须在 [0, 1]")


@dataclass(frozen=True)
class ObservationResult:
    """一次观测注入结果。"""

    value: tuple[float, ...]
    dropped: bool
    fault_ids: tuple[FaultId, ...]


@dataclass(frozen=True)
class TorqueResult:
    """一次力矩注入结果。"""

    value: float
    fault_ids: tuple[FaultId, ...]


class SignalInjector:
    """带确定性随机源的有状态注入器。"""

    def __init__(self, config: InjectionConfig, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self.delay_steps = round(config.delay_ms * config.sample_hz / 1000)
        self._delay_buffer: deque[tuple[float, ...]] = deque(maxlen=self.delay_steps + 1)
        self._last_frame: tuple[float, ...] | None = None

    def reset(self, seed: int | None = None) -> None:
        """清空时延与保持状态；可指定新种子以开启新一轮复现实验。"""
        if seed is not None:
            self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._delay_buffer.clear()
        self._last_frame = None

    def process_observation(self, values: tuple[float, ...] | list[float]) -> ObservationResult:
        """先施加延迟，再执行编码器丢帧判定。"""
        current = tuple(float(value) for value in values)
        fault_ids: list[FaultId] = []

        if self.delay_steps > 0:
            self._delay_buffer.append(current)
            delayed = self._delay_buffer[0]
            fault_ids.append(FaultId.OBS_DELAY)
        else:
            delayed = current

        dropped = False
        if (
            self.config.encoder_dropout_p > 0.0
            and self._rng.random() < self.config.encoder_dropout_p
        ):
            dropped = True
            fault_ids.append(FaultId.ENCODER_DROPOUT)
            if self._last_frame is not None:
                delayed = self._last_frame

        self._last_frame = delayed
        return ObservationResult(value=delayed, dropped=dropped, fault_ids=tuple(fault_ids))

    def apply_torque(self, torque_cmd: float) -> TorqueResult:
        """按配置执行力矩衰减。"""
        value = float(torque_cmd)
        if self.config.torque_scale < 1.0:
            value *= self.config.torque_scale
            return TorqueResult(
                value=value, fault_ids=(FaultId.TORQUE_ATTENUATION,)
            )
        return TorqueResult(value=value, fault_ids=())


def run_trace(
    config: InjectionConfig,
    seed: int,
    ticks: int,
    *,
    sample_hz: int = 1000,
) -> list[tuple[tuple[float, ...], float, tuple[str, ...]]]:
    """生成用于复现性验收的逐拍注入时间线。"""
    if ticks < 0:
        raise ValueError("ticks 必须 >= 0")
    injector = SignalInjector(config, seed)
    trace: list[tuple[tuple[float, ...], float, tuple[str, ...]]] = []
    for tick in range(ticks):
        t_s = tick / sample_hz
        observation = injector.process_observation(
            (t_s, 0.5 * t_s, float(tick))
        )
        torque = injector.apply_torque(1.0 + 0.01 * tick)
        fault_ids = tuple(
            sorted({item.value for item in observation.fault_ids + torque.fault_ids})
        )
        trace.append((observation.value, torque.value, fault_ids))
    return trace
