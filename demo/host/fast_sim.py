"""硬件无关的快速单摆故障仿真核心（Task D / D3.1–D3.3）。

该仿真器复用 ONNX actor、PD、故障注入和 governor 的逻辑，但同步执行策略推理，
用于参数扫描和预标定；它不是 RT-SIL 或真机证据。
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from demo.rt.governor import Governor, GovernorParams, GovernorState
from demo.rt.inference import OnnxCpuBackend
from demo.rt.inject import FaultId, InjectionConfig, SignalInjector

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "examples" / "output" / "policy.onnx"
CONTROL_HZ = 1000
POLICY_HZ = 100
DECIMATION = CONTROL_HZ // POLICY_HZ
K_PD = 50.0
KD_PD = 6.5
TORQUE_LIMIT = 5.0


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class FaultScenario:
    """快速扫描使用的故障场景。"""

    delay_ms: int = 0
    encoder_dropout_p: float = 0.0
    torque_scale: float = 1.0
    fault_at_s: float = 2.0

    def injection(self) -> InjectionConfig:
        return InjectionConfig(
            delay_ms=self.delay_ms,
            encoder_dropout_p=self.encoder_dropout_p,
            torque_scale=self.torque_scale,
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FaultScenario":
        return cls(
            delay_ms=int(value["delay_ms"]),
            encoder_dropout_p=float(value["encoder_dropout_p"]),
            torque_scale=float(value["torque_scale"]),
            fault_at_s=float(value.get("fault_at_s", 2.0)),
        )

    @property
    def severity(self) -> float:
        return (
            self.delay_ms / 100.0
            + self.encoder_dropout_p
            + (1.0 - self.torque_scale)
        )


@dataclass(frozen=True)
class FastSignal:
    """可选的逐拍快速仿真信号。"""

    timestamp_us: int
    theta: float
    theta_dot: float
    torque_cmd: float
    policy_target: float
    envelope_ok: bool
    governor_state: str
    injected_fault_id: str


@dataclass(frozen=True)
class FastSimResult:
    """一次快速仿真的结果。"""

    seed: int
    scenario: FaultScenario
    governor_enabled: bool
    duration_s: float
    diverged: bool
    divergence_s: float | None
    violation_s: float | None
    takeover_s: float | None
    safe_s: float | None
    takeover_to_safe_s: float | None
    max_abs_torque_nm: float
    max_fallback_torque_nm: float
    max_abs_theta_dot: float
    final_abs_error_deg: float
    signals: tuple[FastSignal, ...] = ()

    def to_dict(self, *, include_signals: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if not include_signals:
            result.pop("signals")
        return result


def governor_params_to_dict(params: GovernorParams) -> dict[str, float]:
    """把角度转换为 JSON 友好的度。"""
    value = asdict(params)
    for name in ("theta_max", "arm_threshold", "fall_threshold"):
        value[f"{name}_deg"] = math.degrees(value.pop(name))
    return value


def governor_params_from_dict(
    value: dict[str, Any],
    base: GovernorParams | None = None,
) -> GovernorParams:
    """从 JSON 配置构造 GovernorParams。"""
    base = base or GovernorParams()
    changes: dict[str, Any] = {}
    for name in value:
        if name.endswith("_deg"):
            field_name = name.removesuffix("_deg")
            if field_name in {item.name for item in fields(GovernorParams)}:
                changes[field_name] = math.radians(float(value[name]))
        elif name in {item.name for item in fields(GovernorParams)}:
            current = getattr(base, name)
            changes[name] = (
                int(round(float(value[name])))
                if isinstance(current, int)
                else float(value[name])
            )
    return replace(base, **changes)


class FastPendulum:
    """与 demo 模型同量级的确定性单摆。"""

    def __init__(self, seed: int, *, near_upright: bool = True) -> None:
        rng = random.Random(seed)
        center = math.pi if near_upright else 0.0
        self.theta = center + rng.normalvariate(0.0, 0.05)
        self.theta_dot = rng.normalvariate(0.0, 0.05)
        self.mass = 1.0
        self.l_com = 0.4
        self.inertia = 0.2133
        self.damping = 0.05

    def step(self, torque_nm: float) -> None:
        dt = 0.001
        acceleration = (
            torque_nm
            - self.damping * self.theta_dot
            - self.mass * 9.81 * self.l_com * math.sin(self.theta)
        ) / self.inertia
        self.theta += self.theta_dot * dt
        self.theta_dot += acceleration * dt

    def observation(self, theta: float, theta_dot: float):
        import numpy as np

        error = wrap_to_pi(math.pi - theta)
        return np.array(
            [math.cos(theta), math.sin(theta), theta_dot / 10.0, error / math.pi],
            dtype=np.float32,
        )


class FastSimulator:
    """缓存 ONNX session 的快速仿真器。"""

    def __init__(self, onnx_path: Path = DEFAULT_ONNX) -> None:
        self.backend = OnnxCpuBackend(onnx_path)

    def run(
        self,
        scenario: FaultScenario,
        *,
        seed: int,
        governor_enabled: bool,
        governor_params: GovernorParams | None = None,
        duration_s: float = 6.0,
        start_near_upright: bool = True,
        collect_signals: bool = False,
    ) -> FastSimResult:
        if duration_s <= 0:
            raise ValueError("duration_s 必须 > 0")
        plant = FastPendulum(seed, near_upright=start_near_upright)
        governor = Governor(governor_params)
        injector = SignalInjector(InjectionConfig(), seed=seed)
        fault_injector = SignalInjector(scenario.injection(), seed=seed + 10_000)
        fault_tick = round(scenario.fault_at_s * CONTROL_HZ)
        total_ticks = round(duration_s * CONTROL_HZ)

        current_target = math.pi
        armed = abs(wrap_to_pi(math.pi - plant.theta)) < math.radians(30)
        divergence_s = None
        violation_s = None
        takeover_s = None
        safe_s = None
        max_torque = 0.0
        max_fallback_torque = 0.0
        max_theta_dot = 0.0
        signals: list[FastSignal] = []

        for tick in range(total_ticks):
            now_s = tick / CONTROL_HZ
            active_injector = (
                fault_injector if tick >= fault_tick else injector
            )
            theta_safe = plant.theta
            theta_dot_safe = plant.theta_dot
            observation_result = active_injector.process_observation(
                (theta_safe, theta_dot_safe)
            )
            theta_injected, theta_dot_injected = observation_result.value

            error = wrap_to_pi(math.pi - theta_safe)
            if abs(error) < math.radians(30):
                armed = True
            if divergence_s is None and (
                (armed and abs(error) > math.radians(60))
                or abs(theta_dot_safe) > 20.0
            ):
                divergence_s = now_s

            if tick % DECIMATION == 0:
                observation = plant.observation(theta_injected, theta_dot_injected)
                current_target = float(self.backend.infer(observation)[0])

            pd_error = wrap_to_pi(current_target - theta_injected)
            pd_torque = max(
                -TORQUE_LIMIT,
                min(TORQUE_LIMIT, K_PD * pd_error - KD_PD * theta_dot_injected),
            )
            torque_result = active_injector.apply_torque(pd_torque)
            torque_candidate = torque_result.value
            fault_ids = observation_result.fault_ids + torque_result.fault_ids

            if governor_enabled:
                governor_output = governor.step(
                    theta=theta_safe,
                    theta_dot=theta_dot_safe,
                    tau_candidate=torque_candidate,
                    policy_age_s=0.0,
                    overrun_streak=0,
                    now_s=now_s,
                )
                if governor_output.violated and violation_s is None:
                    violation_s = now_s
                if (
                    governor_output.state is GovernorState.FALLBACK
                    and takeover_s is None
                ):
                    takeover_s = now_s
                if governor_output.state is GovernorState.SAFE and safe_s is None:
                    safe_s = now_s
                torque_cmd = (
                    torque_candidate
                    if governor_output.state is GovernorState.NORMAL
                    else governor_output.tau_cmd
                )
                if governor_output.state is not GovernorState.NORMAL:
                    max_fallback_torque = max(
                        max_fallback_torque, abs(torque_cmd)
                    )
                governor_state = governor_output.state.name
                envelope_ok = not governor_output.violated
            else:
                torque_cmd = torque_candidate
                governor_state = "DISABLED"
                envelope_ok = not (
                    (armed and abs(error) > math.radians(60))
                    or abs(theta_dot_safe) > 20.0
                )

            max_torque = max(max_torque, abs(torque_cmd))
            max_theta_dot = max(max_theta_dot, abs(theta_dot_safe))
            plant.step(torque_cmd)
            if collect_signals:
                signals.append(
                    FastSignal(
                        timestamp_us=round(now_s * 1_000_000),
                        theta=theta_safe,
                        theta_dot=theta_dot_safe,
                        torque_cmd=torque_cmd,
                        policy_target=current_target,
                        envelope_ok=envelope_ok,
                        governor_state=governor_state,
                        injected_fault_id=(
                            "none"
                            if not fault_ids
                            else "|".join(sorted({item.value for item in fault_ids}))
                        ),
                    )
                )

        takeover_to_safe = (
            safe_s - takeover_s if safe_s is not None and takeover_s is not None else None
        )
        return FastSimResult(
            seed=seed,
            scenario=scenario,
            governor_enabled=governor_enabled,
            duration_s=duration_s,
            diverged=divergence_s is not None,
            divergence_s=divergence_s,
            violation_s=violation_s,
            takeover_s=takeover_s,
            safe_s=safe_s,
            takeover_to_safe_s=takeover_to_safe,
            max_abs_torque_nm=max_torque,
            max_fallback_torque_nm=max_fallback_torque,
            max_abs_theta_dot=max_theta_dot,
            final_abs_error_deg=abs(math.degrees(wrap_to_pi(math.pi - plant.theta))),
            signals=tuple(signals),
        )
