"""RT-SIL 四幕集成运行与九信号时间线（Task D / D1.8）。

串联以下组件：
  - D1.1 单摆模型 / D1.2 PPO 策略
  - D1.3 ONNX actor / D1.4 1kHz 周期循环
  - D1.5 governor / D1.6 timeboxed 推理壳 / D1.7 故障注入

第 1/2 幕验证常态与 5ms 延迟；第 3/4 幕使用同一极端注入做无保护/
有保护 A/B。host 上可加开发机参数快速联调；RT 盒上直接运行即为 L2 证据。

运行（RT 盒）：
    sudo python3 demo/rt/run_rt_sil.py

开发机功能联调（不具备实时验收效力）：
    .venv/bin/python demo/rt/run_rt_sil.py --allow-unprivileged --allow-non-rt

候选故障和预调 Governor 复核：
    .venv/bin/python demo/rt/run_rt_sil.py --acts 3 4 \
        --scenario-file examples/output/fault_scan/fault_scenario.json \
        --governor-config examples/output/governor_tuning/governor_params.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.rt.governor import Governor, GovernorParams, GovernorState
from demo.rt.inference import OnnxCpuBackend, TimeboxedInferenceShell
from demo.rt.inject import FaultId, InjectionConfig, SignalInjector
from demo.rt.loop import LoopConfig, LoopResult, run_realtime_loop
from demo.rt.plant_sim import PendulumPlant, wrap_to_pi

OUTPUT_DIR = ROOT / "examples" / "output"
DEFAULT_ONNX = OUTPUT_DIR / "policy.onnx"
DEFAULT_REPORT = OUTPUT_DIR / "rt_sil_report.json"

CONTROL_HZ = 1000
POLICY_HZ = 100
DECIMATION = CONTROL_HZ // POLICY_HZ
K_PD = 50.0
KD_PD = 6.5
TORQUE_LIMIT = 5.0


@dataclass(frozen=True)
class Scenario:
    """单幕运行配置。"""

    act: int
    name: str
    duration_s: float
    governor_enabled: bool
    initial_injection: InjectionConfig = InjectionConfig()
    extreme_injection_at_s: float | None = None
    extreme_injection: InjectionConfig | None = None
    start_near_upright: bool = False


@dataclass(frozen=True)
class SignalRow:
    """九信号时间线的一行。"""

    timestamp_us: int
    theta: float
    theta_dot: float
    torque_cmd: float
    policy_target: float
    policy_latency_ms: float
    envelope_ok: bool
    governor_state: str
    injected_fault_id: str


@dataclass(frozen=True)
class ActResult:
    """单幕验收结果。"""

    act: int
    name: str
    passed: bool
    reached_upright: bool
    max_hold_s: float
    max_abs_theta_error_deg: float
    max_abs_theta_dot: float
    divergence: bool
    takeover_s: float | None
    takeover_latency_ms: float | None
    safe_s: float | None
    timeout_count: int
    loop: LoopResult
    signal_csv: str


SCENARIOS = (
    Scenario(act=1, name="常态起摆保持", duration_s=4.0, governor_enabled=True),
    Scenario(
        act=2,
        name="5ms 延迟",
        duration_s=4.0,
        governor_enabled=True,
        initial_injection=InjectionConfig(delay_ms=5),
    ),
    Scenario(
        act=3,
        name="极端故障 / governor 关闭",
        duration_s=6.0,
        governor_enabled=False,
        extreme_injection_at_s=2.0,
        extreme_injection=InjectionConfig(
            delay_ms=100, encoder_dropout_p=1.0, torque_scale=0.3
        ),
    ),
    Scenario(
        act=4,
        name="极端故障 / governor 开启",
        duration_s=6.0,
        governor_enabled=True,
        extreme_injection_at_s=2.0,
        extreme_injection=InjectionConfig(
            delay_ms=100, encoder_dropout_p=1.0, torque_scale=0.3
        ),
    ),
)


def _fault_text(fault_ids: tuple[FaultId, ...]) -> str:
    """把多个同时发生的故障合并为稳定字符串。"""
    if not fault_ids:
        return FaultId.NONE.value
    return "|".join(sorted({fault_id.value for fault_id in fault_ids}))


def _wait_for_policy(shell: TimeboxedInferenceShell, timeout_s: float = 0.2) -> None:
    """启动阶段等待首条有效策略输出，避免把初始化误判为失联。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if shell.get_latest().valid:
            return
        time.sleep(0.001)
    raise RuntimeError("推理壳未在启动窗口内产生有效策略输出")


def run_act(
    scenario: Scenario,
    *,
    onnx_path: Path,
    output_dir: Path,
    control_cpu: int,
    inference_cpu: int,
    governor_params: GovernorParams | None,
    allow_unprivileged: bool,
    allow_non_rt: bool,
) -> ActResult:
    """运行单幕并落盘九信号 CSV。"""
    plant = PendulumPlant()
    # 第 3/4 幕必须从完全相同的初始状态出发，保证 A/B 只差 governor 开关。
    initial_seed = 300 if scenario.act in (3, 4) else 100 + scenario.act
    plant.reset(seed=initial_seed, near_upright=scenario.start_near_upright)
    injector = SignalInjector(scenario.initial_injection, seed=200 + scenario.act)
    governor = Governor(governor_params)

    shell = TimeboxedInferenceShell(
        OnnxCpuBackend(onnx_path),
        worker_cpu=inference_cpu,
        period_s=1.0 / POLICY_HZ,
    )
    shell.start()

    rows: list[SignalRow] = []
    state = {
        "armed": False,
        "hold_start_s": None,
        "max_hold_s": 0.0,
        "max_error_deg": 0.0,
        "max_theta_dot": 0.0,
        "divergence": False,
        "violation_s": None,
        "takeover_s": None,
        "safe_s": None,
        "previous_governor": GovernorState.NORMAL,
    }
    active_injector = injector

    def step(tick: int, _deadline_ns: int) -> None:
        nonlocal active_injector
        now_s = tick / CONTROL_HZ
        if (
            scenario.extreme_injection_at_s is not None
            and tick == round(scenario.extreme_injection_at_s * CONTROL_HZ)
            and scenario.extreme_injection is not None
        ):
            active_injector = SignalInjector(
                scenario.extreme_injection, seed=500 + scenario.act
            )

        # 安全通道必须读取原始编码器；故障注入只污染策略/PD 的输入链路。
        theta_safe = plant.theta
        theta_dot_safe = plant.theta_dot
        observation_result = active_injector.process_observation(
            (theta_safe, theta_dot_safe)
        )
        theta_injected, theta_dot_injected = observation_result.value
        error_deg = abs(math.degrees(wrap_to_pi(math.pi - theta_safe)))
        if error_deg < 30.0:
            state["armed"] = True
            if state["hold_start_s"] is None:
                state["hold_start_s"] = now_s
            state["max_hold_s"] = max(
                float(state["max_hold_s"]),
                now_s - float(state["hold_start_s"]),
            )
        else:
            state["hold_start_s"] = None
        if state["armed"] and error_deg > 60.0:
            state["divergence"] = True
        if abs(theta_dot_safe) > 20.0:
            state["divergence"] = True

        observation = plant.observation_from_state(theta_injected, theta_dot_injected)
        if tick % DECIMATION == 0:
            shell.update(observation)
        latest = shell.get_latest()
        policy_target = float(latest.action[0]) if latest.valid else math.pi

        pd_error = wrap_to_pi(policy_target - theta_injected)
        pd_torque = max(
            -TORQUE_LIMIT,
            min(TORQUE_LIMIT, K_PD * pd_error - KD_PD * theta_dot_injected),
        )
        torque_result = active_injector.apply_torque(pd_torque)
        torque_candidate = torque_result.value
        fault_ids = observation_result.fault_ids + torque_result.fault_ids

        if scenario.governor_enabled:
            governor_output = governor.step(
                theta=theta_safe,
                theta_dot=theta_dot_safe,
                tau_candidate=torque_candidate,
                policy_age_s=latest.age_s,
                overrun_streak=0,
                now_s=now_s,
            )
            if governor_output.violated and state["violation_s"] is None:
                state["violation_s"] = now_s
            if (
                governor_output.state is GovernorState.FALLBACK
                and state["previous_governor"] is not GovernorState.FALLBACK
                and state["takeover_s"] is None
            ):
                state["takeover_s"] = now_s
            if (
                governor_output.state is GovernorState.SAFE
                and state["safe_s"] is None
            ):
                state["safe_s"] = now_s
            torque_cmd = (
                torque_candidate
                if governor_output.state is GovernorState.NORMAL
                else governor_output.tau_cmd
            )
            governor_state = governor_output.state.name
            envelope_ok = not governor_output.violated
            state["previous_governor"] = governor_output.state
        else:
            torque_cmd = torque_candidate
            governor_state = "DISABLED"
            envelope_ok = not (
                (state["armed"] and error_deg > 60.0)
                or abs(theta_dot_safe) > 20.0
            )

        plant.step(torque_cmd)
        state["max_error_deg"] = max(
            float(state["max_error_deg"]), error_deg
        )
        state["max_theta_dot"] = max(
            float(state["max_theta_dot"]), abs(theta_dot_safe)
        )
        rows.append(
            SignalRow(
                timestamp_us=round(now_s * 1_000_000),
                theta=theta_safe,
                theta_dot=theta_dot_safe,
                torque_cmd=torque_cmd,
                policy_target=policy_target,
                policy_latency_ms=latest.age_s * 1000.0,
                envelope_ok=envelope_ok,
                governor_state=governor_state,
                injected_fault_id=_fault_text(fault_ids),
            )
        )

    try:
        initial_injection = injector.process_observation((plant.theta, plant.theta_dot))
        initial_theta, initial_theta_dot = initial_injection.value
        shell.update(plant.observation_from_state(initial_theta, initial_theta_dot))
        _wait_for_policy(shell)
        loop_result = run_realtime_loop(
            LoopConfig(duration_s=scenario.duration_s, cpu=control_cpu),
            step,
            allow_unprivileged=allow_unprivileged,
            allow_non_rt=allow_non_rt,
        )
    finally:
        shell.stop()

    signal_csv = output_dir / f"rt_sil_act{scenario.act}_signals.csv"
    signal_csv.parent.mkdir(parents=True, exist_ok=True)
    with signal_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SignalRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    try:
        signal_csv_label = str(signal_csv.relative_to(ROOT))
    except ValueError:
        signal_csv_label = str(signal_csv)

    takeover_s = state["takeover_s"]
    violation_s = state["violation_s"]
    takeover_latency_ms = (
        (float(takeover_s) - float(violation_s)) * 1000.0
        if takeover_s is not None and violation_s is not None
        else None
    )
    takeover_safe_s = (
        float(state["safe_s"]) - float(takeover_s)
        if state["safe_s"] is not None and takeover_s is not None
        else None
    )

    if scenario.act in (1, 2):
        passed = bool(state["armed"]) and float(state["max_hold_s"]) >= 2.0
    elif scenario.act == 3:
        passed = bool(state["divergence"])
    else:
        passed = (
            bool(state["divergence"])
            and takeover_latency_ms is not None
            and takeover_latency_ms < 5.0
            and takeover_safe_s is not None
            and takeover_safe_s < 2.0
        )

    return ActResult(
        act=scenario.act,
        name=scenario.name,
        passed=passed,
        reached_upright=bool(state["armed"]),
        max_hold_s=float(state["max_hold_s"]),
        max_abs_theta_error_deg=float(state["max_error_deg"]),
        max_abs_theta_dot=float(state["max_theta_dot"]),
        divergence=bool(state["divergence"]),
        takeover_s=takeover_s,
        takeover_latency_ms=takeover_latency_ms,
        safe_s=state["safe_s"],
        timeout_count=shell.timeout_count,
        loop=loop_result,
        signal_csv=signal_csv_label,
    )


def save_report(
    report_path: Path,
    results: list[ActResult],
    *,
    host_only: bool,
) -> None:
    """保存四幕集成报告。"""
    report = {
        "passed": all(result.passed for result in results),
        "note": (
            "host 开发机运行仅证明软件链路；jitter/overrun 的对外证据必须由 "
            "PREEMPT_RT 目标机产出"
            if host_only
            else "T14 PREEMPT_RT 目标机运行；本报告可作为 L2 时序证据"
        ),
        "acts": [asdict(result) for result in results],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_scenario_file(path: Path) -> tuple[InjectionConfig, float]:
    """加载快速扫描生成的候选故障 JSON。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"delay_ms", "encoder_dropout_p", "torque_scale"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"场景文件缺少字段：{sorted(missing)}")
    injection = InjectionConfig(
        delay_ms=int(value["delay_ms"]),
        encoder_dropout_p=float(value["encoder_dropout_p"]),
        torque_scale=float(value["torque_scale"]),
    )
    return injection, float(value.get("fault_at_s", 2.0))


def select_scenarios(
    acts: list[int],
    scenario_file: Path | None,
) -> tuple[Scenario, ...]:
    """按幕号筛选场景，并用候选配置覆盖第 3/4 幕。"""
    by_act = {scenario.act: scenario for scenario in SCENARIOS}
    if scenario_file is not None:
        injection, fault_at_s = load_scenario_file(scenario_file)
        for act in (3, 4):
            by_act[act] = replace(
                by_act[act],
                extreme_injection_at_s=fault_at_s,
                extreme_injection=injection,
            )
    invalid = [act for act in acts if act not in by_act]
    if invalid:
        raise ValueError(f"未知幕号：{invalid}")
    return tuple(by_act[act] for act in acts)


def load_governor_config(path: Path) -> GovernorParams:
    """从 JSON 加载 Governor 参数；角度字段使用 `_deg` 后缀。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    changes: dict[str, float] = {}
    for name, raw in value.items():
        if name == "theta_max_deg":
            changes["theta_max"] = math.radians(float(raw))
        elif name == "arm_threshold_deg":
            changes["arm_threshold"] = math.radians(float(raw))
        elif name == "fall_threshold_deg":
            changes["fall_threshold"] = math.radians(float(raw))
        elif hasattr(GovernorParams(), name):
            changes[name] = float(raw)
    return replace(GovernorParams(), **changes)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 RT-SIL 第 1–4 幕")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--control-cpu", type=int, default=1)
    parser.add_argument("--inference-cpu", type=int, default=2)
    parser.add_argument("--scenario-file", type=Path)
    parser.add_argument("--governor-config", type=Path)
    parser.add_argument(
        "--acts",
        nargs="+",
        type=int,
        choices=range(1, 5),
        default=[1, 2, 3, 4],
    )
    parser.add_argument("--allow-unprivileged", action="store_true")
    parser.add_argument("--allow-non-rt", action="store_true")
    args = parser.parse_args()

    if not args.onnx.exists():
        parser.error(f"ONNX 不存在：{args.onnx}")

    scenarios = select_scenarios(args.acts, args.scenario_file)
    governor_params = (
        load_governor_config(args.governor_config)
        if args.governor_config is not None
        else None
    )
    results: list[ActResult] = []
    for scenario in scenarios:
        print(f"\n=== 第 {scenario.act} 幕：{scenario.name} ===")
        result = run_act(
            scenario,
            onnx_path=args.onnx,
            output_dir=args.output_dir,
            control_cpu=args.control_cpu,
            inference_cpu=args.inference_cpu,
            governor_params=governor_params,
            allow_unprivileged=args.allow_unprivileged,
            allow_non_rt=args.allow_non_rt,
        )
        results.append(result)
        takeover_text = (
            f"{result.takeover_latency_ms:.3f}ms"
            if result.takeover_latency_ms is not None
            else "未接管"
        )
        print(
            f"armed={result.reached_upright} hold={result.max_hold_s:.2f}s "
            f"divergence={result.divergence} takeover={takeover_text} "
            f"timeout={result.timeout_count} {'✔' if result.passed else '✘'}"
        )

    save_report(
        args.report,
        results,
        host_only=args.allow_non_rt or args.allow_unprivileged,
    )
    print(f"\n报告：{args.report}")
    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
