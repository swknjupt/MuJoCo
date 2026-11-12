"""D1.5 验收测试：governor 状态机（仿真时间，纯标准库，无外部依赖）。

运行（仓库根目录，任意 python3 ≥3.10 均可）：
    .venv/bin/python -m demo.rt.test_governor

验收项（taskD §4.1 D1.5）：
  1. 仿真注入发散场景（强制策略输出大幅错误值驱动 PD），
     越界 → 接管 < 5ms（仿真时间）；
  2. 接管后 2s 内到达 SAFE；
  3. 状态单向性：FALLBACK/SAFE 不自动回 NORMAL；reset() 人工复位有效。
"""

from __future__ import annotations

import math

from demo.rt.governor import Governor, GovernorParams, GovernorState

DT = 0.001  # 1ms，与部署拍周期一致


class PendulumPlant:
    """极简确定性摆模型（Euler 积分），参数与 pendulum_env.xml 一致。"""

    def __init__(self, theta0: float = 0.0, thetadot0: float = 0.0) -> None:
        self.m, self.l_c, self.I, self.b = 1.0, 0.4, 0.2133, 0.05
        self.theta, self.theta_dot = theta0, thetadot0

    def step(self, tau: float) -> None:
        acc = (tau - self.b * self.theta_dot - self.m * 9.81 * self.l_c * math.sin(self.theta)) / self.I
        self.theta += self.theta_dot * DT
        self.theta_dot += acc * DT


def test_divergence_catch() -> None:
    """验收 1+2：发散场景，越界→接管 <5ms，2s 内到 SAFE。

    语义说明：<5ms 测的是"包络越限时刻 → governor 接管"的监视器延迟
    （每拍检查 → 应为 1 拍 = 1ms）；发散本身从策略发疯到状态越限所需的
    物理时间不计入（那是包络阈值标定 D3.2 的研究对象）。
    """
    p = GovernorParams()
    gov = Governor(p)
    # 场景必须先到达直立区，armed 后的角度跌落才代表“平衡后被故障甩飞”。
    # 从悬垂直接注入大目标属于正常起摆路径，不应被角度包络误判。
    plant = PendulumPlant(theta0=math.pi - 0.2, thetadot0=0.0)

    # 场景：策略输出疯狂目标角（模拟发散），1kHz PD 跟踪 → 状态甩出包络
    kp, kd = 25.0, 4.5
    t = 0.0
    violation_t = None      # 包络首次越限时刻（与 governor 同一包络定义）
    takeover_t = None       # governor 进入 FALLBACK 时刻
    safe_t = None           # 到达 SAFE 时刻

    for i in range(6000):   # 最多 6s
        # 策略发散输出：随时间增大的目标角（+π/−π 交替漂移）
        theta_cmd = math.pi * math.sin(6.0 * t) if t < 1.0 else 3 * math.pi
        tau_raw = kp * (theta_cmd - plant.theta) - kd * plant.theta_dot
        tau_written = max(-5.0, min(5.0, tau_raw))   # 写出力矩（裁剪后）

        if gov.state is GovernorState.NORMAL:
            out = gov.step(plant.theta, plant.theta_dot, tau_written,
                           policy_age_s=0.0, overrun_streak=0, now_s=t)
            if out.violated and violation_t is None:
                violation_t = t
            if out.state is GovernorState.FALLBACK and takeover_t is None:
                takeover_t = t
            tau = tau_written
        else:
            out = gov.step(plant.theta, plant.theta_dot, 0.0,
                           policy_age_s=0.0, overrun_streak=0, now_s=t)
            tau = out.tau_cmd
            if out.state is GovernorState.SAFE and safe_t is None:
                safe_t = t
        plant.step(tau)
        t += DT

    assert violation_t is not None, "场景未触发越限（测试设置有误）"
    assert takeover_t is not None, "governor 未接管"
    catch = (takeover_t - violation_t) * 1000.0
    assert catch < 5.0, f"越界→接管耗时 {catch:.2f}ms，超过 5ms"
    assert safe_t is not None, "未在 6s 内到达 SAFE"
    settle = safe_t - takeover_t
    assert settle < 2.0, f"接管→SAFE 耗时 {settle:.2f}s，超过 2s"
    print(f"[验收1] 越界→接管 {catch:.2f}ms (<5ms) ✔")
    print(f"[验收2] 接管→SAFE {settle:.2f}s (<2s) ✔  终态θ={plant.theta:.4f} rad")


def test_one_way_and_reset() -> None:
    """验收 3：状态单向性 + 人工复位。"""
    gov = Governor()
    # 先正常到达直立附近，armed 后才允许角度跌落判据生效
    out = gov.step(theta=math.pi, theta_dot=0.0, tau_candidate=0.0, policy_age_s=0.0,
                   overrun_streak=0, now_s=0.0005)
    assert out.state is GovernorState.NORMAL and gov.armed
    # 制造 armed 后角度违约 → FALLBACK
    out = gov.step(theta=1.8, theta_dot=0.0, tau_candidate=0.0, policy_age_s=0.0,
                   overrun_streak=0, now_s=0.001)   # 1.8rad > 100°限
    assert out.state is GovernorState.FALLBACK
    # 之后即使状态完美也不自动回 NORMAL
    for i in range(2000):
        out = gov.step(theta=0.0, theta_dot=0.0, tau_candidate=0.0, policy_age_s=0.0,
                       overrun_streak=0, now_s=0.001 * (i + 2))
    assert out.state is GovernorState.SAFE, "低能量应进入 SAFE"
    # 人工复位
    gov.reset()
    assert gov.state is GovernorState.NORMAL
    assert not gov.armed
    print("[验收3] 单向状态机 + 人工复位 ✔")


def test_swing_through_is_not_violation() -> None:
    """正常起摆穿越 100° 但尚未接近直立时不得误触发。"""
    gov = Governor()
    for index, theta in enumerate((0.3, 0.9, 1.8, 2.5)):
        out = gov.step(
            theta=theta,
            theta_dot=5.0,
            tau_candidate=2.0,
            policy_age_s=0.0,
            overrun_streak=0,
            now_s=0.001 * (index + 1),
        )
        assert out.state is GovernorState.NORMAL
        assert not out.violated
    assert not gov.armed
    print("[附加] 正常起摆穿越 100° 不误触发 ✔")


def test_staleness_and_overrun_triggers() -> None:
    """附加：策略失联与 overrun 连拍两个触发路径。"""
    gov = Governor()
    out = gov.step(0.0, 0.0, 0.0, policy_age_s=0.040, overrun_streak=0, now_s=0.001)
    assert out.state is GovernorState.FALLBACK, "策略过期 40ms 应触发"
    gov.reset()
    out = gov.step(0.0, 0.0, 0.0, policy_age_s=0.0, overrun_streak=4, now_s=0.001)
    assert out.state is GovernorState.FALLBACK, "overrun 连续 4 拍应触发"
    print("[附加] 失联触发 / overrun 连拍触发 ✔")


if __name__ == "__main__":
    test_divergence_catch()
    test_one_way_and_reset()
    test_swing_through_is_not_violation()
    test_staleness_and_overrun_triggers()
    print("\n=== D1.5 governor 全部验收通过 ===")
