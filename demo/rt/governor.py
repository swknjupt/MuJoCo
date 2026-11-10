"""Governor 状态机 v0（Task D / D1.5）。

运行时保障三件套中的"仲裁 + 回退"（见 taskD §3.1）：
  监视器（包络检查）+ 仲裁状态机 + 能量法回退控制器。

设计约束（taskD D1.5）：
  - 状态: NORMAL → FALLBACK → SAFE（单向 + 迟滞：一旦 FALLBACK 不自动回 NORMAL，
    人工复位才回）；
  - 触发（任一）: |θ|、|θ̇|、|τ| 越限；策略输出过期 > staleness_max；
    overrun 连续 > streak_limit 拍；
  - 角度包络采用 armed 语义：正常起摆从悬垂到倒立必须经过 100°，
    因此角度越界只在曾接近直立后生效；速度/力矩/失联/overrun 始终生效；
  - FALLBACK: 能量法——施加与速度反号、与（相对悬垂的）能量成正比的阻尼力矩，
    把摆缓缓放回下垂位；能量低于阈值 → SAFE（保持阻尼/抱闸）；
  - 纯确定性代码：无动态分配、无外部调用、无浮点环境依赖以外的状态。

接口（1kHz 每拍调用一次 step）：
    gov.step(theta, theta_dot, policy_age_s, overrun_streak, now_s)
        -> GovernorOutput(state, tau_cmd, transition_age_s, violated)

注意：本模块刻意只用标准库（部署目标为 RT 盒，依赖越少确定性越好）。
物理常数与 demo/models/pendulum_env.xml 一致：m=1kg, 杆长 0.8m（质心 l_c=0.4m）,
绕轴转动惯量 I = m*L²/3 ≈ 0.2133 kg·m²（MuJoCo capsule 细杆近似）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class GovernorState(Enum):
    NORMAL = 0    # 策略指挥（PD 跟踪策略目标）
    FALLBACK = 1  # 回退控制器指挥（能量法放倒）
    SAFE = 2      # 稳定在安全态（低能量 + 抱闸/保持阻尼）


@dataclass(frozen=True)
class GovernorOutput:
    state: GovernorState
    tau_cmd: float          # 回退模式下 governor 给出的力矩；NORMAL 模式下为 0（不参与）
    in_state_age_s: float   # 当前状态已持续时间
    violated: bool          # 本拍是否发生包络/条件违约


@dataclass(frozen=True)
class GovernorParams:
    # 包络（与 taskD 信号字典/阈值标定 D3.2 对应）
    # 注意：theta_max 必须允许"合法越顶"——起摆/直立本身就是 180°，若设 100° 会在
    # 正常操作中误触发（2026-11-10 由五幕动画集成复现发现）。发散检测的主力信号
    # 是角速度与连续疯转（270° = 3/4 圈，正常摆动绝不会越过，疯转必越）。
    theta_max: float = math.radians(270.0)  # 相对悬垂：拦截连续疯转
    arm_threshold: float = math.radians(30.0)  # 相对倒立的 armed 阈值
    fall_threshold: float = math.radians(60.0) # armed 后允许的倒立偏差
    theta_dot_max: float = 20.0              # rad/s
    tau_write_max: float = 5.0               # N·m，写出力矩上限（写路径断言，≤ 不越限；
                                             #   饱和本身是正常工况，不作为发散判据）
    policy_staleness_max_s: float = 0.030    # 策略输出保鲜期
    overrun_streak_limit: int = 3            # 连续 overrun 拍数上限
    # 能量法回退（泄能 + 对准悬垂）
    k_energy: float = 1.2                    # 能量→粘性阻尼系数比例
    c_visc_min: float = 0.4                  # 粘性阻尼系数下限（抑制末段振荡）
    kp_hang: float = 6.0                     # 对准悬垂的位置增益
    tau_fallback_max: float = 4.5            # 回退力矩上限（略低于电机极限，留裕度）
    e_safe_threshold: float = 0.05           # J，低于此值判 SAFE
    safe_damping: float = 2.0                # SAFE 态保持阻尼 N·m·s/rad
    # 物理常数（与模型一致）
    mass: float = 1.0
    l_com: float = 0.4
    inertia: float = 0.2133


class Governor:
    """每拍调用 step()；状态单向迁移，reset() 人工复位。"""

    def __init__(self, params: GovernorParams | None = None) -> None:
        self.p = params or GovernorParams()
        self._state = GovernorState.NORMAL
        self._state_since = 0.0
        self._last_now = 0.0
        self._armed = False

    # ---------------------------------------------------------- 包络检查
    def envelope_ok(self, theta: float, theta_dot: float, tau_written: float) -> bool:
        """监视器：三个不等式。tau_written 为本拍即将写出的力矩（裁剪后），
        ≤ 上限即合法——饱和是正常工况，此检查是写路径的最后断言；
        发散检测依靠 θ / θ̇ / 保鲜期 / overrun 四路信号。"""
        p = self.p
        return (
            abs(theta) <= p.theta_max
            and abs(theta_dot) <= p.theta_dot_max
            and abs(tau_written) <= p.tau_write_max
        )

    def theta_error(self, theta: float) -> float:
        """相对倒立位的角度误差，范围为 (-π, π]。"""
        return (math.pi - theta + math.pi) % (2.0 * math.pi) - math.pi

    def _armed_angle_ok(self, theta: float) -> bool:
        """接近直立后开启角度包络，避免正常起摆穿越 100° 时误触发。"""
        if abs(self.theta_error(theta)) < self.p.arm_threshold:
            self._armed = True
        return not self._armed or abs(self.theta_error(theta)) <= self.p.fall_threshold

    # ---------------------------------------------------------- 能量
    def energy(self, theta: float, theta_dot: float) -> float:
        """相对悬垂静止位的机械能（悬垂静止 = 0，倒立 ≈ 2*m*g*l_c ≈ 7.85 J）。"""
        p = self.p
        return (
            0.5 * p.inertia * theta_dot * theta_dot
            + p.mass * 9.81 * p.l_com * (1.0 - math.cos(theta))
        )

    # ---------------------------------------------------------- 回退控制器
    def fallback_torque(self, theta: float, theta_dot: float) -> float:
        """能量法回退 = 泄能项 + 对准项：τ = clip(−(k·E + c_min)·θ̇ − kp·θ, ±τmax)。

        - 泄能项（与速度反号、系数与能量成正比）负责消耗多余能量；
        - 对准项（−kp·θ，θ 相对悬垂）把摆缓缓拉回下垂位，消除纯阻尼的指数慢尾；
        - 两个教训（D1.5 测试中发现并修复）：
          ① 纯 sign 库仑制动会在制动力 > 重力矩处把摆"钉"在半空；
          ② 纯粘性阻尼的末段指数尾巴使 SAFE 到达时间超预算（4.2s > 2s）。
          本方案静止时力矩→0（θ→0, θ̇→0），无钉死陷阱，收敛快。
        """
        p = self.p
        e = self.energy(theta, theta_dot)
        c = p.k_energy * e + p.c_visc_min
        tau = -c * theta_dot - p.kp_hang * theta
        return max(-p.tau_fallback_max, min(p.tau_fallback_max, tau))

    def safe_torque(self, theta_dot: float) -> float:
        """SAFE 态：抱闸近似——速度阻尼。"""
        return max(-self.p.tau_fallback_max,
                   min(self.p.tau_fallback_max, -self.p.safe_damping * theta_dot))

    # ---------------------------------------------------------- 主入口
    def step(
        self,
        theta: float,
        theta_dot: float,
        tau_candidate: float,
        policy_age_s: float,
        overrun_streak: int,
        now_s: float,
    ) -> GovernorOutput:
        """每拍调用。tau_candidate = 本拍即将写出的力矩（NORMAL 态为裁剪后的 PD 输出）。"""
        dt = now_s - self._last_now if now_s >= self._last_now else 0.0
        self._last_now = now_s
        self._state_since += dt
        p = self.p

        if self._state is GovernorState.NORMAL:
            violated = (
                not self._armed_angle_ok(theta)
                or abs(theta_dot) > p.theta_dot_max
                or abs(tau_candidate) > p.tau_write_max
                or policy_age_s > p.policy_staleness_max_s
                or overrun_streak > p.overrun_streak_limit
            )
            if violated:
                self._state, self._state_since = GovernorState.FALLBACK, 0.0
            tau = 0.0
        elif self._state is GovernorState.FALLBACK:
            violated = False
            tau = self.fallback_torque(theta, theta_dot)
            if self.energy(theta, theta_dot) < p.e_safe_threshold:
                self._state, self._state_since = GovernorState.SAFE, 0.0
        else:  # SAFE
            violated = False
            tau = self.safe_torque(theta_dot)

        return GovernorOutput(
            state=self._state,
            tau_cmd=tau,
            in_state_age_s=self._state_since,
            violated=violated,
        )

    # ---------------------------------------------------------- 复位
    def reset(self) -> None:
        """人工复位（对应演示流程中确认安全后恢复 NORMAL）。"""
        self._state = GovernorState.NORMAL
        self._state_since = 0.0
        self._last_now = 0.0
        self._armed = False

    @property
    def state(self) -> GovernorState:
        return self._state

    @property
    def armed(self) -> bool:
        """是否已进入过直立区并开启跌落包络。"""
        return self._armed
