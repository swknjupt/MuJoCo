"""单摆 Gymnasium 环境（Task D / D1.1）。

核心设计约束：**环境结构必须镜像部署架构**——
  策略以 100Hz 输出目标角度（action），env 内部以 1kHz 执行 PD 子步跟踪（decimation=10）。
这与真机部署形态（100Hz 策略出目标 + 1kHz 低层 PD 执行）天然同构，
训练出的策略可零改动迁移到 RT 盒推理。

约定：
  - 摆角 θ 取 MuJoCo 关节角，qpos=0 为竖直下垂，倒立位为 ±π；
  - θ_err = wrapToPi(π − θ) 为相对倒立位的偏差（0 = 直立）；
  - 观测 [cosθ, sinθ, θ̇/v_max, θ_err/π]，均归一化到 O(1)；
  - 奖励 = 直立项 − 角度偏差 − 速度惩罚 − 力矩惩罚（权重可配，供 D1.2 迭代）；
  - 终止采用 armed 逻辑：摆曾接近直立（|θ_err|<30°）后若 |θ_err|>60° 判倒下——
    与 governor 包络语义一致（"到过顶后跌落"才算失效，初始悬垂不误杀）；
  - obs_delay_ms：观测延迟（零阶保持，按 PD 子步数取整），为 D3 退化实验预留。

运行（仓库根目录）：
    .venv/bin/python -m demo.envs.test_pendulum_env
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

# 模型文件相对本文件定位（CWD 无关，遵循仓库约定）
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pendulum_env.xml"


def wrap_to_pi(angle: float) -> float:
    """把角度归约到 (-π, π]。"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class PendulumEnv(gym.Env):
    """单摆起摆-保持环境（MuJoCo 后端，1kHz 物理 + 1kHz PD + 100Hz 策略）。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        pd_freq: int = 1000,
        policy_freq: int = 100,
        kp: float = 50.0,
        kd: float = 6.5,
        torque_limit: float = 5.0,
        obs_delay_ms: float = 0.0,
        fall_threshold_deg: float = 60.0,
        arm_threshold_deg: float = 30.0,
        max_episode_s: float = 10.0,
        w_upright: float = 1.0,
        w_angle: float = 0.1,
        w_vel: float = 0.01,
        w_torque: float = 0.001,
        vel_norm: float = 10.0,
    ) -> None:
        super().__init__()
        assert pd_freq % policy_freq == 0, "pd_freq 必须是 policy_freq 的整数倍"
        self.decimation = pd_freq // policy_freq
        self.dt_policy = 1.0 / policy_freq
        self.pd_freq = pd_freq

        # PD 增益（1kHz 执行）：kp=50/kd=6.5——临界阻尼附近；
        # kp 取高的原因：163° 目标处重力矩 3.7 N·m，P 控制静差 ≈ 重力/kp，
        # kp=50 时 <0.03 rad（kp=25 实测 0.055 rad 超验收线 0.05）。
        self.kp, self.kd = float(kp), float(kd)
        self.torque_limit = float(torque_limit)

        # 观测延迟（按 PD 子步数取整，零阶保持）
        self.obs_delay_steps = int(round(obs_delay_ms * pd_freq / 1000.0))
        self._obs_history: deque[np.ndarray] = deque(maxlen=self.obs_delay_steps + 1)

        # 终止/奖励参数
        self.fall_threshold = np.deg2rad(fall_threshold_deg)
        self.arm_threshold = np.deg2rad(arm_threshold_deg)
        self.max_episode_steps = int(max_episode_s * policy_freq)
        self.w_upright, self.w_angle = float(w_upright), float(w_angle)
        self.w_vel, self.w_torque = float(w_vel), float(w_torque)
        self.vel_norm = float(vel_norm)

        # MuJoCo：1ms 物理步长 = 1kHz PD
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        if abs(self.model.opt.timestep - 1.0 / pd_freq) > 1e-9:
            raise ValueError(
                f"模型 timestep({self.model.opt.timestep}) 与 pd_freq({pd_freq}) 不匹配"
            )
        self.data = mujoco.MjData(self.model)
        self._motor_id = self.model.actuator("motor").id
        self._hinge_qadr = self.model.jnt_qposadr[self.model.joint("hinge").id]
        self._hinge_vadr = self.model.jnt_dofadr[self.model.joint("hinge").id]

        # 策略动作 = 目标角度 θ_cmd ∈ [-π, π]；观测 = 4 维
        self.action_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self._steps = 0
        self._armed = False

    # ------------------------------------------------------------------ 状态
    @property
    def theta(self) -> float:
        return float(self.data.qpos[self._hinge_qadr])

    @property
    def theta_dot(self) -> float:
        return float(self.data.qvel[self._hinge_vadr])

    @property
    def theta_err(self) -> float:
        """相对倒立位的偏差（0 = 直立）。"""
        return wrap_to_pi(np.pi - self.theta)

    def _obs_now(self) -> np.ndarray:
        err = self.theta_err
        return np.array(
            [
                np.cos(self.theta),
                np.sin(self.theta),
                self.theta_dot / self.vel_norm,
                err / np.pi,
            ],
            dtype=np.float32,
        )

    def _obs_delayed(self) -> np.ndarray:
        """带 obs_delay_ms 延迟的观测（零阶保持：回放历史缓冲）。"""
        if self.obs_delay_steps <= 0 or len(self._obs_history) <= self.obs_delay_steps:
            # 历史不足时用最旧的一条（起始瞬间）
            return self._obs_history[0] if self._obs_history else self._obs_now()
        return self._obs_history[-1 - self.obs_delay_steps]

    # ------------------------------------------------------------------ 接口
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        mujoco.mj_resetData(self.model, self.data)
        # 初始状态：竖直下垂附近的小扰动（保证起摆任务的非平凡性）
        self.data.qpos[self._hinge_qadr] = self.np_random.normal(0.0, 0.1)
        self.data.qvel[self._hinge_vadr] = self.np_random.normal(0.0, 0.5)
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._armed = False
        self._obs_history.clear()
        obs = self._obs_now()
        self._obs_history.append(obs.copy())
        return obs, {"theta": self.theta, "theta_err": self.theta_err}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        theta_cmd = float(np.clip(np.asarray(action).reshape(-1)[0], -np.pi, np.pi))

        # ---- 1kHz PD 子步（decimation 个），能量/力矩沿子步累计 ----
        tau_sq_acc = 0.0
        cos_err_acc = 0.0
        err_sq_acc = 0.0
        vel_sq_acc = 0.0
        for _ in range(self.decimation):
            err = wrap_to_pi(theta_cmd - self.theta)
            tau = np.clip(
                self.kp * err - self.kd * self.theta_dot,
                -self.torque_limit,
                self.torque_limit,
            )
            self.data.ctrl[self._motor_id] = tau
            mujoco.mj_step(self.model, self.data)

            e = self.theta_err
            tau_sq_acc += (tau / self.torque_limit) ** 2
            cos_err_acc += np.cos(e)
            err_sq_acc += (e / np.pi) ** 2
            vel_sq_acc += (self.theta_dot / self.vel_norm) ** 2

        n = self.decimation
        self._steps += 1

        # ---- 奖励（子步平均）：直立项 − 角度偏差 − 速度惩罚 − 力矩惩罚 ----
        reward = (
            self.w_upright * (cos_err_acc / n)
            - self.w_angle * (err_sq_acc / n)
            - self.w_vel * (vel_sq_acc / n)
            - self.w_torque * (tau_sq_acc / n)
        )

        # ---- 终止（armed 逻辑，与 governor 包络语义一致）----
        if abs(self.theta_err) < self.arm_threshold:
            self._armed = True
        terminated = bool(self._armed and abs(self.theta_err) > self.fall_threshold)
        truncated = bool(self._steps >= self.max_episode_steps)

        # ---- 观测（先记账最新值，再按延迟回放）----
        self._obs_history.append(self._obs_now())
        obs = self._obs_delayed()

        info = {
            "theta": self.theta,
            "theta_err": self.theta_err,
            "theta_dot": self.theta_dot,
            "tau_last": float(self.data.ctrl[self._motor_id]),
            "armed": self._armed,
            "steps": self._steps,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:  # pragma: no cover
        pass
