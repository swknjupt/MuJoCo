"""RT-SIL 单摆 Plant（Task D / D1.8）。

直接包装 MuJoCo 模型，按 1kHz 显式步进，供 RT-SIL 集成脚本从外部写入
力矩并读取编码器等价状态。该类不包含策略、governor 或故障注入。
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "demo" / "models" / "pendulum_env.xml"


def wrap_to_pi(angle: float) -> float:
    """把角度归约到 (-π, π]。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PendulumPlant:
    """无限位单摆的实时可调用仿真对象。"""

    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.dt_s = float(self.model.opt.timestep)
        self._motor_id = self.model.actuator("motor").id
        self._hinge_qadr = self.model.jnt_qposadr[self.model.joint("hinge").id]
        self._hinge_vadr = self.model.jnt_dofadr[self.model.joint("hinge").id]

    def reset(self, seed: int, *, near_upright: bool = False) -> None:
        """重置到悬垂或近直立位的小扰动状态。"""
        rng = random.Random(seed)
        mujoco.mj_resetData(self.model, self.data)
        center = math.pi if near_upright else 0.0
        self.data.qpos[self._hinge_qadr] = center + rng.normalvariate(0.0, 0.1)
        self.data.qvel[self._hinge_vadr] = rng.normalvariate(0.0, 0.5)
        mujoco.mj_forward(self.model, self.data)

    @property
    def theta(self) -> float:
        return float(self.data.qpos[self._hinge_qadr])

    @property
    def theta_dot(self) -> float:
        return float(self.data.qvel[self._hinge_vadr])

    @property
    def theta_error(self) -> float:
        return wrap_to_pi(math.pi - self.theta)

    def observation_from_state(self, theta: float, theta_dot: float) -> np.ndarray:
        """使用（可能被注入污染的）传感器状态构造策略观测。"""
        error = wrap_to_pi(math.pi - theta)
        return np.array(
            [math.cos(theta), math.sin(theta), theta_dot / 10.0, error / math.pi],
            dtype=np.float32,
        )

    def step(self, torque_cmd: float) -> None:
        """执行一个 1ms 物理步。"""
        self.data.ctrl[self._motor_id] = float(torque_cmd)
        mujoco.mj_step(self.model, self.data)
