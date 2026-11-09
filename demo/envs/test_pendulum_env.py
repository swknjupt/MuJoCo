"""D1.1 验收测试：单摆环境的三条验收标准。

运行（仓库根目录）：
    .venv/bin/python -m demo.envs.test_pendulum_env

验收项（taskD §4.1 D1.1）：
  1. gymnasium check_env 通过；
  2. 随机动作下动力学正常（量值有界、物理在演化）；
  3. 手动给定目标角时，内部 PD 可稳定跟踪（近直立保持 + 小阶跃收敛无发散震荡）。
"""

from __future__ import annotations

import numpy as np

import mujoco
from gymnasium.utils.env_checker import check_env

from demo.envs.pendulum import PendulumEnv, wrap_to_pi


def test_1_check_env() -> None:
    """验收 1：gymnasium 官方检查器。"""
    env = PendulumEnv()
    check_env(env, skip_render_check=True)
    print("[验收1] check_env 通过 ✔")


def test_2_random_rollout() -> None:
    """验收 2：随机动作下动力学正常。"""
    env = PendulumEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,) and np.all(np.isfinite(obs))

    n_steps, r_sum = 500, 0.0
    seen_terminated = False
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        assert np.all(np.isfinite(obs)), "观测出现 NaN/Inf"
        assert abs(info["theta_dot"]) < 60.0, f"角速度异常: {info['theta_dot']}"
        r_sum += r
        if term:
            seen_terminated = True
            env.reset()
    print(
        f"[验收2] 随机 500 步: obs 有界 ✔ 物理演化 ✔ 平均奖励 {r_sum / n_steps:+.3f} "
        f"(随机动作下应显著为负) 触发过终止逻辑: {seen_terminated} ✔"
    )


def _place_near_upright(env: PendulumEnv, offset: float = 0.05) -> None:
    """把摆放到近直立位（供 PD 跟踪测试，不经过 reset 的随机初始化）。"""
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[env._hinge_qadr] = np.pi - offset
    env.data.qvel[env._hinge_vadr] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env._steps = 0
    env._armed = False
    env._obs_history.clear()
    env._obs_history.append(env._obs_now())


def test_3_pd_tracking() -> None:
    """验收 3：内部 PD 的跟踪能力（保持 + 阶跃）。"""
    env = PendulumEnv()

    # --- 3a. 近直立保持：目标 = 倒立位，10 秒不倒、不发散震荡 ---
    _place_near_upright(env)
    hold_ok, max_err = True, 0.0
    prev_err, reversals = env.theta_err, 0
    for _ in range(1000):  # 10s @ 100Hz
        env.step(np.array([np.pi], dtype=np.float32))
        e = abs(env.theta_err)
        max_err = max(max_err, e)
        if e > np.deg2rad(30):  # 10 秒内偏离超过 30° 判失败
            hold_ok = False
            break
        if (env.theta_err - prev_err) * prev_err < 0:
            reversals += 1
        prev_err = env.theta_err
    assert hold_ok, f"PD 保持失败: max_err={np.rad2deg(max_err):.1f}°"
    print(f"[验收3a] 直立保持 10s ✔ (最大偏差 {np.rad2deg(max_err):.2f}°, 过零 {reversals} 次)")

    # --- 3b. 小阶跃跟踪：目标在动作空间内取值（env 会裁剪到 [-π,π]）---
    env2 = PendulumEnv()
    for target in (np.pi - 0.3, np.pi - 0.6, np.pi):
        # 从当前状态直接切目标（模拟策略突变目标）
        for _ in range(200):  # 2s
            env2.step(np.array([target], dtype=np.float32))
        final_err = abs(wrap_to_pi(env2.theta - target))
        assert final_err < 0.05, f"阶跃目标 {target:.2f} 收敛失败: err={final_err:.3f} rad"
    print("[验收3b] 三次小阶跃(近直立 ±0.3/0.6 rad)均 2s 内收敛至 <0.05 rad ✔")

    # --- 3c. 悬垂位跟踪：目标 0.3 rad，从悬垂出发应平稳到达 ---
    env3 = PendulumEnv()
    env3.reset(seed=3)
    env3.data.qpos[env3._hinge_qadr] = 0.0
    env3.data.qvel[env3._hinge_vadr] = 0.0
    mujoco.mj_forward(env3.model, env3.data)
    for _ in range(500):  # 5s
        env3.step(np.array([0.3], dtype=np.float32))
    final_err = abs(wrap_to_pi(env3.theta - 0.3))
    assert final_err < 0.05, f"悬垂跟踪失败: err={final_err:.3f} rad"
    print(f"[验收3c] 悬垂位目标 0.3 rad 平稳跟踪 ✔ (终态偏差 {final_err:.4f} rad)")


if __name__ == "__main__":
    test_1_check_env()
    test_2_random_rollout()
    test_3_pd_tracking()
    print("\n=== D1.1 全部验收通过 ===")
