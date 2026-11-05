"""最小 RL 训练示例：用 PPO 在 MuJoCo 环境 InvertedPendulum-v5 上训练倒立摆。

运行: .venv/bin/python examples/05_train_rl_ppo.py
"""

import time

import gymnasium as gym
from stable_baselines3 import PPO

TOTAL_TIMESTEPS = 50_000


def main() -> None:
    print(f"Gymnasium 版本: {gym.__version__}")

    env = gym.make("InvertedPendulum-v5")
    print(f"环境: InvertedPendulum-v5 | "
          f"观测维度: {env.observation_space.shape} | 动作维度: {env.action_space.shape}")

    model = PPO("MlpPolicy", env, verbose=0, device="auto", seed=0)
    print(f"SB3 训练设备: {model.device} (auto = 有 GPU 用 GPU, 否则 CPU)")

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    dt = time.time() - t0
    print(f"训练完成: {TOTAL_TIMESTEPS:,} 步用时 {dt:.1f}s "
          f"({TOTAL_TIMESTEPS / dt:,.0f} 步/秒)")

    # 用训练好的策略评估 10 局
    rewards = []
    for i in range(10):
        obs, _ = env.reset(seed=42 + i)
        ep_rew, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rew += float(reward)
            done = terminated or truncated
        rewards.append(ep_rew)
    avg = sum(rewards) / len(rewards)
    print(f"评估 10 局平均回报: {avg:.1f} (上限 1000, 立杆坚持越久回报越高)")

    path = "examples/output/ppo_inverted_pendulum"
    model.save(path)
    print(f"模型已保存: {path}.zip")
    print(f"\n{'✓ 训练链路验证通过 (MuJoCo → Gymnasium → SB3 → PyTorch)' if avg > 500 else '⚠ 回报偏低, 训练链路可用但需更多步数'}")


if __name__ == "__main__":
    main()
