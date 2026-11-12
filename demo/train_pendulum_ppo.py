"""训练并验收单摆起摆 PPO 策略（Task D / D1.2）。

环境保持部署同构：策略 100Hz 输出目标角，env 内部 1kHz PD 执行。
训练产物保存到 examples/output/，支持从已有 checkpoint 继续训练。

运行（仓库根目录）：
    .venv/bin/python demo/train_pendulum_ppo.py --timesteps 500000
    .venv/bin/python demo/train_pendulum_ppo.py --timesteps 250000 --resume
    .venv/bin/python -m demo.envs.pendulum_eval
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 多进程采样优先使用 CPU 核；限制 BLAS 线程，避免小网络训练时线程争抢。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from demo.envs.pendulum import PendulumEnv
from demo.envs.pendulum_eval import DEFAULT_REPORT, evaluate_policy, save_report

DEFAULT_MODEL = ROOT / "examples" / "output" / "ppo_pendulum_demo.zip"
DEFAULT_TENSORBOARD = ROOT / "examples" / "output" / "tensorboard"


def _make_env(seed: int) -> PendulumEnv:
    """构造单个训练环境。"""
    return PendulumEnv()


def build_vec_env(n_envs: int, seed: int):
    """构造并并行化训练环境。"""
    env_fns = [(lambda rank=rank: _make_env(seed + rank)) for rank in range(n_envs)]
    # Python 3.14 默认 forkserver；显式 fork 可兼容无本地套接字权限的运行环境。
    vec_env = (
        DummyVecEnv(env_fns)
        if n_envs == 1
        else SubprocVecEnv(env_fns, start_method="fork")
    )
    vec_env.seed(seed)
    return vec_env


def make_model(env, seed: int, device: str) -> PPO:
    """创建适合小规模单摆任务的 PPO 配置。"""
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [64, 64]},
        tensorboard_log=str(DEFAULT_TENSORBOARD),
        verbose=0,
        device=device,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="训练单摆起摆 PPO 策略")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if args.n_envs < 1:
        parser.error("--n-envs 必须 >= 1")

    if args.eval_only:
        model = PPO.load(args.model, device=args.device)
        passed, results = evaluate_policy(model)
        save_report(args.model, args.report, passed, results)
        if not passed:
            raise SystemExit(1)
        return

    if args.resume and not args.model.exists():
        parser.error(f"--resume 指定的模型不存在：{args.model}")

    args.model.parent.mkdir(parents=True, exist_ok=True)
    vec_env = build_vec_env(args.n_envs, args.seed)
    try:
        if args.resume:
            model = PPO.load(
                args.model,
                env=vec_env,
                device=args.device,
                tensorboard_log=str(DEFAULT_TENSORBOARD),
            )
            print(f"从 checkpoint 继续训练：{args.model}")
        else:
            model = make_model(vec_env, args.seed, args.device)

        print(
            f"训练设备：{model.device} | 并行环境：{args.n_envs} | "
            f"新增步数：{args.timesteps:,}"
        )
        t0 = time.perf_counter()
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=not args.resume,
            progress_bar=False,
        )
        elapsed = time.perf_counter() - t0
        model.save(args.model)
        print(
            f"训练完成：{args.timesteps:,} 步，用时 {elapsed:.1f}s "
            f"({args.timesteps / elapsed:,.0f} 步/s)"
        )
        print(f"模型：{args.model}")
    finally:
        vec_env.close()

    if not args.skip_eval:
        model = PPO.load(args.model, device=args.device)
        passed, results = evaluate_policy(model)
        save_report(args.model, args.report, passed, results)
        print(
            f"\nD1.2 验收：{'通过 ✔' if passed else '未通过 ✘'} "
            f"({sum(result.success for result in results)}/{len(results)})"
        )
        print(f"报告：{args.report}")
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
