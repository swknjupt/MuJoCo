"""单摆策略的确定性验收评估（Task D / D1.2）。

验收口径：10 个固定种子全部完成起摆，并在首次进入直立区后连续保持
至少 30s。直立区默认定义为相对倒立位偏差小于 30°，与 env 的 armed
阈值一致；评估单局上限延长到 40s，避免训练时的 10s 截断掩盖长期稳定性。

运行（仓库根目录）：
    .venv/bin/python -m demo.envs.pendulum_eval
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from demo.envs.pendulum import PendulumEnv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "examples" / "output" / "ppo_pendulum_demo.zip"
DEFAULT_REPORT = ROOT / "examples" / "output" / "ppo_pendulum_eval.json"


@dataclass(frozen=True)
class EpisodeResult:
    """单个评估回合的结果。"""

    seed: int
    reached_upright: bool
    swing_time_s: float | None
    max_hold_s: float
    final_abs_error_deg: float
    success: bool


def evaluate_policy(
    model: PPO,
    episodes: int = 10,
    seed_start: int = 1000,
    upright_threshold_deg: float = 30.0,
    hold_s: float = 30.0,
    max_episode_s: float = 40.0,
    verbose: bool = True,
) -> tuple[bool, list[EpisodeResult]]:
    """评估策略是否完成起摆并长期保持直立。"""
    env = PendulumEnv(max_episode_s=max_episode_s)
    threshold_rad = np.deg2rad(upright_threshold_deg)
    results: list[EpisodeResult] = []

    try:
        for episode in range(episodes):
            seed = seed_start + episode
            obs, info = env.reset(seed=seed)
            elapsed_s = 0.0
            reached_upright = False
            swing_time_s: float | None = None
            hold_start_s: float | None = None
            max_hold_s = 0.0
            success = False

            while True:
                abs_error = abs(float(info["theta_err"]))
                if abs_error < threshold_rad:
                    if not reached_upright:
                        reached_upright = True
                        swing_time_s = elapsed_s
                    if hold_start_s is None:
                        hold_start_s = elapsed_s
                    max_hold_s = max(max_hold_s, elapsed_s - hold_start_s)
                    if max_hold_s >= hold_s:
                        success = True
                        break
                else:
                    hold_start_s = None

                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                elapsed_s += env.dt_policy

                if terminated or truncated:
                    abs_error = abs(float(info["theta_err"]))
                    if abs_error < threshold_rad:
                        if not reached_upright:
                            reached_upright = True
                            swing_time_s = elapsed_s
                        if hold_start_s is None:
                            hold_start_s = elapsed_s
                        max_hold_s = max(max_hold_s, elapsed_s - hold_start_s)
                        success = max_hold_s >= hold_s
                    break

            result = EpisodeResult(
                seed=seed,
                reached_upright=reached_upright,
                swing_time_s=swing_time_s,
                max_hold_s=max_hold_s,
                final_abs_error_deg=float(np.rad2deg(abs(info["theta_err"]))),
                success=success,
            )
            results.append(result)
            if verbose:
                swing_text = (
                    f"{result.swing_time_s:.2f}s"
                    if result.swing_time_s is not None
                    else "未起摆"
                )
                print(
                    f"[评估 {episode + 1:02d}/{episodes:02d}] seed={seed} "
                    f"起摆={swing_text} 最长保持={result.max_hold_s:.2f}s "
                    f"末态误差={result.final_abs_error_deg:.2f}° "
                    f"{'✔' if result.success else '✘'}"
                )
    finally:
        env.close()

    return all(result.success for result in results), results


def save_report(
    model_path: Path,
    report_path: Path,
    passed: bool,
    results: list[EpisodeResult],
) -> None:
    """保存机器可读的验收报告。"""
    report = {
        "model": str(model_path),
        "passed": passed,
        "episodes": len(results),
        "success_count": sum(result.success for result in results),
        "criteria": {
            "all_episodes_success": True,
            "upright_threshold_deg": 30.0,
            "continuous_hold_s": 30.0,
            "max_episode_s": 40.0,
        },
        "results": [asdict(result) for result in results],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="评估单摆 PPO 策略的起摆与长期保持能力")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    passed, results = evaluate_policy(model, episodes=args.episodes)
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
