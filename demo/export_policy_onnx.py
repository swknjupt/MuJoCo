"""导出单摆 PPO actor 为 ONNX 并测量 CPU 推理延迟（Task D / D1.3）。

只导出部署需要的 actor 路径，不包含 critic 和训练相关算子。导出后用
onnxruntime CPUExecutionProvider 做数值对齐，再执行 1000 次单样本推理，
报告 P50/P95/P99/Max。D1.3 的目标机是 RT 盒 CPU；在开发机运行只能作为
功能验证，不能替代目标机验收。

运行（仓库根目录）：
    .venv/bin/python demo/export_policy_onnx.py
    .venv/bin/python demo/export_policy_onnx.py --model examples/output/ppo_pendulum_demo.zip
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from stable_baselines3 import PPO

from demo.envs.pendulum import PendulumEnv

DEFAULT_MODEL = ROOT / "examples" / "output" / "ppo_pendulum_demo.zip"
DEFAULT_ONNX = ROOT / "examples" / "output" / "policy.onnx"
DEFAULT_REPORT = ROOT / "examples" / "output" / "policy_latency.json"
DEFAULT_REFERENCE = ROOT / "examples" / "output" / "policy_onnx_reference.npz"
LATENCY_LIMIT_MS = 10.0


class ActorOnly(nn.Module):
    """与 SB3 deterministic predict 等价的 actor 前向路径。"""

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(
            observation, self.policy.pi_features_extractor
        )
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent_pi)


def collect_validation_observations(env: PendulumEnv, count: int = 256) -> np.ndarray:
    """从悬垂、随机动作和近直立状态收集代表性观测。"""
    observations: list[np.ndarray] = []
    obs, _ = env.reset(seed=20261109)
    observations.append(np.asarray(obs, dtype=np.float32))

    while len(observations) < count:
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(np.asarray(obs, dtype=np.float32))
        if terminated or truncated:
            obs, _ = env.reset(seed=20261109 + len(observations))

    # 补齐接近倒立位的典型状态，避免只覆盖起摆阶段。
    angles = np.linspace(np.pi - 0.5, np.pi + 0.5, count // 4, dtype=np.float32)
    velocities = np.linspace(-2.0, 2.0, count // 4, dtype=np.float32)
    for angle, velocity in zip(angles, velocities, strict=True):
        error = (np.pi - angle + np.pi) % (2.0 * np.pi) - np.pi
        observations.append(
            np.array(
                [np.cos(angle), np.sin(angle), velocity / 10.0, error / np.pi],
                dtype=np.float32,
            )
        )
    return np.stack(observations)


def percentile_summary(latencies_ms: np.ndarray) -> dict[str, float]:
    """计算延迟统计量。"""
    return {
        "samples": int(latencies_ms.size),
        "mean_ms": float(np.mean(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "max_ms": float(np.max(latencies_ms)),
    }


def export_actor(model_path: Path, onnx_path: Path) -> tuple[PPO, ActorOnly]:
    """加载 checkpoint 并导出确定性 actor。"""
    model = PPO.load(model_path, device="cpu")
    model.policy.set_training_mode(False)
    actor = ActorOnly(model.policy).cpu().eval()
    dummy_observation = torch.zeros(1, 4, dtype=torch.float32, device="cpu")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        dummy_observation,
        onnx_path,
        input_names=["observation"],
        output_names=["action"],
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    return model, actor


def benchmark(
    model_path: Path,
    onnx_path: Path,
    report_path: Path,
    samples: int,
    warmup: int,
) -> bool:
    """导出、校验并测量 actor ONNX 的单样本 CPU 推理延迟。"""
    model, actor = export_actor(model_path, onnx_path)
    env = PendulumEnv()
    try:
        validation_obs = collect_validation_observations(env)
    finally:
        env.close()

    with torch.inference_mode():
        torch_output = actor(torch.from_numpy(validation_obs)).numpy()
    reference_path = report_path.parent / "policy_onnx_reference.npz"
    np.savez_compressed(
        reference_path,
        observation=validation_obs,
        torch_action=torch_output,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    onnx_output = np.concatenate(
        [
            session.run(None, {input_name: observation[None, :]})[0]
            for observation in validation_obs
        ],
        axis=0,
    )
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    parity_ok = bool(np.allclose(torch_output, onnx_output, rtol=1e-5, atol=1e-6))

    single_obs = validation_obs[0:1]
    for _ in range(warmup):
        session.run(None, {input_name: single_obs})

    latencies_ns = np.empty(samples, dtype=np.int64)
    for index in range(samples):
        t0 = time.perf_counter_ns()
        session.run(None, {input_name: single_obs})
        latencies_ns[index] = time.perf_counter_ns() - t0

    stats = percentile_summary(latencies_ns / 1_000_000.0)
    passed = parity_ok and stats["p99_ms"] < LATENCY_LIMIT_MS
    report = {
        "passed": passed,
        "target": {
            "metric": "onnxruntime single-sample p99_ms",
            "limit_ms": LATENCY_LIMIT_MS,
            "required_hardware": "RT 盒 CPU",
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
            "cpu_count": os.cpu_count(),
        },
        "artifacts": {
            "checkpoint": str(model_path),
            "onnx": str(onnx_path),
            "reference": str(reference_path),
        },
        "parity": {
            "samples": int(validation_obs.shape[0]),
            "rtol": 1e-5,
            "atol": 1e-6,
            "max_abs_diff": max_abs_diff,
            "passed": parity_ok,
        },
        "latency": stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"ONNX 数值对齐：max_abs_diff={max_abs_diff:.3e} "
        f"{'✔' if parity_ok else '✘'}"
    )
    print(
        f"延迟（{samples} 次，单线程 CPU）：mean={stats['mean_ms']:.3f}ms "
        f"P50={stats['p50_ms']:.3f}ms P95={stats['p95_ms']:.3f}ms "
        f"P99={stats['p99_ms']:.3f}ms Max={stats['max_ms']:.3f}ms"
    )
    print(f"ONNX：{onnx_path}")
    print(f"报告：{report_path}")
    if not passed:
        print("D1.3 未通过：数值对齐失败或 P99 >= 10ms")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="导出单摆 PPO actor 并测 ONNX 延迟")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples 必须 >= 1")
    if not args.model.exists():
        parser.error(f"模型不存在：{args.model}")
    passed = benchmark(
        args.model,
        args.onnx,
        args.report,
        samples=args.samples,
        warmup=args.warmup,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
