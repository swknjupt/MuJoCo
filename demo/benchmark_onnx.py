"""目标机独立的 ONNX CPU 延迟基准（Task D / D1.3）。

只依赖 numpy 与 onnxruntime，不加载 PyTorch、Stable-Baselines3 或 MuJoCo。
如果提供开发机生成的 reference NPZ，则同时验证目标机输出数值一致。

运行（仓库根目录）：
    .venv/bin/python demo/benchmark_onnx.py
    python3 demo/benchmark_onnx.py --reference examples/output/policy_onnx_reference.npz
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT / "examples" / "output" / "policy.onnx"
DEFAULT_REFERENCE = ROOT / "examples" / "output" / "policy_onnx_reference.npz"
DEFAULT_REPORT = ROOT / "examples" / "output" / "policy_latency_target.json"
LATENCY_LIMIT_MS = 10.0


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


def main() -> None:
    parser = argparse.ArgumentParser(description="目标机 ONNX CPU 延迟基准")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    if not args.onnx.exists():
        parser.error(f"ONNX 不存在：{args.onnx}")
    if args.samples < 1:
        parser.error("--samples 必须 >= 1")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(args.onnx),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    parity: dict[str, object] = {"provided": False, "passed": None}
    if args.reference.exists():
        with np.load(args.reference) as reference:
            observations = reference["observation"].astype(np.float32, copy=False)
            expected = reference["torch_action"].astype(np.float32, copy=False)
        actual = np.concatenate(
            [
                session.run(None, {input_name: observation[None, :]})[0]
                for observation in observations
            ],
            axis=0,
        )
        max_abs_diff = float(np.max(np.abs(expected - actual)))
        parity_passed = bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-6))
        parity = {
            "provided": True,
            "passed": parity_passed,
            "samples": int(observations.shape[0]),
            "rtol": 1e-5,
            "atol": 1e-6,
            "max_abs_diff": max_abs_diff,
        }
        single_observation = observations[0:1]
    else:
        single_observation = np.zeros((1, 4), dtype=np.float32)

    for _ in range(args.warmup):
        session.run(None, {input_name: single_observation})

    latencies_ns = np.empty(args.samples, dtype=np.int64)
    for index in range(args.samples):
        t0 = time.perf_counter_ns()
        session.run(None, {input_name: single_observation})
        latencies_ns[index] = time.perf_counter_ns() - t0

    stats = percentile_summary(latencies_ns / 1_000_000.0)
    latency_passed = stats["p99_ms"] < LATENCY_LIMIT_MS
    passed = latency_passed and parity["passed"] is not False
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
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
            "cpu_count": os.cpu_count(),
        },
        "artifacts": {
            "onnx": str(args.onnx),
            "reference": str(args.reference),
        },
        "parity": parity,
        "latency": stats,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if parity["provided"]:
        print(
            f"目标机数值对齐：{'✔' if parity['passed'] else '✘'} "
            f"max_abs_diff={parity['max_abs_diff']:.3e}"
        )
    else:
        print("警告：未提供 reference NPZ，仅测延迟，数值对齐待补")
    print(
        f"延迟（{args.samples} 次，单线程 CPU）：mean={stats['mean_ms']:.3f}ms "
        f"P50={stats['p50_ms']:.3f}ms P95={stats['p95_ms']:.3f}ms "
        f"P99={stats['p99_ms']:.3f}ms Max={stats['max_ms']:.3f}ms"
    )
    print(f"报告：{args.report}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
