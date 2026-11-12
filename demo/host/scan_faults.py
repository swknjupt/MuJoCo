"""故障组合快速扫描与候选场景生成（Task D / D3.1）。

运行（仓库根目录）：
    .venv/bin/python demo/host/scan_faults.py
    .venv/bin/python demo/host/scan_faults.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.host.fast_sim import FaultScenario, FastSimulator

DEFAULT_OUTPUT = ROOT / "examples" / "output" / "fault_scan"


@dataclass(frozen=True)
class ConfigSummary:
    scenario: FaultScenario
    divergence_count: int
    seeds: int
    mean_divergence_s: float

    @property
    def divergence_rate(self) -> float:
        return self.divergence_count / self.seeds


def scan(
    simulator: FastSimulator,
    *,
    delays: list[int],
    dropout_rates: list[float],
    torque_scales: list[float],
    seeds: list[int],
    duration_s: float,
    output_dir: Path,
) -> dict[str, object]:
    """扫描参数矩阵并写出逐场景 CSV、候选 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fault_scan_results.csv"
    summaries: list[ConfigSummary] = []
    total = len(delays) * len(dropout_rates) * len(torque_scales)
    current = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "delay_ms",
                "encoder_dropout_p",
                "torque_scale",
                "seed",
                "diverged",
                "divergence_s",
                "max_abs_theta_dot",
            ),
        )
        writer.writeheader()
        for delay_ms in delays:
            for dropout_p in dropout_rates:
                for torque_scale in torque_scales:
                    current += 1
                    scenario = FaultScenario(
                        delay_ms=delay_ms,
                        encoder_dropout_p=dropout_p,
                        torque_scale=torque_scale,
                        fault_at_s=2.0,
                    )
                    results = [
                        simulator.run(
                            scenario,
                            seed=seed,
                            governor_enabled=False,
                            duration_s=duration_s,
                        )
                        for seed in seeds
                    ]
                    for result in results:
                        writer.writerow(
                            {
                                "delay_ms": delay_ms,
                                "encoder_dropout_p": dropout_p,
                                "torque_scale": torque_scale,
                                "seed": result.seed,
                                "diverged": result.diverged,
                                "divergence_s": result.divergence_s,
                                "max_abs_theta_dot": result.max_abs_theta_dot,
                            }
                        )
                    diverged = [result for result in results if result.diverged]
                    summaries.append(
                        ConfigSummary(
                            scenario=scenario,
                            divergence_count=len(diverged),
                            seeds=len(seeds),
                            mean_divergence_s=(
                                statistics.mean(r.divergence_s for r in diverged)
                                if diverged
                                else float("inf")
                            ),
                        )
                    )
                    print(
                        f"[{current:03d}/{total:03d}] "
                        f"delay={delay_ms:3d}ms p={dropout_p:.2f} "
                        f"tau={torque_scale:.2f} "
                        f"divergence={len(diverged)}/{len(seeds)}"
                    )

    reliable = [
        item for item in summaries if item.divergence_rate == 1.0
    ]
    if not reliable:
        raise RuntimeError("扫描矩阵中没有 100% 发散组合")
    selected = min(
        reliable,
        key=lambda item: (
            item.scenario.severity,
            item.mean_divergence_s,
        ),
    )

    candidate = selected.scenario.to_dict()
    candidate["divergence_rate"] = selected.divergence_rate
    candidate["mean_divergence_s"] = selected.mean_divergence_s
    candidate["seeds"] = seeds
    (output_dir / "fault_scenario.json").write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "passed": True,
        "scope": "fast synchronous simulation; final confirmation requires RT-SIL and real hardware",
        "matrix": {
            "delays_ms": delays,
            "encoder_dropout_rates": dropout_rates,
            "torque_scales": torque_scales,
            "seeds": seeds,
            "duration_s": duration_s,
        },
        "configs": total,
        "runs": total * len(seeds),
        "reliable_configs": len(reliable),
        "selected": candidate,
        "candidate_count": len(reliable),
        "results_csv": str(csv_path),
    }
    (output_dir / "fault_scan_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="故障组合快速扫描")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        delays = [5, 30, 100]
        dropout_rates = [0.0, 0.5, 1.0]
        torque_scales = [1.0, 0.5, 0.2]
        seeds = [0, 1]
    else:
        delays = [5, 10, 20, 30, 50, 100]
        dropout_rates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        torque_scales = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
        seeds = list(range(10))

    report = scan(
        FastSimulator(),
        delays=delays,
        dropout_rates=dropout_rates,
        torque_scales=torque_scales,
        seeds=seeds,
        duration_s=args.duration,
        output_dir=args.output_dir,
    )
    print(
        f"\n选定场景：{json.dumps(report['selected'], ensure_ascii=False)}\n"
        f"报告：{args.output_dir / 'fault_scan_report.json'}"
    )


if __name__ == "__main__":
    main()
