"""governor 阈值与 fallback 参数预标定（Task D / D3.2–D3.3）。

运行（仓库根目录）：
    .venv/bin/python demo/host/tune_governor.py --mode all
    .venv/bin/python demo/host/tune_governor.py --mode all --quick
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.host.fast_sim import (
    FaultScenario,
    FastSimulator,
    governor_params_from_dict,
    governor_params_to_dict,
)
from demo.rt.governor import GovernorParams

SCAN_DIR = ROOT / "examples" / "output" / "fault_scan"
DEFAULT_OUTPUT = ROOT / "examples" / "output" / "governor_tuning"


def _load_scenario(path: Path) -> FaultScenario:
    value = json.loads(path.read_text(encoding="utf-8"))
    return FaultScenario.from_dict(value)


def tune_thresholds(
    simulator: FastSimulator,
    scenario: FaultScenario,
    *,
    seeds: list[int],
    normal_duration_s: float,
    quick: bool,
) -> dict[str, object]:
    fall_degrees = [45.0, 60.0, 75.0]
    theta_dot_limits = [12.0, 16.0, 20.0, 24.0]
    if quick:
        fall_degrees = [60.0, 75.0]
        theta_dot_limits = [16.0, 20.0]

    candidates: list[dict[str, object]] = []
    for fall_deg in fall_degrees:
        for theta_dot_max in theta_dot_limits:
            params = replace(
                GovernorParams(),
                fall_threshold=math.radians(fall_deg),
                theta_dot_max=theta_dot_max,
            )
            normal = simulator.run(
                FaultScenario(),
                seed=777,
                governor_enabled=True,
                governor_params=params,
                duration_s=normal_duration_s,
                start_near_upright=True,
            )
            results = [
                simulator.run(
                    scenario,
                    seed=seed,
                    governor_enabled=True,
                    governor_params=params,
                    duration_s=6.0,
                    start_near_upright=True,
                )
                for seed in seeds
            ]
            all_detected = all(result.takeover_s is not None for result in results)
            all_safe = all(
                result.takeover_s is not None
                and result.safe_s is not None
                and result.takeover_to_safe_s is not None
                and result.takeover_to_safe_s < 2.0
                and result.max_abs_torque_nm <= 5.0
                for result in results
            )
            false_trigger = normal.takeover_s is not None
            mean_takeover = (
                statistics.mean(result.takeover_s for result in results)
                if all(result.takeover_s is not None for result in results)
                else float("inf")
            )
            candidates.append(
                {
                    "fall_threshold_deg": fall_deg,
                    "theta_dot_max": theta_dot_max,
                    "false_trigger": false_trigger,
                    "all_faults_detected": all_detected,
                    "all_faults_safe": all_safe,
                    "mean_takeover_s": mean_takeover,
                    "params": governor_params_to_dict(params),
                }
            )
            print(
                f"[threshold] fall={fall_deg:.0f}° "
                f"theta_dot<{theta_dot_max:.0f} "
                f"false={false_trigger} detected={all_detected} safe={all_safe}"
            )

    valid = [
        item
        for item in candidates
        if not item["false_trigger"] and item["all_faults_detected"]
    ]
    if not valid:
        raise RuntimeError("没有同时满足 30min 零误触发和故障必触发的阈值组合")
    selected = min(valid, key=lambda item: float(item["mean_takeover_s"]))
    return {
        "passed": True,
        "scope": "fast simulation pre-tuning; real hardware recalibration remains required",
        "normal_duration_s": normal_duration_s,
        "candidates": candidates,
        "selected": selected,
    }


def tune_fallback(
    simulator: FastSimulator,
    scenario: FaultScenario,
    *,
    seeds: list[int],
    threshold_result: dict[str, object],
    quick: bool,
) -> dict[str, object]:
    selected_thresholds = threshold_result["selected"]
    base = governor_params_from_dict(selected_thresholds["params"])
    k_energy_values = [0.4, 0.8, 1.2, 1.6]
    c_visc_values = [0.2, 0.4, 0.6]
    kp_hang_values = [6.0, 9.0, 12.0, 15.0, 18.0]
    e_safe_values = [0.05, 0.10, 0.20, 0.30]
    if quick:
        k_energy_values = [0.8, 1.2]
        c_visc_values = [0.6]
        kp_hang_values = [15.0]
        e_safe_values = [0.30]

    candidates: list[dict[str, object]] = []
    for k_energy in k_energy_values:
        for c_visc_min in c_visc_values:
            for kp_hang in kp_hang_values:
                for e_safe in e_safe_values:
                    params = replace(
                        base,
                        k_energy=k_energy,
                        c_visc_min=c_visc_min,
                        kp_hang=kp_hang,
                        e_safe_threshold=e_safe,
                    )
                    results = [
                        simulator.run(
                            scenario,
                            seed=seed,
                            governor_enabled=True,
                            governor_params=params,
                            duration_s=6.0,
                            start_near_upright=True,
                        )
                        for seed in seeds
                    ]
                    valid = all(
                        result.takeover_to_safe_s is not None
                        and result.takeover_to_safe_s < 2.0
                        and result.max_fallback_torque_nm <= 4.5
                        for result in results
                    )
                    safe_times = [
                        result.takeover_to_safe_s
                        for result in results
                        if result.takeover_to_safe_s is not None
                    ]
                    candidates.append(
                        {
                            "k_energy": k_energy,
                            "c_visc_min": c_visc_min,
                            "kp_hang": kp_hang,
                            "e_safe_threshold": e_safe,
                            "valid": valid,
                            "max_safe_time_s": (
                                max(safe_times) if safe_times else float("inf")
                            ),
                            "max_torque_nm": max(
                                result.max_fallback_torque_nm for result in results
                            ),
                            "params": governor_params_to_dict(params),
                        }
                    )
                    print(
                        f"[fallback] k={k_energy:.1f} c={c_visc_min:.1f} "
                        f"kp={kp_hang:.0f} e={e_safe:.2f} valid={valid}"
                    )

    valid_candidates = [item for item in candidates if item["valid"]]
    if not valid_candidates:
        raise RuntimeError("没有满足 SAFE<2s 与力矩≤4.5N·m 的 fallback 组合")
    selected = min(
        valid_candidates,
        key=lambda item: (
            float(item["max_safe_time_s"]),
            float(item["max_torque_nm"]),
        ),
    )
    return {
        "passed": True,
        "scope": "fast simulation pre-tuning; real hardware recalibration remains required",
        "candidates": candidates,
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="governor 仿真预标定")
    parser.add_argument("--scenario", type=Path, default=SCAN_DIR / "fault_scenario.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("threshold", "fallback", "all"), default="all")
    parser.add_argument("--normal-duration", type=float, default=1800.0)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    scenario = _load_scenario(args.scenario)
    seeds = list(range(args.seed_count))
    simulator = FastSimulator()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    threshold_result: dict[str, object] | None = None
    if args.mode in ("threshold", "all"):
        threshold_result = tune_thresholds(
            simulator,
            scenario,
            seeds=seeds,
            normal_duration_s=args.normal_duration,
            quick=args.quick,
        )
        (args.output_dir / "threshold_tuning.json").write_text(
            json.dumps(threshold_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "governor_thresholds.json").write_text(
            json.dumps(threshold_result["selected"]["params"], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    if args.mode in ("fallback", "all"):
        if threshold_result is None:
            threshold_path = args.output_dir / "threshold_tuning.json"
            threshold_result = json.loads(threshold_path.read_text(encoding="utf-8"))
        fallback_result = tune_fallback(
            simulator,
            scenario,
            seeds=seeds,
            threshold_result=threshold_result,
            quick=args.quick,
        )
        (args.output_dir / "fallback_tuning.json").write_text(
            json.dumps(fallback_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "governor_params.json").write_text(
            json.dumps(fallback_result["selected"]["params"], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    print(f"\n输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
