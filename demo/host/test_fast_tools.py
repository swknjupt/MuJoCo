"""D3.1–D3.3 快速仿真工具链验收。

运行（仓库根目录）：
    .venv/bin/python -m demo.host.test_fast_tools
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from demo.host.fast_sim import FastSimulator
from demo.host.scan_faults import scan
from demo.host.tune_governor import tune_fallback, tune_thresholds


def main() -> None:
    simulator = FastSimulator()
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        report = scan(
            simulator,
            delays=[5, 30],
            dropout_rates=[0.5, 1.0],
            torque_scales=[1.0, 0.5],
            seeds=[0, 1],
            duration_s=4.0,
            output_dir=output_dir,
        )
        candidate = report["selected"]
        assert candidate["divergence_rate"] == 1.0
        assert (output_dir / "fault_scan_results.csv").exists()
        scenario_path = output_dir / "fault_scenario.json"
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        assert scenario["encoder_dropout_p"] == 1.0
        print("[D3.1] 小矩阵扫描与候选场景生成 ✔")

        from demo.host.fast_sim import FaultScenario

        fault = FaultScenario.from_dict(scenario)
        thresholds = tune_thresholds(
            simulator,
            fault,
            seeds=[0, 1],
            normal_duration_s=5.0,
            quick=True,
        )
        selected_thresholds = thresholds["selected"]
        assert not selected_thresholds["false_trigger"]
        assert selected_thresholds["all_faults_detected"]
        print("[D3.2] 零误触发 + 故障必触发阈值筛选 ✔")

        fallback = tune_fallback(
            simulator,
            fault,
            seeds=[0, 1],
            threshold_result=thresholds,
            quick=True,
        )
        selected_fallback = fallback["selected"]
        assert selected_fallback["valid"]
        assert selected_fallback["max_safe_time_s"] < 2.0
        assert selected_fallback["max_torque_nm"] <= 4.5
        print("[D3.3] SAFE<2s + fallback 力矩≤4.5N·m ✔")

    print("\n=== D3.1–D3.3 快速仿真工具链验收通过 ===")


if __name__ == "__main__":
    main()
