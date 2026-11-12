"""候选场景与 Governor 参数加载验收。

运行（仓库根目录）：
    .venv/bin/python -m demo.rt.test_run_rt_sil_config
"""

from __future__ import annotations

import math
from pathlib import Path

from demo.rt.run_rt_sil import load_governor_config, select_scenarios

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    scenario_file = ROOT / "examples" / "output" / "fault_scan" / "fault_scenario.json"
    governor_file = (
        ROOT / "examples" / "output" / "governor_tuning" / "governor_params.json"
    )
    scenarios = select_scenarios([3, 4], scenario_file)
    assert [item.act for item in scenarios] == [3, 4]
    assert scenarios[0].extreme_injection is not None
    assert scenarios[0].extreme_injection.encoder_dropout_p == 1.0
    assert scenarios[1].extreme_injection == scenarios[0].extreme_injection

    params = load_governor_config(governor_file)
    assert math.isclose(params.fall_threshold, math.radians(45.0))
    assert params.theta_dot_max == 12.0
    assert params.kp_hang == 15.0
    assert params.e_safe_threshold == 0.3

    stage1_file = (
        ROOT / "demo" / "configs" / "stage1_brake_governor.json"
    )
    stage1 = load_governor_config(stage1_file)
    assert stage1.theta_dot_max == 3.5
    assert math.isclose(stage1.fall_threshold, math.radians(45.0))
    assert stage1.kp_hang == 15.0
    print("候选场景与调参 Governor 配置加载 ✔")


if __name__ == "__main__":
    main()
