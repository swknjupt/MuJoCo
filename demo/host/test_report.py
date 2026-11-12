"""D3.6 验收：事件识别、时间窗和单文件 HTML 报告。

运行（仓库根目录）：
    .venv/bin/python -m demo.host.test_report
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from demo.host.replay import DEFAULT_RUN_DIR, get_act, load_run, select_window
from demo.host.report import NUMERIC_SIGNALS, build_html


def main() -> None:
    run = load_run(DEFAULT_RUN_DIR)
    assert len(run.acts) == 4
    assert run.passed and run.target_evidence

    for act in run.acts:
        assert len(act.samples) in (4000, 6000)
        assert act.samples[0].timestamp_us == 0
        if act.act == 1:
            assert not act.events
        else:
            assert len(act.events) >= 1
        print(
            f"[幕 {act.act}] {len(act.samples)} 行 / "
            f"{len(act.events)} 个事件 ✔"
        )

    act4 = get_act(run, 4)
    kinds = {event.kind for event in act4.events}
    assert {"injection", "violation", "takeover", "safe"} <= kinds
    window = select_window(act4.samples, 2.0, 3.5)
    assert 1499 <= len(window) <= 1501

    report = build_html(run)
    assert report.startswith("<!doctype html>")
    assert "第 1 幕" in report and "第 4 幕" in report
    for field, _ in NUMERIC_SIGNALS:
        assert f'data-field="{field}"' in report
    assert "injected_fault_id" in report
    assert "governor_state" in report
    assert "T14 PREEMPT_RT 目标机证据" in report
    assert report.count('class="act-panel"') == 4
    assert "<script src=" not in report
    assert "<link " not in report
    assert "https://" not in report

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "report.html"
        output.write_text(report, encoding="utf-8")
        assert output.stat().st_size > 100_000
        print(f"[HTML] {output.stat().st_size:,} bytes ✔")

    print("\n=== D3.6 报告与回放验收通过 ===")


if __name__ == "__main__":
    main()
