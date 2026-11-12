"""RT-SIL 九信号数据加载、事件识别与时间窗回放（Task D / D3.6）。

运行（仓库根目录）：
    .venv/bin/python demo/host/replay.py --run-dir examples/output/t14-rt-20261110 --act 4
    .venv/bin/python demo/host/replay.py --run-dir examples/output/t14-rt-20261110 \
        --act 4 --start 2.0 --end 3.5 --export-csv /tmp/act4_window.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "examples" / "output" / "t14-rt-20261110"


@dataclass(frozen=True)
class SignalSample:
    """九信号时间线的一行。"""

    timestamp_us: int
    theta: float
    theta_dot: float
    torque_cmd: float
    policy_target: float
    policy_latency_ms: float
    envelope_ok: bool
    governor_state: str
    injected_fault_id: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "SignalSample":
        return cls(
            timestamp_us=int(row["timestamp_us"]),
            theta=float(row["theta"]),
            theta_dot=float(row["theta_dot"]),
            torque_cmd=float(row["torque_cmd"]),
            policy_target=float(row["policy_target"]),
            policy_latency_ms=float(row["policy_latency_ms"]),
            envelope_ok=row["envelope_ok"] == "True",
            governor_state=row["governor_state"],
            injected_fault_id=row["injected_fault_id"],
        )


@dataclass(frozen=True)
class TimelineEvent:
    """报告中的时间区间标记。"""

    kind: str
    label: str
    start_us: int
    end_us: int | None = None


@dataclass(frozen=True)
class ActData:
    """单幕报告元数据和完整信号。"""

    act: int
    name: str
    passed: bool
    summary: dict
    signal_csv: Path
    samples: list[SignalSample]
    events: tuple[TimelineEvent, ...]


@dataclass(frozen=True)
class RunData:
    """一次 RT-SIL 运行的完整报告数据。"""

    run_dir: Path
    passed: bool
    target_evidence: bool
    note: str
    acts: tuple[ActData, ...]


def _resolve_signal_csv(run_dir: Path, value: str) -> Path:
    """兼容报告中的相对路径、目标机路径和同目录改名副本。"""
    name = Path(value).name
    candidates = (
        run_dir / name,
        ROOT / value,
        Path(value),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到 {value!r}；尝试过：{[str(p) for p in candidates]}")


def _read_signals(path: Path) -> list[SignalSample]:
    samples: list[SignalSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                samples.append(SignalSample.from_row(row))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: 九信号解析失败") from exc
    if not samples:
        raise ValueError(f"{path} 没有信号数据")
    return samples


def _contiguous_windows(samples: list[SignalSample], predicate) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start: int | None = None
    for sample in samples:
        active = predicate(sample)
        if active and start is None:
            start = sample.timestamp_us
        elif not active and start is not None:
            windows.append((start, sample.timestamp_us))
            start = None
    if start is not None:
        windows.append((start, samples[-1].timestamp_us))
    return windows


def _first_time(samples: list[SignalSample], predicate) -> int | None:
    for sample in samples:
        if predicate(sample):
            return sample.timestamp_us
    return None


def detect_events(samples: list[SignalSample]) -> tuple[TimelineEvent, ...]:
    """从九信号推导注入、违约、接管和 SAFE 事件。"""
    events: list[TimelineEvent] = []
    for start_us, end_us in _contiguous_windows(
        samples, lambda row: row.injected_fault_id != "none"
    ):
        events.append(
            TimelineEvent(
                kind="injection",
                label="故障注入",
                start_us=start_us,
                end_us=end_us,
            )
        )

    violation_us = _first_time(samples, lambda row: not row.envelope_ok)
    if violation_us is not None:
        events.append(
            TimelineEvent(
                kind="violation",
                label="包络违约",
                start_us=violation_us,
            )
        )

    takeover_us = _first_time(
        samples, lambda row: row.governor_state == "FALLBACK"
    )
    if takeover_us is not None:
        events.append(
            TimelineEvent(
                kind="takeover",
                label="Governor 接管",
                start_us=takeover_us,
            )
        )

    safe_us = _first_time(samples, lambda row: row.governor_state == "SAFE")
    if safe_us is not None:
        events.append(
            TimelineEvent(kind="safe", label="进入 SAFE", start_us=safe_us)
        )
    return tuple(events)


def load_run(run_dir: Path) -> RunData:
    """加载 `rt_sil_report.json` 及其四幕 CSV。"""
    report_path = run_dir / "rt_sil_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"报告不存在：{report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    acts: list[ActData] = []
    for item in report.get("acts", []):
        signal_csv = _resolve_signal_csv(run_dir, item["signal_csv"])
        samples = _read_signals(signal_csv)
        acts.append(
            ActData(
                act=int(item["act"]),
                name=str(item["name"]),
                passed=bool(item["passed"]),
                summary=item,
                signal_csv=signal_csv,
                samples=samples,
                events=detect_events(samples),
            )
        )
    if not acts:
        raise ValueError(f"{report_path} 不包含 acts")

    target_evidence = all(
        bool(act.summary.get("loop", {}).get("rt_kernel")) for act in acts
    )
    return RunData(
        run_dir=run_dir,
        passed=bool(report.get("passed")),
        target_evidence=target_evidence,
        note=str(report.get("note", "")),
        acts=tuple(acts),
    )


def get_act(run: RunData, act: int) -> ActData:
    for item in run.acts:
        if item.act == act:
            return item
    raise KeyError(f"运行中没有第 {act} 幕")


def select_window(
    samples: list[SignalSample],
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[SignalSample]:
    """按秒截取时间窗。"""
    start_us = 0 if start_s is None else round(start_s * 1_000_000)
    end_us = (
        samples[-1].timestamp_us
        if end_s is None
        else round(end_s * 1_000_000)
    )
    if end_us < start_us:
        raise ValueError("end_s 必须 >= start_s")
    selected = [
        row for row in samples if start_us <= row.timestamp_us <= end_us
    ]
    if not selected:
        raise ValueError("时间窗内没有采样")
    return selected


def export_csv(path: Path, samples: list[SignalSample]) -> None:
    """导出重放时间窗。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(SignalSample.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="RT-SIL 时间线回放")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--act", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--export-csv", type=Path)
    args = parser.parse_args()

    run = load_run(args.run_dir)
    act = get_act(run, args.act)
    window = select_window(act.samples, args.start, args.end)
    print(
        f"第 {act.act} 幕 {act.name}: {len(act.samples)} 行；"
        f"时间窗 {len(window)} 行"
    )
    for event in act.events:
        end = f"..{event.end_us / 1e6:.3f}s" if event.end_us is not None else ""
        print(
            f"  {event.label}: {event.start_us / 1e6:.3f}s{end}"
        )
    if args.export_csv:
        export_csv(args.export_csv, window)
        print(f"已导出：{args.export_csv}")


if __name__ == "__main__":
    main()
