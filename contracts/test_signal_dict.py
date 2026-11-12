"""校验 D1.8 九信号 CSV 是否符合 signal-dict schema v0.1。

运行（仓库根目录）：
    .venv/bin/python contracts/test_signal_dict.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts" / "signal-dict.schema.json"
OUTPUT_DIR = ROOT / "examples" / "output"
CSV_PATHS = sorted(OUTPUT_DIR.rglob("rt_sil_act*_signals.csv"))


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"非法 boolean：{value!r}")


def validate_row(row: dict[str, str], schema: dict) -> dict[str, object]:
    """把 CSV 字符串转成 schema 类型并执行 v0.1 必需检查。"""
    required = set(schema["required"])
    if set(row) != required:
        missing = required - set(row)
        extra = set(row) - required
        raise ValueError(f"字段集不一致；missing={missing}, extra={extra}")

    parsed: dict[str, object] = {
        "timestamp_us": int(row["timestamp_us"]),
        "theta": float(row["theta"]),
        "theta_dot": float(row["theta_dot"]),
        "torque_cmd": float(row["torque_cmd"]),
        "policy_target": float(row["policy_target"]),
        "policy_latency_ms": float(row["policy_latency_ms"]),
        "envelope_ok": _parse_bool(row["envelope_ok"]),
        "governor_state": row["governor_state"],
        "injected_fault_id": row["injected_fault_id"],
    }
    properties = schema["properties"]
    if parsed["timestamp_us"] < 0:
        raise ValueError("timestamp_us 不得为负")
    if parsed["policy_latency_ms"] < 0:
        raise ValueError("policy_latency_ms 不得为负")
    if parsed["governor_state"] not in properties["governor_state"]["enum"]:
        raise ValueError(f"非法 governor_state：{parsed['governor_state']}")
    if not re.fullmatch(
        properties["injected_fault_id"]["pattern"], str(parsed["injected_fault_id"])
    ):
        raise ValueError(f"非法 injected_fault_id：{parsed['injected_fault_id']}")
    for field in (
        "theta",
        "theta_dot",
        "torque_cmd",
        "policy_target",
        "policy_latency_ms",
    ):
        value = float(parsed[field])
        if value != value:
            raise ValueError(f"{field} 为 NaN")
    return parsed


def validate_csv(path: Path, schema: dict) -> tuple[int, int]:
    """校验一个 CSV，返回 (行数, 最大故障数)。"""
    previous_timestamp: int | None = None
    count = 0
    max_faults = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                parsed = validate_row(row, schema)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            timestamp = int(parsed["timestamp_us"])
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(f"{path}:{line_number}: timestamp 未严格递增")
            previous_timestamp = timestamp
            fault_text = str(parsed["injected_fault_id"])
            if fault_text != "none":
                max_faults = max(max_faults, len(fault_text.split("|")))
            count += 1
    return count, max_faults


def main() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not CSV_PATHS:
        raise SystemExit("未找到 D1.8 九信号 CSV")

    total_rows = 0
    for path in CSV_PATHS:
        rows, max_faults = validate_csv(path, schema)
        total_rows += rows
        print(
            f"{path.relative_to(ROOT)}: {rows} 行，"
            f"最多同时故障数 {max_faults} ✔"
        )

    acts = {path.name for path in CSV_PATHS}
    required_acts = {f"rt_sil_act{act}_signals.csv" for act in range(1, 5)}
    if acts != required_acts:
        raise SystemExit(f"四幕文件不完整：{sorted(acts)}")
    print(
        f"\n=== signal-dict v0.1 校验通过："
        f"{len(CSV_PATHS)} 个文件 / {total_rows} 行 ==="
    )


if __name__ == "__main__":
    main()
