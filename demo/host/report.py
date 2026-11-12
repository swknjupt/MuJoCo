"""生成单文件 RT-SIL HTML 报告与九信号重放（Task D / D3.6）。

运行（仓库根目录）：
    .venv/bin/python demo/host/report.py
    .venv/bin/python demo/host/report.py --run-dir examples/output/t14-rt-20261110
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.host.replay import (
    DEFAULT_RUN_DIR,
    ActData,
    RunData,
    SignalSample,
    load_run,
)

SVG_WIDTH = 1120
SVG_HEIGHT = 120
PLOT_LEFT = 72
PLOT_RIGHT = 1080
PLOT_TOP = 16
PLOT_BOTTOM = 88
MAX_POINTS = 600

NUMERIC_SIGNALS = (
    ("theta", "theta / rad"),
    ("theta_dot", "theta_dot / rad·s⁻¹"),
    ("torque_cmd", "torque_cmd / N·m"),
    ("policy_target", "policy_target / rad"),
    ("policy_latency_ms", "policy_latency_ms / ms"),
)

STATE_COLORS = {
    "NORMAL": "#2f80ed",
    "FALLBACK": "#f2994a",
    "SAFE": "#27ae60",
    "DISABLED": "#7f8c8d",
}
FAULT_COLORS = {
    "none": "#d9dee7",
    "obs_delay": "#8e44ad",
    "encoder_dropout": "#c0392b",
    "torque_attenuation": "#d68910",
}


def _downsample(samples: list[SignalSample]) -> list[SignalSample]:
    stride = max(1, math.ceil(len(samples) / MAX_POINTS))
    reduced = samples[::stride]
    if reduced[-1] is not samples[-1]:
        reduced.append(samples[-1])
    return reduced


def _x(timestamp_us: int, start_us: int, end_us: int) -> float:
    span = max(1, end_us - start_us)
    return PLOT_LEFT + (timestamp_us - start_us) / span * (PLOT_RIGHT - PLOT_LEFT)


def _event_markers(act: ActData) -> str:
    start_us = act.samples[0].timestamp_us
    end_us = act.samples[-1].timestamp_us
    colors = {
        "injection": "#8e44ad",
        "violation": "#c0392b",
        "takeover": "#f2994a",
        "safe": "#27ae60",
    }
    markers: list[str] = []
    for event in act.events:
        x1 = _x(event.start_us, start_us, end_us)
        if event.end_us is not None:
            x2 = _x(event.end_us, start_us, end_us)
            markers.append(
                f'<rect class="event injection-window" x="{x1:.2f}" y="{PLOT_TOP}" '
                f'width="{max(1, x2 - x1):.2f}" height="{PLOT_BOTTOM - PLOT_TOP}" '
                f'fill="{colors[event.kind]}" opacity="0.12">'
                f"<title>{html.escape(event.label)}</title></rect>"
            )
        else:
            markers.append(
                f'<line class="event {event.kind}" x1="{x1:.2f}" y1="{PLOT_TOP}" '
                f'x2="{x1:.2f}" y2="{PLOT_BOTTOM}" stroke="{colors[event.kind]}" '
                f'stroke-width="2"><title>{html.escape(event.label)}</title></line>'
            )
    return "".join(markers)


def _numeric_chart(act: ActData, field: str, label: str) -> str:
    rows = _downsample(act.samples)
    values = [float(getattr(row, field)) for row in rows]
    low, high = min(values), max(values)
    if math.isclose(low, high):
        low -= 1.0
        high += 1.0
    padding = (high - low) * 0.08
    low -= padding
    high += padding
    start_us = act.samples[0].timestamp_us
    end_us = act.samples[-1].timestamp_us
    points = " ".join(
        f"{_x(row.timestamp_us, start_us, end_us):.2f},"
        f"{PLOT_BOTTOM - (float(getattr(row, field)) - low) / (high - low) * (PLOT_BOTTOM - PLOT_TOP):.2f}"
        for row in rows
    )
    return f"""
    <div class="chart-wrap">
      <div class="chart-label">{html.escape(label)}</div>
      <svg viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-label="{html.escape(label)}">
        <rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_RIGHT - PLOT_LEFT}" height="{PLOT_BOTTOM - PLOT_TOP}" class="plot-bg"/>
        {_event_markers(act)}
        <polyline points="{points}" class="trace" data-field="{field}"/>
        <text x="8" y="{PLOT_TOP + 6}" class="axis">{high:.3g}</text>
        <text x="8" y="{PLOT_BOTTOM}" class="axis">{low:.3g}</text>
      </svg>
    </div>"""


def _band_lane(act: ActData, field: str, label: str, palette: dict[str, str]) -> str:
    start_us = act.samples[0].timestamp_us
    end_us = act.samples[-1].timestamp_us
    bands: list[tuple[int, int, str]] = []
    current_value = getattr(act.samples[0], field)
    band_start = act.samples[0].timestamp_us
    for row in act.samples[1:]:
        value = getattr(row, field)
        if value != current_value:
            bands.append((band_start, row.timestamp_us, current_value))
            current_value = value
            band_start = row.timestamp_us
    bands.append((band_start, end_us, current_value))
    rects = "".join(
        f'<rect x="{_x(start, start_us, end_us):.2f}" y="{PLOT_TOP}" '
        f'width="{max(1, _x(end, start_us, end_us) - _x(start, start_us, end_us)):.2f}" '
        f'height="{PLOT_BOTTOM - PLOT_TOP}" '
        f'fill="{palette.get(value, palette.get(str(value), "#888"))}">'
        f"<title>{html.escape(str(value))}</title></rect>"
        for start, end, value in bands
    )
    return f"""
    <div class="chart-wrap">
      <div class="chart-label">{html.escape(label)}</div>
      <svg viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-label="{html.escape(label)}">
        <rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_RIGHT - PLOT_LEFT}" height="{PLOT_BOTTOM - PLOT_TOP}" class="plot-bg"/>
        {rects}
      </svg>
    </div>"""


def _boolean_lane(act: ActData) -> str:
    start_us = act.samples[0].timestamp_us
    end_us = act.samples[-1].timestamp_us
    chart = _band_lane(
        act,
        "envelope_ok",
        "envelope_ok",
        {True: "#27ae60", False: "#c0392b"},
    )
    del start_us, end_us
    return chart


def _fault_lane(act: ActData) -> str:
    return _band_lane(
        act,
        "injected_fault_id",
        "injected_fault_id",
        FAULT_COLORS,
    )


def _act_summary(act: ActData) -> str:
    loop = act.summary.get("loop", {})
    takeover = act.summary.get("takeover_latency_ms")
    safe = act.summary.get("safe_s")
    safe_delay = (
        f"{safe - act.summary['takeover_s']:.3f}s"
        if safe is not None and act.summary.get("takeover_s") is not None
        else "—"
    )
    rows = (
        ("状态", "通过" if act.passed else "未通过"),
        ("起摆/保持", f"{act.summary.get('max_hold_s', 0):.3f}s"),
        ("发散", "是" if act.summary.get("divergence") else "否"),
        ("越界→接管", "—" if takeover is None else f"{takeover:.3f}ms"),
        ("接管→SAFE", safe_delay),
        ("overrun", str(loop.get("overrun_count", "—"))),
        ("P99.99 jitter", f"{loop.get('jitter_p99_99_us', 0):.3f}µs"),
        ("Max jitter", f"{loop.get('jitter_max_us', 0):.3f}µs"),
    )
    cells = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>"
        for name, value in rows
    )
    return f'<table class="act-summary">{cells}</table>'


def _act_section(act: ActData) -> str:
    chart_html = "".join(
        _numeric_chart(act, field, label) for field, label in NUMERIC_SIGNALS
    )
    chart_html += _boolean_lane(act)
    chart_html += _band_lane(
        act, "governor_state", "governor_state", STATE_COLORS
    )
    chart_html += _fault_lane(act)
    start_s = act.samples[0].timestamp_us / 1e6
    end_s = act.samples[-1].timestamp_us / 1e6
    return f"""
  <section class="act-panel" data-act="{act.act}" hidden>
    <div class="act-header">
      <div><h2>第 {act.act} 幕：{html.escape(act.name)}</h2>
      <p>timestamp_us：{start_s:.3f}s–{end_s:.3f}s · {len(act.samples)} 行</p></div>
      {_act_summary(act)}
    </div>
    {chart_html}
    <div class="cursor-readout" id="readout-{act.act}"></div>
  </section>"""


def _embedded_data(run: RunData) -> str:
    payload = {
        str(act.act): [asdict(row) for row in act.samples] for act in run.acts
    }
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
    )


def build_html(run: RunData) -> str:
    """构建无外部资源依赖的单文件 HTML 报告。"""
    if not run.acts:
        raise ValueError("没有可报告的幕")
    target_text = "T14 PREEMPT_RT 目标机证据" if run.target_evidence else "Host 功能证据"
    tabs = "".join(
        f'<button type="button" class="tab{" active" if index == 0 else ""}" '
        f'data-target="{act.act}">第 {act.act} 幕</button>'
        for index, act in enumerate(run.acts)
    )
    sections = "".join(_act_section(act) for act in run.acts)
    embedded = _embedded_data(run)
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    passed_class = "pass" if run.passed else "fail"
    passed_text = "全部通过" if run.passed else "存在失败"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>机器人 MBD Task D · 九信号报告</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#667085;--line:#d8deea;--accent:#2457d6}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{padding:28px 36px 18px;background:#101a33;color:#fff}} header h1{{margin:0 0 8px;font-size:26px}}
.meta{{color:#c6d2ee}} main{{padding:24px 32px 48px;max-width:1280px;margin:auto}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.badge{{display:inline-block;padding:3px 9px;border-radius:999px;color:#fff;font-weight:700}}
.pass{{background:#217a4b}} .fail{{background:#b42318}}
.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}}
.tab{{border:1px solid var(--line);background:#fff;color:var(--ink);padding:9px 15px;border-radius:9px;cursor:pointer}}
.tab.active{{background:var(--accent);border-color:var(--accent);color:#fff}}
.act-panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}}
.act-header{{display:grid;grid-template-columns:1fr 360px;gap:20px;align-items:start;margin-bottom:16px}}
.act-header h2{{margin:0 0 6px}} .act-header p{{margin:0;color:var(--muted)}}
.act-summary{{width:100%;border-collapse:collapse}} .act-summary th,.act-summary td{{padding:4px 7px;border-bottom:1px solid var(--line);text-align:left}}
.act-summary th{{color:var(--muted);font-weight:500;width:43%}}
.chart-wrap{{position:relative;margin:10px 0 18px}} .chart-label{{font-weight:650;margin:0 0 5px 4px}}
svg{{display:block;width:100%;height:auto;overflow:visible}} .plot-bg{{fill:#f8faff;stroke:#e3e8f2}}
.trace{{fill:none;stroke:var(--accent);stroke-width:2;vector-effect:non-scaling-stroke}}
.axis{{font-size:11px;fill:#667085}} .cursor{{stroke:#111827;stroke-width:1.5;stroke-dasharray:4 3;pointer-events:none}}
.controls{{position:sticky;top:0;z-index:5;background:rgba(245,247,251,.96);padding:12px 0;border-bottom:1px solid var(--line)}}
.controls input{{width:100%}} .cursor-readout{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}}
.value{{background:#f7f9fd;border:1px solid var(--line);border-radius:8px;padding:8px 10px}}
.value b{{display:block;color:var(--muted);font-size:12px}} @media(max-width:800px){{.cards{{grid-template-columns:repeat(2,1fr)}}.act-header{{grid-template-columns:1fr}}.cursor-readout{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<header>
  <h1>机器人 MBD Task D · 九信号时间线报告</h1>
  <div class="meta">{html.escape(target_text)} · {len(run.acts)} 幕 · 生成于 {html.escape(generated)}</div>
</header>
<main>
  <div class="cards">
    <div class="card"><div>运行结果</div><div class="badge {passed_class}">{passed_text}</div></div>
    <div class="card"><div>数据来源</div><strong>{html.escape(run.run_dir.name)}</strong></div>
    <div class="card"><div>信号总量</div><strong>{sum(len(a.samples) for a in run.acts):,} 行</strong></div>
    <div class="card"><div>事件类型</div><strong>注入 / 违约 / 接管 / SAFE</strong></div>
  </div>
  <p class="meta">{html.escape(run.note)}</p>
  <div class="tabs">{tabs}</div>
  <div class="controls"><input id="cursor" type="range" min="0" max="1000" value="0" aria-label="回放位置"></div>
  {sections}
</main>
<script id="signal-data" type="application/json">{embedded}</script>
<script>
const DATA=JSON.parse(document.getElementById('signal-data').textContent);
let active=String(Object.keys(DATA)[0]);
const cursor=document.getElementById('cursor');
function panelsForAct(){{return document.querySelectorAll('.act-panel');}}
function cursorsForAct(){{
  return document.querySelectorAll(`.act-panel[data-act="${{active}}"] .chart-wrap svg`);
}}
function setAct(act){{
  active=String(act);
  document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.target===active));
  document.querySelectorAll('.act-panel').forEach(p=>p.hidden=p.dataset.act!==active);
  cursor.value=0; render();
}}
function render(){{
  const rows=DATA[active], row=rows[Math.min(rows.length-1,Math.round(Number(cursor.value)/1000*(rows.length-1)))];
  const start=rows[0].timestamp_us,end=rows[rows.length-1].timestamp_us;
  const x=72+(row.timestamp_us-start)/Math.max(1,end-start)*1008;
  cursorsForAct().forEach(svg=>{{
    let line=svg.querySelector('.cursor');
    if(!line){{line=document.createElementNS('http://www.w3.org/2000/svg','line');line.setAttribute('class','cursor');line.setAttribute('y1','16');line.setAttribute('y2','88');svg.appendChild(line);}}
    line.setAttribute('x1',x);line.setAttribute('x2',x);
  }});
  document.getElementById('readout-'+active).innerHTML=Object.entries(row).map(([k,v])=>`<div class="value"><b>${{k}}</b>${{String(v)}}</div>`).join('');
}}
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>setAct(b.dataset.target)));
cursor.addEventListener('input',render);
setAct(Object.keys(DATA)[0]);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 RT-SIL 单文件 HTML 报告")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run = load_run(args.run_dir)
    output = args.output or (args.run_dir / "report.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(run), encoding="utf-8")
    print(
        f"报告：{output}\n"
        f"来源：{'T14 RT' if run.target_evidence else 'host'} · "
        f"{len(run.acts)} 幕 · {sum(len(a.samples) for a in run.acts)} 行"
    )


if __name__ == "__main__":
    main()
