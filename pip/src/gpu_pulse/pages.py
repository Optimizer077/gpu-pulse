"""Per-page refresh logic for the dashboard, split by cost (live vs slow).

Only the visible page is rebuilt; switching tabs triggers a full refresh of the new page.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

from rich.console import Group
from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from . import cards as C
from . import panels as P
from . import render as R
from .models import GpuSample, Sample
from .monitor import Analysis
from .widgets import Panel, PulseStrip, RingGauge

if TYPE_CHECKING:
    from .app import GpuDashboard

RANGES = (60, 300, 900, 3600)


def _panel(app: GpuDashboard, pid: str) -> Panel:
    return app.query_one(f"#{pid}", Panel)


def _pct(v: Optional[float]) -> str:
    return R.fmt(v, 0, "%")


# ---- rings (every sample, Overview only) ---------------------------------------------------
def rings(app: GpuDashboard, s: Sample, g: Optional[GpuSample], a: Analysis) -> None:
    hour = a.windows[2]
    q = app.query_one
    gpu = q("#ring-gpu", RingGauge)
    if g is None:
        gpu.set_target(None, "no GPU detected")
        for rid in ("vram", "temp", "power"):
            q(f"#ring-{rid}", RingGauge).set_target(None, "n/a")
    else:
        gpu.border_title = f"GPU · {g.name.replace('NVIDIA GeForce ', '').replace('NVIDIA ', '')}"
        caption = f"SM {g.sm_active:.0f}%" if g.sm_active is not None else "utilization"
        sub = f"1h avg {hour.util:.0f}%" if hour.util is not None else ""
        side = [("1h avg", _pct(hour.util)), ("peak", _pct(hour.util_peak))]
        if g.sm_active is not None:
            side.append(("SM", _pct(g.sm_active)))
        if g.tensor_active is not None:
            side.append(("tensor", _pct(g.tensor_active)))
        side += [("clock", f"{R.fmt(g.clock_mhz)} MHz"), ("state", g.pstate or "–")]
        gpu.set_target(
            g.util,
            caption,
            sub,
            marker=hour.util / 100 if hour.util is not None else None,
            side=tuple(side),
        )
        gpu.set_state("normal", f"◆ 1h {hour.util:.0f}%" if hour.util is not None else "")
        used, total = (g.mem_used_mib or 0) / 1024, (g.mem_total_mib or 0) / 1024
        q("#ring-vram", RingGauge).set_target(
            g.vram_pct,
            f"{used:.1f} / {total:.0f} GB",
            f"1h avg {hour.vram:.0f}%" if hour.vram is not None else "",
            side=(("used", f"{used:.1f} GB"), ("total", f"{total:.1f} GB"), ("1h avg", _pct(hour.vram)),
                  ("bus", _pct(g.mem_util))),
        )  # fmt: skip
        warn = app.mon.cfg.temp_warn
        temp = q("#ring-temp", RingGauge)
        temp.set_target(
            g.temp_c,
            "GPU core",
            f"1h peak {hour.temp_peak:.0f}°C" if hour.temp_peak else "",
            max_value=max(100.0, warn + 10),
            side=(("1h peak", R.fmt(hour.temp_peak, 0, "°C")), ("warn at", f"{warn:.0f}°C"), ("fan", _pct(g.fan_pct))),
        )
        t = g.temp_c or 0
        temp.set_state(
            *(("hot", "⬣ over limit") if t >= warn else ("warn", "▲ near limit") if t >= warn - 10 else ("normal", ""))
        )
        power = q("#ring-power", RingGauge)
        capped = any("power" in r.lower() for r in C.throttle_reasons(g, app.diag))
        power.set_target(
            g.power_w,
            f"of {R.fmt(g.power_limit_w)} W cap",
            f"1h avg {hour.power:.0f} W" if hour.power else "",
            max_value=g.power_limit_w or 100.0,
            side=(("cap", f"{R.fmt(g.power_limit_w)} W"), ("1h avg", R.fmt(hour.power, 0, " W")),
                  ("draw", _pct((g.power_w or 0) / g.power_limit_w * 100 if g.power_limit_w else None))),
        )  # fmt: skip
        power.set_state("warn", "⚡ power capped") if capped else power.set_state("normal", "")
    hot = max(s.cpu_per_core, default=0.0)
    cpu_side = [("max core", f"{hot:.0f}%"), ("threads", str(len(s.cpu_per_core)))]
    if s.cpu_freq_mhz:
        cpu_side.append(("clock", f"{s.cpu_freq_mhz / 1000:.2f} GHz"))
    if s.cpu_temp_c is not None:
        cpu_side.append(("temp", f"{s.cpu_temp_c:.0f}°C"))
    q("#ring-cpu", RingGauge).set_target(
        s.cpu_util,
        f"{s.cpu_freq_mhz / 1000:.1f} GHz" if s.cpu_freq_mhz else f"{len(s.cpu_per_core)} threads",
        f"max core {hot:.0f}%",
        side=tuple(cpu_side),
    )
    gb_used, gb_total = s.mem_used / 2**30, s.mem_total / 2**30
    q("#ring-ram", RingGauge).set_target(
        s.mem_pct,
        f"{gb_used:.1f} / {gb_total:.0f} GB",
        f"swap {s.swap_pct:.0f}%" if s.swap_pct is not None else "",
        side=(("used", f"{gb_used:.1f} GB"), ("total", f"{gb_total:.1f} GB"), ("swap", _pct(s.swap_pct))),
    )


# ---- live panels (every sample) ------------------------------------------------------------
def _series(app: GpuDashboard, s: Sample, g: Optional[GpuSample], rng: int) -> dict[str, list[Optional[float]]]:
    pts = [x for x in app.mon.history.recent if x.t >= s.t - rng]
    uid = g.uuid if g else None
    picked = [next((y for y in x.gpus if y.uuid == uid), None) for x in pts]
    out: dict[str, list[Optional[float]]] = {
        "util": [p.util if p else None for p in picked],
        "vram": [p.vram_pct if p else None for p in picked],
        "sm": [p.sm_active if p else None for p in picked],
        "tensor": [p.tensor_active if p else None for p in picked],
        "dram": [p.dram_bw_util if p else None for p in picked],
        "power": [p.power_w / p.power_limit_w * 100 if p and p.power_w is not None and p.power_limit_w else None
                  for p in picked],
    }  # fmt: skip
    expected = max(1, int(rng / app.mon.cfg.interval_s))  # left-pad so the chart spans the range
    if len(picked) < expected:
        missing = expected - len(picked)
        pad: list[Optional[float]] = [None] * missing
        out = {k: pad + v for k, v in out.items()}
        # Backfill utilization from per-minute history (e.g. the logger ran before this session).
        first_t = pts[0].t if pts else s.t
        _, cells = app.mon.history.heatmap(uid, hours=max(1, math.ceil(rng / 3600)))
        step = app.mon.cfg.interval_s
        minute_now = int(s.t // 60)
        for i in range(missing):
            t = first_t - (missing - i) * step
            k = len(cells) - 1 - (minute_now - int(t // 60))  # heatmap's last cell is the current minute
            if 0 <= k < len(cells):
                out["util"][i] = cells[k]
    return out


def _truth_cols(want: int, width: int) -> int:
    return 2 if want == 2 and width >= P.TRUE_TWO_COL_MIN else 1


def _truth_rows(g: Optional[GpuSample], cols: int) -> int:
    n = sum(1 for _, attr, _, _ in P.TRUE_METRICS if getattr(g, attr, None) is not None)
    return 3 if n == 0 else math.ceil(n / cols) + 2


def live(app: GpuDashboard, s: Sample, g: Optional[GpuSample], a: Analysis) -> None:
    L = app.layout_plan
    if app.tab == "overview":
        app.query_one(PulseStrip).util = g.util if g else None
        rng = RANGES[app.range_i]
        if L.columns >= 2:
            tl = _panel(app, "timeline")
            tl.border_title = f"GPU TIMELINE · last {rng // 60}m (r)"
            tl.update(P.timeline(_series(app, s, g, rng), g, tl.inner_width, tl.inner_height))
        truth = _panel(app, "truth" if L.columns == 3 else "truth-r")
        cols = _truth_cols(1 if L.columns == 3 else 2, truth.inner_width)
        truth.styles.height = _truth_rows(g, cols) + 2
        truth.update(P.true_usage(g, truth.inner_width, _truth_rows(g, cols), cols))
        lo = _panel(app, "live-o")
        if L.columns == 3 and lo.display:
            lo.update(P.gpu_live(g, lo.inner_width))
        if L.show_bottom:
            io = _panel(app, "io")
            io.update(P.io_block(list(app.mon.history.recent)[-240:], s, io.inner_width, io.inner_height))
        arch = app.query_one("#ov-arch", Static)
        if arch.display:
            line = Text("▍", Style(color=R.GPU))
            line.append_text(P.arch_summary(app.detail_for(g), g))
            arch.update(line)
    elif app.tab == "gpu":
        live_p = _panel(app, "gpu-live")
        live_p.update(P.gpu_live(g, live_p.inner_width))
        tg = _panel(app, "truth-g")
        tg.update(P.true_usage(g, tg.inner_width, _truth_rows(g, 1), 1))
        tr = _panel(app, "gpu-trace")
        rng = RANGES[app.range_i]
        tr.border_title = f"TRACES · last {rng // 60}m (r) · 0–100% scale"
        tr.update(P.gpu_traces(_series(app, s, g, rng), tr.inner_width, tr.inner_height))
    elif app.tab == "system":
        recent = list(app.mon.history.recent)[-int(300 / app.mon.cfg.interval_s) :]
        cpu = _panel(app, "cpu")
        threads = app.specs.cpu_threads if app.specs else None
        cpu.update(P.cpu_block(recent, s, cpu.inner_width, cpu.inner_height, threads))
        io = _panel(app, "io-s")
        io.border_title = "DISK & NETWORK · last 5 min, shared scale per device"
        io.update(P.io_block(recent, s, io.inner_width, io.inner_height))


# ---- slow panels (every ~2 s) --------------------------------------------------------------
def slow(app: GpuDashboard, s: Sample, g: Optional[GpuSample], a: Analysis) -> None:
    L = app.layout_plan
    d = app.diag
    grade = a.grade[0] if a.grade else None
    uid = g.uuid if g else None
    tab = app.tab
    if tab in ("overview", "history"):
        start, cells = app.mon.history.heatmap(uid, hours=25)
    if tab == "overview":
        diag = _panel(app, "diag")
        diag.border_subtitle = f"score {d.score}/100" if d.score is not None else ""
        diag.update(C.diagnosis_card(d, a, g, diag.inner_width, diag.inner_height))
        rows = C.health_rows(d, s, g, app.mon.cfg.temp_warn, grade)
        problems = [r for r in rows if r.level in ("warning", "serious", "critical")]
        note = f"▲ {len(problems)} need attention" if problems else "✔ all clear"
        hr = _panel(app, "health-r")
        truth_h = _truth_rows(g, _truth_cols(2, _panel(app, "truth-r").inner_width)) + 2
        mid_h = L.middle_h - (1 if app.query_one("#ov-arch").display else 0)
        center_left = mid_h - (_truth_rows(g, 1) + 2) - (len(rows) + 2)
        packed = center_left < 0  # not enough height: two entries per line instead of clipping rows
        health_h = (math.ceil(len(rows) / 2) if packed else len(rows)) + 2
        for pid in ("health", "health-r"):
            hp = _panel(app, pid)
            hp.border_subtitle = note
            two = packed if pid == "health" else False
            hp.update(C.health_grid(rows, hp.inner_width, detail=hp.inner_width >= 30, columns=2 if two else 1))
        _panel(app, "health").styles.height = health_h
        hr.styles.height = len(rows) + 2
        hr.display = L.columns < 3 and mid_h - truth_h - (len(rows) + 2) >= 10
        _panel(app, "live-o").display = L.columns == 3 and center_left >= 8
        if L.show_bottom:
            usage = _panel(app, "usage")
            usage.update(
                P.usage_strip(
                    a.windows, [cells[-w.minutes :] for w in a.windows], usage.inner_width, usage.inner_height
                )
            )
    elif tab == "gpu":
        arch = _panel(app, "arch")
        arch.update(P.arch_card(app.detail_for(g), g, arch.inner_width))
        procs = _panel(app, "procs")
        mine = [p for p in app.mon.processes if not g or p.gpu_uuid == g.uuid]
        procs.border_title = f"GPU PROCESSES · {len(mine)}"
        if app.mon.vendor != "nvidia" and not app.mon.cfg.demo:
            procs.update(Text("Process listing needs an NVIDIA GPU (NVML).", Style(color=R.MUTED)))
        elif not mine:
            procs.update(Text("No compute processes on this GPU.", Style(color=R.MUTED)))
        else:
            procs.update(P.procs_table(mine, g.mem_total_mib if g else None, procs.inner_width, 20))
    elif tab == "history":
        for i, w in enumerate(a.windows):
            wp = _panel(app, f"win{i}")
            wp.update(P.window_card(w, cells[-w.minutes :], wp.inner_width, wp.inner_height))
        heat = _panel(app, "heat")
        col_w = 2 if heat.inner_width >= 126 else 1
        tall = app.layout_plan.height - 2 - app.query_one("#wins").outer_size.height >= 32
        _, sm = app.mon.history.heatmap(uid, hours=25, metric="sm")
        parts: list[Any] = [R.heatmap(start, cells, col_w=col_w, tall=tall), Text(), R.heat_legend()]
        if any(v is not None for v in sm[-1440:]):  # true SM activity per minute (GPM), same 24 h
            line = Text("   SM ", Style(color=R.MUTED), no_wrap=True)
            line.append_text(R.sparkline(sm[-1440:], 60 * col_w, R.mix(R.GPU_GLOW, "#ffffff", 0.35)))
            line.append("  24 h SM activity", Style(color=R.DIM))
            parts[1:1] = [Text(), line]
        heat.update(Group(*parts))
        runs = C.find_sessions(start, cells)
        sp = _panel(app, "sessions")
        sp.border_title = f"SESSIONS · last 24 h · {len(runs)}"
        head = C.session_line(getattr(d, "session", None))
        body = C.sessions_table(runs, sp.inner_width, 12)
        sp.update(Group(head, Text(), body) if head is not None else body)
    elif tab == "diagnosis":
        df = _panel(app, "diag-full")
        df.update(C.diagnosis_detail(d, a, g, df.inner_width))
        hf = _panel(app, "health-full")
        hf.update(C.health_grid(C.health_rows(d, s, g, app.mon.cfg.temp_warn, grade), hf.inner_width))
        _panel(app, "insights").update(C.insights_list(C.merged_insights(a, d)))
        td = _panel(app, "truth-d")
        td.update(P.true_usage(g, td.inner_width, _truth_rows(g, 1), 1))
