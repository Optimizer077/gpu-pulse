"""Intelligence-facing cards: diagnosis, system health, insights, sessions.

Everything here is a pure function from data to a Rich renderable, sized to the panel it
lands in. Engine fields are read defensively (``getattr``) so the UI keeps working while the
analysis engine grows.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

from rich.console import Group, RenderableType
from rich.style import Style
from rich.table import Table
from rich.text import Text

from . import anim
from . import render as R
from ._intel_gpu import ACTIVE_UTIL
from .intel import SESSION_GAP_MIN
from .models import GpuSample, Insight, Sample
from .monitor import Analysis

BOTTLENECK_LABEL = {
    "none": "No bottleneck",
    "idle": "Idle",
    "input": "Input pipeline",
    "launch": "Low SM parallelism",
    "cpu": "CPU-bound",
    "power": "Power cap",
    "thermal": "Thermal limit",
    "memory": "Memory bandwidth",
    "vram_held": "VRAM held while idle",
    "ram": "System RAM pressure",
    "disk": "Disk I/O",
    "network": "Network I/O",
    "unknown": "Undetermined",
}
BOTTLENECK_LEVEL = {"none": "good", "idle": "info", "unknown": "info", "thermal": "serious"}


def bottleneck_label(b: str) -> str:
    return BOTTLENECK_LABEL.get(b, b.replace("_", " ").title())


def _line(text: str = "", color: str = R.INK, bold: bool = False) -> Text:
    return Text(text, Style(color=color, bold=bold), no_wrap=True, overflow="ellipsis")


def throttle_reasons(g: Optional[GpuSample], diag: Any) -> tuple[str, ...]:
    seen: list[str] = []
    for r in tuple(getattr(g, "throttle", ()) or ()) + tuple(getattr(diag, "throttle", ()) or ()):
        if r and r.lower() not in (x.lower() for x in seen):
            seen.append(r)
    return tuple(seen)


def badges(reasons: Sequence[str]) -> Text:
    t = Text(no_wrap=True, overflow="ellipsis")
    for r in reasons:
        hot = "thermal" in r.lower() or "hw" in r.lower()
        t.append_text(R.badge(r, R.HOT if hot else R.WARN))
        t.append(" ")
    return t


def confidence_meter(conf: float, cells: int = 5) -> Text:
    n = round(max(0.0, min(1.0, conf)) * cells)
    t = Text("▰" * n, Style(color=R.GPU_GLOW))
    t.append("▱" * (cells - n), Style(color=R.DIM))
    return t


def session_line(sess: Any, mem: str = "VRAM") -> Optional[Text]:
    """``mem`` names the GPU's memory ("VRAM", or "GPU mem" on Apple Silicon)."""
    if sess is None:
        return None
    live = getattr(sess, "end", None) is None
    t = _line("⏱ ", R.GPU_GLOW)
    t.append("Session " if live else "Last session ", Style(color=R.MUTED))
    t.append(f"{getattr(sess, 'minutes', 0)} min", Style(color=R.INK, bold=True))
    t.append(f" · avg {getattr(sess, 'mean_util', 0):.0f}%", Style(color=R.INK))
    t.append(f" · {mem} peak {getattr(sess, 'peak_vram_pct', 0):.0f}%", Style(color=R.MUTED))
    wh = getattr(sess, "energy_wh", None)
    if wh is not None:
        t.append(f" · {wh:.0f} Wh", Style(color=R.POWER_GLOW))
    if not live:
        t.append(f" · ended {time.strftime('%H:%M', time.localtime(sess.end))}", Style(color=R.MUTED))
    return t


# ---- diagnosis -----------------------------------------------------------------------------
BASIS_LABEL = {"sm_active": "SM", "utilization": "util"}


def score_block(score: Optional[int], verdict: str, width: int, basis: Optional[str] = None) -> list[Text]:
    """3 rows: big score digits, then label / bar / verdict beside them."""
    color = R.score_color(score)
    top, mid, bot = anim.big_text(str(score) if score is not None else "-")
    dw = max(len(top), len(mid), len(bot), 3)
    bar_w = max(6, min(24, width - dw - 12))
    right = [
        _line("EFFICIENCY", R.MUTED),
        R.bar((score or 0) / 100, bar_w, R.mix(color, R.BG, 0.25), color),
        _line(verdict, color, True),
    ]
    tag = " · 1h" + (f" · {BASIS_LABEL.get(basis, basis)}" if basis else "") if score is not None else " · collecting"
    right[0].append(tag, Style(color=R.DIM))
    right[1].append(f" {score}/100" if score is not None else "  –/100", Style(color=R.INK, bold=True))
    rows = []
    for digits, side in zip((top, mid, bot), right):
        t = Text(digits.ljust(dw) + "   ", Style(color=color, bold=True), no_wrap=True, overflow="ellipsis")
        t.append_text(side)
        t.truncate(width, overflow="ellipsis")
        rows.append(t)
    return rows


def diagnosis_card(
    diag: Any, a: Analysis, g: Optional[GpuSample], width: int, height: int, extra: Sequence[Insight] = ()
) -> Group:
    """Overview card: fit the most important lines into ``height`` rows, in reading order."""
    score: Optional[int] = getattr(diag, "score", None)
    bottleneck: str = getattr(diag, "bottleneck", "unknown")
    neutral = score is None and bottleneck == "unknown"
    grade = a.grade[0] if a.grade else None
    verdict = getattr(diag, "verdict", "") or "Collecting data"
    headline = grade if neutral and grade else verdict
    items: list[tuple[int, int, list[Text]]] = []  # (priority, order, rows)
    if height >= 6:
        items.append((0, 0, score_block(score, headline, width, getattr(diag, "score_basis", None))))
    else:
        t = _line(f"{score if score is not None else '–'}/100 ", R.score_color(score), True)
        t.append(headline, Style(color=R.INK, bold=True))
        items.append((0, 0, [t]))
    if not neutral:
        lvl = BOTTLENECK_LEVEL.get(bottleneck, "warning")
        t = _line(f"{R.LEVEL_ICON[lvl]} ", R.LEVEL_COLOR[lvl], True)
        t.append(bottleneck_label(bottleneck), Style(color=R.INK, bold=True))
        conf = float(getattr(diag, "confidence", 0.0) or 0.0)
        t.append("  ")
        t.append_text(confidence_meter(conf))
        t.append(f" {conf * 100:.0f}% sure", Style(color=R.MUTED))
        items.append((1, 1, [t]))
        if verdict != headline:
            items.append((2, 2, [_line(verdict, R.INK)]))
    reasons = throttle_reasons(g, diag)
    if reasons:
        items.append((2, 3, [badges(reasons)]))
    evidence = list(getattr(diag, "evidence", ()) or ())
    if neutral:  # engine has nothing yet: lean on the rule-based insights
        evidence = [f"{i.title}" for i in a.insights][:3]
    for k, e in enumerate(evidence[:4]):
        t = _line(" ▸ ", R.GPU_GLOW)
        t.append(e, Style(color=R.MUTED))
        items.append((3 + 2 * k, 4 + k, [t]))
    for k, act in enumerate(list(getattr(diag, "actions", ()) or ())[:3]):
        t = _line(" → ", R.LEVEL_COLOR["good"], True)
        t.append(act, Style(color=R.INK))
        items.append((4 + 2 * k, 10 + k, [t]))
    tops = merged_insights(a, diag, extra)[:4]
    if tops:
        items.append((9, 30, [_line("INSIGHTS", R.MUTED, True)]))
        for k, ins in enumerate(tops):
            t = _line(f"{R.LEVEL_ICON.get(ins.level, '·')} ", R.LEVEL_COLOR.get(ins.level, R.MUTED), True)
            t.append(ins.title, Style(color=R.INK))
            items.append((10 + k, 31 + k, [t]))
    sess = session_line(getattr(diag, "session", None), g.mem_short if g else "VRAM")
    if sess is not None:
        items.append((5, 20, [sess]))
    fc = getattr(diag, "vram_forecast", None)
    if fc:
        t = _line("◔ ", R.VRAM_GLOW)
        t.append(fc, Style(color=R.VRAM_GLOW))
        items.append((4, 21, [t]))
    chosen, used = [], 0
    for item in sorted(items, key=lambda x: x[0]):
        need = len(item[2]) + (1 if item[1] in (0, 30) and height >= 9 else 0)
        if used + need <= height:
            chosen.append(item)
            used += need
    if not any(o > 30 for _, o, _ in chosen):  # never leave a dangling INSIGHTS heading
        chosen = [c for c in chosen if c[1] != 30]
    rows: list[RenderableType] = []
    for _, order, lines in sorted(chosen, key=lambda x: x[1]):
        if order == 30 and height >= 9:
            rows.append(Text())
        rows.extend(lines)
        if order == 0 and height >= 9:
            rows.append(Text())
    return Group(*rows)


def diagnosis_detail(diag: Any, a: Analysis, g: Optional[GpuSample], width: int) -> Group:
    """Diagnosis tab: everything, wrapped instead of truncated."""
    score: Optional[int] = getattr(diag, "score", None)
    bottleneck: str = getattr(diag, "bottleneck", "unknown")
    verdict = getattr(diag, "verdict", "") or "Collecting data"
    rows: list[RenderableType] = list(score_block(score, verdict, width, getattr(diag, "score_basis", None)))
    rows.append(Text())
    lvl = BOTTLENECK_LEVEL.get(bottleneck, "warning")
    t = Text("Bottleneck  ", Style(color=R.MUTED))
    t.append(f"{R.LEVEL_ICON[lvl]} {bottleneck_label(bottleneck)}", Style(color=R.LEVEL_COLOR[lvl], bold=True))
    conf = float(getattr(diag, "confidence", 0.0) or 0.0)
    t.append("   confidence ", Style(color=R.MUTED))
    t.append_text(confidence_meter(conf, 10))
    t.append(f" {conf * 100:.0f}%", Style(color=R.INK, bold=True))
    rows.append(t)
    runner = getattr(diag, "runner_up", None)
    if runner:
        ru = Text("Runner-up   ", Style(color=R.MUTED))
        ru.append(bottleneck_label(runner), Style(color=R.INK))
        rows.append(ru)
    basis = getattr(diag, "score_basis", None)
    if basis:
        sb = Text("Score basis ", Style(color=R.MUTED))
        sb.append(
            "SM activity (NVML GPM): real compute, not just kernel time"
            if basis == "sm_active"
            else "utilization (GPM counters unavailable)",
            Style(color=R.INK),
        )
        rows.append(sb)
    if a.grade:
        g_t = Text("Last hour   ", Style(color=R.MUTED))
        g_t.append(f"● {a.grade[0]}", Style(color=R.GRADE_COLOR[a.grade[1]], bold=True))
        rows.append(g_t)
    reasons = throttle_reasons(g, diag)
    th = Text("Throttling  ", Style(color=R.MUTED))
    unseen = g is not None and g.vendor == "apple"  # macOS reports no clocks or throttle reasons without root
    none = "not reported by this GPU (no clock or power readings)" if unseen else "none — clocks unconstrained"
    th.append_text(badges(reasons) if reasons else Text(none, Style(color=R.MUTED)))
    rows.append(th)

    def section(title: str, items: Sequence[str], icon: str, color: str) -> None:
        if not items:
            return
        rows.append(Text())
        rows.append(Text(title, Style(color=R.MUTED, bold=True)))
        for it in items:
            line = Text(f" {icon} ", Style(color=color, bold=True))
            line.append(it, Style(color=R.INK))
            rows.append(line)

    section("WHY", list(getattr(diag, "evidence", ()) or ()), "▸", R.GPU_GLOW)
    section("WHAT TO TRY", list(getattr(diag, "actions", ()) or ()), "→", R.LEVEL_COLOR["good"])
    mem = g.mem_short if g else "VRAM"
    extra = [x for x in (session_line(getattr(diag, "session", None), mem),) if x is not None]
    fc = getattr(diag, "vram_forecast", None)
    if extra or fc:
        rows.append(Text())
        rows.extend(extra)
        if fc:
            rows.append(Text(f"◔ {fc}", Style(color=R.VRAM_GLOW)))
    if score is None and bottleneck == "unknown":
        rows.append(Text())
        rows.append(
            Text(
                "The diagnosis firms up after a few minutes of history: it compares utilization, clocks, "
                "throttle reasons, CPU and I/O to name what is holding the GPU back.",
                Style(color=R.MUTED),
            )
        )
    return Group(*rows)


# ---- system health -------------------------------------------------------------------------
class HealthRow(NamedTuple):
    name: str
    level: str
    value: str
    detail: str


def _lvl(v: Optional[float], warn: float, bad: float, bad_level: str = "serious") -> str:
    if v is None:
        return "info"
    return bad_level if v >= bad else "warning" if v >= warn else "good"


def fallback_health(s: Sample, g: Optional[GpuSample], temp_warn: float, grade: Optional[str]) -> list[HealthRow]:
    """Rule-of-thumb health rows from the live sample, used until the engine supplies ``health``."""
    rows: list[HealthRow] = []
    if g is not None:
        u = g.util
        state = "idle" if (u or 0) < 10 else "busy" if (u or 0) >= 70 else "light load"
        rows.append(HealthRow("GPU", "info" if state == "idle" else "good", R.fmt(u, 0, "%"), grade or state))
        v = g.vram_pct
        used = f"{(g.mem_used_mib or 0) / 1024:.1f} / {(g.mem_total_mib or 0) / 1024:.0f} GB"
        if g.unified_memory:
            rows.append(HealthRow("GPU memory", _lvl(v, 85, 95), R.fmt(v, 0, "%"), f"{used} · unified"))
        else:
            rows.append(HealthRow("VRAM", _lvl(v, 85, 95), R.fmt(v, 0, "%"), used))
        t = g.temp_c
        unread = t is None and g.vendor == "apple"
        rows.append(
            HealthRow(
                "GPU temp",
                _lvl(t, temp_warn - 10, temp_warn, "critical"),
                "n/a" if unread else R.fmt(t, 0, "°C"),
                "not read on macOS" if unread else f"warn at {temp_warn:.0f}°C",
            )
        )
    hot_core = max(s.cpu_per_core, default=0.0)
    cpu_detail = f"max core {hot_core:.0f}%"
    if s.cpu_freq_mhz:
        cpu_detail += f" · {s.cpu_freq_mhz / 1000:.1f} GHz"
    if s.cpu_temp_c is not None:
        cpu_detail += f" · {s.cpu_temp_c:.0f}°C"
    rows.append(HealthRow("CPU", _lvl(s.cpu_util, 85, 97, "warning"), f"{s.cpu_util:.0f}%", cpu_detail))
    swap = f" · swap {s.swap_pct:.0f}%" if s.swap_pct is not None else ""
    ram_lvl = _lvl(s.mem_pct, 85, 95)
    if s.swap_pct is not None and s.swap_pct >= 50 and ram_lvl == "good":
        ram_lvl = "warning"
    rows.append(
        HealthRow("RAM", ram_lvl, f"{s.mem_pct:.0f}%", f"{s.mem_used / 2**30:.1f} / {s.mem_total / 2**30:.0f} GB{swap}")
    )
    for name, a_, b_, la, lb in (
        ("Disk", s.disk_read_bps, s.disk_write_bps, "R", "W"),
        ("Network", s.net_rx_bps, s.net_tx_bps, "↓", "↑"),
    ):
        if a_ is None and b_ is None:
            rows.append(HealthRow(name, "info", "n/a", "waiting for a second reading"))
        else:
            total = (a_ or 0) + (b_ or 0)
            rows.append(HealthRow(name, "good", R.fmt_rate(total), f"{la} {R.fmt_rate(a_)} · {lb} {R.fmt_rate(b_)}"))
    return rows


def health_rows(
    diag: Any, s: Sample, g: Optional[GpuSample], temp_warn: float, grade: Optional[str]
) -> list[HealthRow]:
    engine = getattr(diag, "health", ()) or ()
    if engine:
        return [
            HealthRow(
                str(getattr(h, "name", "?")),
                str(getattr(h, "level", "info")),
                str(getattr(h, "value", "") or ""),
                str(getattr(h, "detail", "") or ""),
            )
            for h in engine
        ]
    return fallback_health(s, g, temp_warn, grade)


def health_grid(rows: Sequence[HealthRow], width: int, detail: bool = True, columns: int = 1) -> Group:
    """``✔ Name   value ··· detail``: value left-aligned, detail right-aligned when it fits.

    ``columns=2`` packs two short entries per line (no detail) for panels short on height.
    """
    if columns == 2:
        col_w = (width - 2) // 2
        cells = [health_grid([r], col_w, detail=False).renderables[0] for r in rows]
        lines: list[RenderableType] = []
        for i in range(0, len(cells), 2):
            left = cells[i]
            assert isinstance(left, Text)
            line = left.copy()
            line.truncate(col_w, overflow="ellipsis", pad=True)
            if i + 1 < len(cells):
                line.append("  ")
                line.append_text(cells[i + 1])  # type: ignore[arg-type]
            lines.append(line)
        return Group(*lines)
    if columns == 1 and detail:
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True, overflow="ellipsis", ratio=3)
        grid.add_column(no_wrap=True, overflow="ellipsis", justify="right", ratio=2)
        for r in rows:
            lvl = r.level if r.level in R.LEVEL_COLOR else "info"
            bad = lvl in ("warning", "serious", "critical")
            grid.add_row(
                Text(R.LEVEL_ICON[lvl], Style(color=R.LEVEL_COLOR[lvl], bold=True)),
                Text(r.name, Style(color=R.INK)),
                Text(r.value, Style(color=R.LEVEL_COLOR[lvl] if lvl != "good" else R.INK, bold=True)),
                Text(r.detail, Style(color=R.LEVEL_COLOR[lvl] if bad else R.DIM)),
            )
        return Group(grid)
    name_w = max((len(r.name) for r in rows), default=4) + 1
    out: list[RenderableType] = []
    for r in rows:
        lvl = r.level if r.level in R.LEVEL_COLOR else "info"
        t = Text(no_wrap=True, overflow="ellipsis")
        t.append(f"{R.LEVEL_ICON[lvl]} ", Style(color=R.LEVEL_COLOR[lvl], bold=True))
        t.append(f"{r.name:<{name_w}}", Style(color=R.INK))
        avail = width - t.cell_len
        value = r.value if len(r.value) <= avail else r.value[: max(0, avail - 1)] + "…"
        t.append(value, Style(color=R.LEVEL_COLOR[lvl] if lvl != "good" else R.INK, bold=True))
        out.append(t)
    return Group(*out)


# ---- insights ------------------------------------------------------------------------------
def merged_insights(a: Analysis, diag: Any, extra: Sequence[Insight] = ()) -> list[Insight]:
    """Engine findings, ``extra`` (cross-GPU) findings and rule insights, deduplicated, worst first."""
    order = {"critical": 0, "serious": 1, "warning": 2, "info": 3, "good": 4}
    seen: set[str] = set()
    out: list[Insight] = []
    for i in list(extra) + list(getattr(diag, "findings", ()) or ()) + list(a.insights):
        key = i.title.lower()
        if key not in seen:
            seen.add(key)
            out.append(i)
    return sorted(out, key=lambda i: order.get(i.level, 5))


def insights_list(items: Sequence[Insight]) -> Group:
    rows: list[RenderableType] = []
    for i in items:
        t = Text(f"{R.LEVEL_ICON.get(i.level, '·')} ", Style(color=R.LEVEL_COLOR.get(i.level, R.MUTED), bold=True))
        t.append(i.title, Style(color=R.INK, bold=True))
        rows.append(t)
        rows.append(Text(f"  {i.detail}", Style(color=R.MUTED)))
    if not rows:
        rows.append(Text("✔ No issues in the last hour.", Style(color=R.LEVEL_COLOR["good"])))
    return Group(*rows)


# ---- sessions ------------------------------------------------------------------------------
class Run(NamedTuple):
    start_ms: int
    minutes: int
    mean: float
    cells: tuple[Optional[float], ...]


def find_sessions(
    start_ms: int, cells: Sequence[Optional[float]], busy: float = ACTIVE_UTIL, gap: int = SESSION_GAP_MIN
) -> list[Run]:
    """Contiguous active stretches (≥ ``busy`` %) in per-minute cells, bridging gaps ≤ ``gap`` min.

    Defaults match the engine's session rule (intel._session) so the History table and the
    Diagnosis session line describe the same runs.
    """
    runs: list[Run] = []
    i, n = 0, len(cells)
    while i < n:
        if (cells[i] or 0) < busy:
            i += 1
            continue
        j, last = i, i
        while j < n and j - last <= gap + 1:  # up to `gap` inactive minutes between active ones
            if (cells[j] or 0) >= busy:
                last = j
            j += 1
        seg = tuple(cells[i : last + 1])
        vals = [v for v in seg if v is not None]
        if last - i + 1 >= 3 and vals:
            runs.append(Run(start_ms + i * 60_000, last - i + 1, sum(vals) / len(vals), seg))
        i = last + 1
    return runs


def sessions_table(runs: Sequence[Run], width: int, limit: int, now: Optional[float] = None) -> RenderableType:
    if not runs:
        return Text(f"No GPU sessions (≥ 3 min above {ACTIVE_UTIL:.0f}%) in the last 24 h.", Style(color=R.MUTED))
    now = now if now is not None else time.time()
    t = Table(box=None, header_style=Style(color=R.MUTED), expand=True, pad_edge=False, padding=(0, 1))
    t.add_column("Start", no_wrap=True)
    t.add_column("Length", justify="right", no_wrap=True)
    t.add_column("Avg", justify="right", no_wrap=True)
    t.add_column("Shape", no_wrap=True, ratio=1)
    spark_w = max(6, width - 26)
    for r in list(reversed(runs))[:limit]:
        end = r.start_ms / 1000 + r.minutes * 60
        live = now - end < 120
        start = time.strftime("%H:%M", time.localtime(r.start_ms / 1000))
        h, m = divmod(r.minutes, 60)
        t.add_row(
            Text(start + (" ●" if live else ""), Style(color=R.LEVEL_COLOR["good"] if live else R.INK)),
            f"{h}h {m:02d}m" if h else f"{m}m",
            Text(f"{r.mean:.0f}%", Style(color=R.GPU_GLOW, bold=True)),
            R.sparkline(list(r.cells), spark_w, R.GPU_GLOW),
        )
    return t
