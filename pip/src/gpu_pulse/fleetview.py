"""The Overview's ALL GPUS panel: one compact row per GPU, the selected one highlighted.

Each row: selection marker, health dot, NVML index, name, a utilization bar whose bright part is
real SM activity (GPM) and dim part kernel-running-but-idle time, VRAM, temperature, power,
throttle badges and cross-GPU tags (straggler, x4 link, …). The last line names the top
cross-GPU finding and machine totals. Pure functions; the app maps clicks back to GPUs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple, Optional

from rich.console import Group
from rich.style import Style
from rich.text import Text

from . import render as R
from ._intel_gpu import POWER_REASONS, THERMAL_REASONS
from .fleet import MIN_SPAN_S, Fleet, aggregate, gpu_flag, span_text
from .models import GpuSample

SELECT_BG = "#1b2a3a"
TWO_COL_MIN = 170  # inner width needed for two GPUs per line
MAX_ROWS = 8  # GPU lines shown at once; more GPUs scroll with the selection
COL_GAP = 3


class Hit(NamedTuple):
    """Clickable area of one GPU entry, in content coordinates."""

    line: int
    x0: int
    x1: int
    pos: int  # position in the sample's GPU tuple


def fleet_columns(n: int, width: int) -> int:
    return 2 if n > 4 and width >= TWO_COL_MIN else 1


def fleet_lines(n: int, width: int, height: int) -> int:
    """GPU lines the panel shows: all of them up to MAX_ROWS, fewer on short terminals; 0 for one GPU."""
    if n <= 1:
        return 0
    room = max(1, height - 2 - 3 - 18)  # keep ~18 rows for rings + the middle block
    return max(1, min(MAX_ROWS, room, math.ceil(n / fleet_columns(n, width))))


def fleet_height(n: int, width: int, height: int = 200) -> int:
    """Outer height of the panel (border + GPU lines + summary line); 0 for a single GPU."""
    lines = fleet_lines(n, width, height)
    return lines + 3 if lines else 0


def short_name(name: str) -> str:
    for junk in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", " (simulated)"):
        name = name.replace(junk, "")
    return name


def usage_bar(util: Optional[float], sm: Optional[float], width: int) -> Text:
    """Utilization bar: bright = SM activity (when GPM reports it), dim = kernel running, SMs idle."""
    u = max(0.0, min(1.0, (util or 0.0) / 100))
    nu = round(u * width)
    ns = nu if sm is None else min(width, round(max(0.0, min(1.0, sm / 100)) * width))
    nu = max(nu, ns)
    t = Text()
    span = max(1, width - 1)
    for i in range(ns):
        t.append("█", R.style(R.mix(R.GPU, R.GPU_GLOW, i / span)))
    if nu > ns:
        t.append("▒" * (nu - ns), R.style(R.mix(R.GPU, R.BG, 0.35)))
    if width > nu:
        t.append(" " * (width - nu), R.style(bgcolor=R.TRACK))
    return t


def _tiers(width: int) -> tuple[int, int, int, bool, bool]:
    """(name width, util bar width, VRAM bar width, show VRAM GB, show power) for a column width."""
    if width >= 150:
        return 22, 20, 10, True, True
    if width >= 110:
        return 14, 14, 6, False, True
    if width >= 80:
        return 0, 10, 4, False, True
    return 0, 6, 0, False, False


def gpu_row(g: GpuSample, selected: bool, fleet: Optional[Fleet], width: int, temp_warn: float) -> Text:
    flag = gpu_flag(g, fleet, temp_warn)
    name_w, ub, vb, gb, power = _tiers(width)
    t = Text(no_wrap=True, overflow="ellipsis")
    t.append("▶ " if selected else "  ", R.style(R.GPU_GLOW, True))
    t.append("● ", R.style(R.LEVEL_COLOR.get(flag.level, R.MUTED)))
    t.append(f"{g.index:>2} ", R.style(R.INK if selected else R.MUTED, selected))
    if name_w:
        t.append(f"{short_name(g.name)[:name_w]:<{name_w}} ", R.style(R.INK, selected))
    t.append_text(usage_bar(g.util, g.sm_active, ub))
    t.append(f" {R.fmt(g.util, 0, '%'):>4}", R.style(R.INK, True))
    if g.sm_active is not None and width >= 80:
        t.append(f" SM {g.sm_active:>3.0f}%", R.style(R.GPU_GLOW))
    t.append("  VRAM ", R.style(R.MUTED))
    v = g.vram_pct
    if vb:
        t.append_text(R.bar((v or 0) / 100, vb, R.VRAM, R.VRAM_GLOW))
        t.append(" ")
    vcol = R.LEVEL_COLOR["serious"] if (v or 0) >= 95 else R.INK
    t.append(f"{R.fmt(v, 0, '%'):>4}", R.style(vcol, True))
    if gb and g.mem_total_mib:
        t.append(f" {(g.mem_used_mib or 0) / 1024:4.1f}/{g.mem_total_mib / 1024:.0f}G", R.style(R.MUTED))
    temp = g.temp_c
    tcol = (
        R.LEVEL_COLOR["critical"]
        if (temp or 0) >= temp_warn
        else R.LEVEL_COLOR["warning"]
        if (temp or 0) >= temp_warn - 10
        else R.INK
    )
    t.append(f"  {R.fmt(temp, 0, '°C'):>5}", R.style(tcol, True))
    if power:
        t.append(f"  {R.fmt(g.power_w, 0, 'W'):>5}", R.style(R.INK))
    for r in g.throttle:
        if r in THERMAL_REASONS:
            t.append(" ")
            t.append_text(R.badge("thermal", R.HOT))
        elif r in POWER_REASONS:
            t.append(" ")
            t.append_text(R.badge("pwr cap", R.WARN))
    for tag in flag.tags:
        t.append(f"  ▲ {tag}", R.style(R.WARN_GLOW))
    t.truncate(width, overflow="ellipsis", pad=True)
    if selected:
        t.stylize_before(Style(bgcolor=SELECT_BG))
    return t


def summary_line(gpus: Sequence[GpuSample], fleet: Optional[Fleet], width: int) -> Text:
    agg = aggregate(gpus)
    right = Text(no_wrap=True)
    if agg.util_mean is not None:
        right.append(f"avg {agg.util_mean:.0f}%", R.style(R.INK, True))
    right.append(f" · VRAM {agg.vram_used_gib:.0f}/{agg.vram_total_gib:.0f} GB", R.style(R.MUTED))
    if agg.power_w is not None:
        cap = f"/{agg.power_limit_w:.0f}" if agg.power_limit_w else ""
        right.append(f" · {agg.power_w:.0f}{cap} W", R.style(R.MUTED))
    left = Text(no_wrap=True, overflow="ellipsis")
    findings = fleet.findings if fleet else ()
    if findings:
        top = findings[0]
        left.append(f"{R.LEVEL_ICON.get(top.level, '▲')} ", R.style(R.LEVEL_COLOR.get(top.level, R.MUTED), True))
        left.append(top.title, R.style(R.INK, True))
        more = len(findings) - 1
        left.append(f"   {'+' + str(more) + ' more · ' if more else ''}details: 5 Diagnosis", R.style(R.DIM))
    elif fleet is not None and len(fleet.stats) >= 2:
        left.append("✔ ", R.style(R.LEVEL_COLOR["good"], True))
        span = span_text(min(s.span_s for s in fleet.stats))
        left.append(f"Balanced: no cross-GPU issues in the last {span}", R.style(R.MUTED))
    else:
        left.append(f"Comparing GPUs after {MIN_SPAN_S:.0f} s of samples…", R.style(R.DIM))
    room = width - right.cell_len - 2
    if room < 20:  # too narrow for both: the finding matters more than the totals
        left.truncate(width, overflow="ellipsis")
        return left
    left.truncate(room, overflow="ellipsis", pad=True)
    left.append("  ")
    left.append_text(right)
    return left


def fleet_panel(
    gpus: Sequence[GpuSample],
    selected: int,
    fleet: Optional[Fleet],
    width: int,
    temp_warn: float = 80.0,
    max_lines: int = MAX_ROWS,
) -> tuple[Group, list[Hit]]:
    """The panel body and the clickable area of every GPU entry shown."""
    n = len(gpus)
    cols = fleet_columns(n, width)
    lines_total = math.ceil(n / cols)
    shown = max(1, min(max_lines, lines_total))
    sel_line = selected // cols if cols == 2 else selected
    first = 0 if lines_total <= shown else max(0, min(lines_total - shown, sel_line - shown + 1))
    col_w = (width - COL_GAP) // 2 if cols == 2 else width
    rows: list[Text] = []
    hits: list[Hit] = []
    for line in range(first, first + shown):
        out = Text(no_wrap=True, overflow="crop")
        for c in range(cols):
            pos = line * cols + c
            if pos >= n:
                break
            x0 = c * (col_w + COL_GAP)
            if c:
                out.append(" " * COL_GAP)
            out.append_text(gpu_row(gpus[pos], pos == selected, fleet, col_w, temp_warn))
            hits.append(Hit(line - first, x0, x0 + col_w, pos))
        rows.append(out)
    summary = summary_line(gpus, fleet, width)
    if first or first + shown < lines_total:
        summary = Text(
            f"GPUs {first * cols}–{min(n, (first + shown) * cols) - 1} of {n} · g/G to scroll  ", R.style(R.DIM)
        )
        summary.append_text(summary_line(gpus, fleet, max(20, width - summary.cell_len)))
    rows.append(summary)
    return Group(*rows), hits
