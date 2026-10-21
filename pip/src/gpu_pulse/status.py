"""``gpu-pulse status``: a one-shot snapshot printed with Rich (no full-screen UI)."""

from __future__ import annotations

import time
from typing import Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from . import render as R
from .models import GpuSample, Sample, WindowStats
from .monitor import Analysis, Monitor


def _metric(label: str, frac: Optional[float], value: str, color: str, glow: str, width: int) -> Text:
    t = Text(f"{label:<6}", Style(color=R.MUTED))
    t.append_text(R.bar(frac, width, color, glow))
    t.append(f" {value}", Style(color=R.INK, bold=True))
    return t


def _bar_width(console_width: int) -> int:
    """Two metrics per row: label (6) + bar + value (~24) each, inside a padded panel."""
    return max(6, min(28, (console_width - 8) // 2 - 30))


def gpu_block(g: GpuSample, hour: Optional[WindowStats], bw: int) -> Group:
    head = Text(f"GPU {g.index}  ", Style(color=R.MUTED))
    head.append(g.name, Style(color=R.INK, bold=True))
    extra = " · ".join(x for x in [g.pstate, f"PCIe {g.pcie_gen}.0 x{g.pcie_width}" if g.pcie_gen else None] if x)
    if extra:
        head.append(f"   {extra}", Style(color=R.MUTED))
    if hour and hour.util is not None:
        head.append(f"   1h avg {hour.util:.0f}%", Style(color=R.GPU_GLOW))
    vram = g.vram_pct
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_row(
        _metric("Util", (g.util or 0) / 100, f"{R.fmt(g.util)}%", R.GPU, R.GPU_GLOW, bw),
        _metric(
            "VRAM",
            (vram or 0) / 100,
            f"{R.fmt(vram)}% {(g.mem_used_mib or 0) / 1024:.1f}/{(g.mem_total_mib or 0) / 1024:.0f}G",
            R.VRAM,
            R.VRAM_GLOW,
            bw,
        ),
    )
    power_frac = (g.power_w or 0) / g.power_limit_w if g.power_limit_w else 0
    grid.add_row(
        _metric("Temp", (g.temp_c or 0) / 100, f"{R.fmt(g.temp_c)}°C", R.RAM, R.RAM_GLOW, bw),
        _metric("Power", power_frac, f"{R.fmt(g.power_w)}/{R.fmt(g.power_limit_w)}W", R.CPU, R.CPU_GLOW, bw),
    )
    return Group(head, grid)


def windows_table(windows: tuple[WindowStats, ...]) -> Table:
    t = Table(box=box.SIMPLE_HEAD, header_style=Style(color=R.MUTED), pad_edge=False)
    for col in ("Window", "GPU avg", "Peak", "Idle", "Busy", "Held idle", "CPU/RAM", "Recorded"):
        t.add_column(col, justify="left" if col == "Window" else "right", no_wrap=True)

    def pct(x: Optional[float]) -> str:
        return "–" if x is None else f"{x * 100:.0f}%"

    for w in windows:
        t.add_row(
            w.label,
            Text(R.fmt(w.util, 1, "%"), Style(color=R.GPU_GLOW, bold=True)),
            R.fmt(w.util_peak, 0, "%"),
            pct(w.idle_share),
            pct(w.busy_share),
            pct(w.held_idle_share),
            f"{R.fmt(w.cpu)}/{R.fmt(w.mem)}%",
            f"{w.covered_minutes}/{w.minutes}m",
        )
    return t


def insights_block(a: Analysis) -> Group:
    rows = []
    for i in a.insights:
        t = Text(f" {R.LEVEL_ICON[i.level]} ", Style(color=R.LEVEL_COLOR[i.level], bold=True))
        t.append(i.title, Style(color=R.INK, bold=True))
        t.append(f"  {i.detail}", Style(color=R.MUTED))
        rows.append(t)
    return Group(*rows) if rows else Group(Text(" ✔ No issues in the last hour.", Style(color=R.LEVEL_COLOR["good"])))


def run_status(mon: Monitor, console: Optional[Console] = None) -> None:
    console = console or Console()
    bw = _bar_width(console.width)
    mon.tick()
    time.sleep(0.6)  # CPU % needs two readings
    s: Sample = mon.tick()
    sp = mon.specs()
    first = s.gpus[0].uuid if s.gpus else None
    a = mon.analysis(first)

    parts: list = []
    for g in s.gpus:
        parts += [gpu_block(g, a.windows[2] if g.uuid == first else None, bw), Text()]
    if not s.gpus:
        parts.append(Text("No supported GPU detected (NVIDIA via NVML, or AMD amdgpu on Linux).", Style(color=R.MUTED)))
    cpu_ram = Table.grid(padding=(0, 2))
    cpu_ram.add_column(no_wrap=True)
    cpu_ram.add_column(no_wrap=True)
    cpu_ram.add_row(
        _metric("CPU", s.cpu_util / 100, f"{s.cpu_util:.0f}% ×{sp.cpu_threads}", R.CPU, R.CPU_GLOW, bw),
        _metric(
            "RAM",
            s.mem_pct / 100,
            f"{s.mem_pct:.0f}% {s.mem_used / 2**30:.1f}/{s.mem_total / 2**30:.0f}G",
            R.RAM,
            R.RAM_GLOW,
            bw,
        ),
    )
    parts += [cpu_ram, Text()]
    if a.grade:
        verdict = Text("Last hour: ", Style(color=R.MUTED))
        verdict.append(f"● {a.grade[0]}", Style(color=R.GRADE_COLOR[a.grade[1]], bold=True))
        parts.append(verdict)
    parts.append(windows_table(a.windows))
    _, cells = mon.history.heatmap(first, hours=1)
    spark = Text("Last 60 min  ", Style(color=R.MUTED))
    spark.append_text(R.sparkline(cells, min(60, console.width - 20), R.GPU_GLOW))
    parts += [spark, Text(), insights_block(a)]
    if mon.processes:
        parts.append(Text())
        pt = Table(box=box.SIMPLE_HEAD, header_style=Style(color=R.MUTED), pad_edge=False)
        pt.add_column("Process", no_wrap=True, overflow="ellipsis")
        pt.add_column("PID", justify="right")
        pt.add_column("VRAM", justify="right")
        for p in sorted(mon.processes, key=lambda p: -(p.used_mib or -1))[:8]:
            pt.add_row(p.name, str(p.pid), "n/a" if p.used_mib is None else f"{p.used_mib:.0f} MiB")
        parts.append(pt)

    title = Text("GPU PULSE", Style(color=R.GPU_GLOW, bold=True))
    title.append(f"  {sp.hostname} · {sp.os}", Style(color=R.MUTED))
    sub = (
        f"{sp.cpu_model} · {sp.mem_total / 2**30:.0f} GB RAM"
        + (f" · driver {sp.driver}" if sp.driver else "")
        + (f" · CUDA {sp.cuda}" if sp.cuda else "")
    )
    console.print(
        Panel(
            Group(*parts),
            title=title,
            title_align="left",
            subtitle=Text(sub, Style(color=R.MUTED)),
            subtitle_align="left",
            border_style=Style(color="#2a3340"),
            padding=(1, 2),
        )
    )
