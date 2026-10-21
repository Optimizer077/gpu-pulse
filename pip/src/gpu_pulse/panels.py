"""Chart and fact panels: timeline, usage windows, I/O, CPU, processes, GPU architecture, specs.

Pure functions from data to Rich renderables sized for the panel they land in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from rich.console import Group, RenderableType
from rich.style import Style
from rich.table import Table
from rich.text import Text

from . import anim
from . import render as R
from .models import GpuProcess, GpuSample, Sample, Specs, WindowStats

NA = "–"


def kv(label: str, value: str, width: int = 12, color: str = R.INK) -> Text:
    t = Text(f"{label:<{width}}", Style(color=R.MUTED), no_wrap=True, overflow="ellipsis")
    t.append(value, Style(color=color, bold=True))
    return t


def legend(*items: tuple[str, str, str]) -> Text:
    """(swatch colour, label, value) items on one line."""
    t = Text(no_wrap=True, overflow="ellipsis")
    for color, label, value in items:
        t.append("■ ", Style(color=color))
        t.append(f"{label} ", Style(color=R.MUTED))
        t.append(f"{value}   ", Style(color=R.INK, bold=True))
    return t


# ---- GPU timeline --------------------------------------------------------------------------
def _spark_row(label: str, values: list[Optional[float]], width: int, color: str) -> Text:
    t = Text(f"{label:>4} ", Style(color=R.MUTED), no_wrap=True)
    t.append_text(R.sparkline(values, max(1, width - 5), color))
    return t


def timeline(series: dict[str, list[Optional[float]]], g: Optional[GpuSample], width: int, height: int) -> Group:
    """GPU utilization area chart, with VRAM / SM-active / Tensor lines underneath when known."""
    extra = [
        (label, key, color)
        for label, key, color in (("SM", "sm", R.mix(R.GPU_GLOW, "#ffffff", 0.35)), ("TNS", "tensor", R.TENSOR_GLOW))
        if any(v is not None for v in series.get(key, ()))
    ]
    extra = [("VRAM", "vram", R.VRAM_GLOW)] + extra
    extra = extra[: max(0, height - 4)]
    chart_h = max(2, height - 1 - len(extra))
    items = [(R.GPU_GLOW, "util", R.fmt(g.util if g else None, 0, "%"))]
    if g is not None and g.sm_active is not None:
        items.append((R.GPU_GLOW, "SM", R.fmt(g.sm_active, 0, "%")))
    if g is not None and g.tensor_active is not None:
        items.append((R.TENSOR_GLOW, "tensor", R.fmt(g.tensor_active, 0, "%")))
    items.append((R.VRAM_GLOW, "VRAM", R.fmt(g.vram_pct if g else None, 0, "%")))
    rows: list[RenderableType] = [legend(*items), R.area_chart(series["util"], width, chart_h, R.GPU, R.GPU_GLOW)]
    rows += [_spark_row(label, series[key], width, color) for label, key, color in extra]
    return Group(*rows)


# ---- per-metric traces (GPU tab) -----------------------------------------------------------
TRACE_METRICS = (
    ("util", "Utilization", R.GPU, R.GPU_GLOW),
    ("sm", "SM active", R.GPU, R.mix(R.GPU_GLOW, "#ffffff", 0.35)),
    ("tensor", "Tensor cores", R.TENSOR, R.TENSOR_GLOW),
    ("dram", "DRAM bandwidth", R.VRAM, R.VRAM_GLOW),
    ("power", "Power (% of cap)", R.POWER, R.POWER_GLOW),
)


def gpu_traces(series: dict[str, list[Optional[float]]], width: int, height: int) -> Group:
    """Stacked braille traces on a fixed 0–100% scale, so heights compare across metrics."""
    shown = [m for m in TRACE_METRICS if any(v is not None for v in series.get(m[0], ()))]
    if not shown:
        return Group(Text("Collecting samples…", Style(color=R.MUTED)))
    per = max(2, height // len(shown))
    rows: list[RenderableType] = []
    for key, label, c, gl in shown:
        vals = series[key]
        known = [v for v in vals if v is not None]
        head = Text(f"{label:<17}", Style(color=R.INK, bold=True), no_wrap=True, overflow="ellipsis")
        head.append(f"now {R.fmt(known[-1] if known else None, 0, '%'):>4}", Style(color=gl, bold=True))
        if known:
            head.append(f"   avg {sum(known) / len(known):.0f}%   peak {max(known):.0f}%", Style(color=R.MUTED))
        rows.append(head)
        rows.append(anim.trace(vals, width, per - 1, c, gl, max_v=100.0))
    return Group(*rows)


# ---- true usage (NVML GPM) -----------------------------------------------------------------
TRUE_METRICS = (
    ("SM active", "sm_active", R.GPU, R.GPU_GLOW),
    ("SM occupancy", "sm_occupancy", R.GPU, R.GPU_GLOW),
    ("Tensor cores", "tensor_active", R.TENSOR, R.TENSOR_GLOW),
    ("DRAM bandwidth", "dram_bw_util", R.VRAM, R.VRAM_GLOW),
    ("FP32 pipe", "fp32_active", R.TEMP, R.TEMP_GLOW),
    ("FP16 pipe", "fp16_active", R.TEMP, R.TEMP_GLOW),
    ("FP64 pipe", "fp64_active", R.TEMP, R.TEMP_GLOW),
    ("INT pipe", "int_active", R.TEMP, R.TEMP_GLOW),
)
TRUE_TWO_COL_MIN = 36
SHORT = {"SM active": "SM", "SM occupancy": "Occup.", "Tensor cores": "Tensor", "DRAM bandwidth": "DRAM"}


def divergence(g: Optional[GpuSample]) -> Optional[tuple[str, str]]:
    """(level, message) when utilization and real SM activity tell different stories."""
    if g is None or g.util is None or g.sm_active is None:
        return None
    if g.util >= 50 and g.util - g.sm_active >= 40:
        return "warning", f"Util {g.util:.0f}% but SM only {g.sm_active:.0f}%: tiny or stalled kernels"
    if g.util >= 50 and g.sm_active >= 0.6 * g.util:
        return "good", f"SM {g.sm_active:.0f}% active: real compute load"
    return None


def true_usage(g: Optional[GpuSample], width: int, height: int, columns: int = 1) -> Group:
    """GPM gauges in 1 or 2 columns, a one-line explanation and a divergence callout."""
    vals = [(name, getattr(g, attr, None), c, gl) for name, attr, c, gl in TRUE_METRICS]
    if g is None or all(v is None for _, v, _, _ in vals):
        return Group(
            Text("True-usage counters not available on this GPU/driver (needs NVML GPM).", Style(color=R.MUTED)),
            Text(),
            Text("Utilization only says a kernel was running, not how much of the GPU it used.", Style(color=R.DIM)),
        )
    shown = [x for x in vals if x[1] is not None]
    cols = 2 if columns == 2 and width >= TRUE_TWO_COL_MIN else 1
    col_w = (width - (cols - 1) * 2) // cols
    label_w = 7 if cols == 2 else 15
    bw = max(4, col_w - label_w - 5)
    cells: list[Text] = []
    for name, v, c, gl in shown:
        label = SHORT.get(name, name.replace(" pipe", "")) if cols == 2 else name
        t = Text(f"{label:<{label_w}}", Style(color=R.MUTED), no_wrap=True, overflow="crop")
        t.append_text(R.bar((v or 0) / 100, bw, c, gl))
        t.append(f"{v:>4.0f}%", Style(color=R.INK, bold=True))
        cells.append(t)
    rows: list[RenderableType] = []
    for i in range(0, len(cells), cols):
        line = Text(no_wrap=True, overflow="crop")
        for k, cell in enumerate(cells[i : i + cols]):
            if k:
                line.append("  ")
            cell.truncate(col_w, pad=True)
            line.append_text(cell)
        rows.append(line)
    budget = height - len(rows)
    div = divergence(g)
    if div and budget >= 1:
        lvl, msg = div
        t = Text(f"{R.LEVEL_ICON[lvl]} ", Style(color=R.LEVEL_COLOR[lvl], bold=True), no_wrap=True, overflow="ellipsis")
        t.append(msg, Style(color=R.WARN_GLOW if lvl == "warning" else R.INK, bold=lvl == "warning"))
        rows.insert(0, t)
        budget -= 1
    if budget >= 1:
        rows.append(
            Text(
                "util = any kernel running · SM = GPU actually computing",
                Style(color=R.DIM),
                no_wrap=True,
                overflow="ellipsis",
            )
        )
    return Group(*rows)


# ---- usage windows -------------------------------------------------------------------------
def _stack(w: WindowStats, width: int) -> Text:
    idle, busy = w.idle_share or 0.0, w.busy_share or 0.0
    mid = max(0.0, 1 - idle - busy)
    t = Text(no_wrap=True)
    if w.util is None:
        t.append("▬" * width, Style(color=R.TRACK))
        return t
    a, b = round(idle * width), round((idle + mid) * width)
    t.append("▬" * a, Style(color=R.DIM))
    t.append("▬" * (b - a), Style(color=R.GPU))
    t.append("▬" * (width - b), Style(color=R.GPU_GLOW))
    return t


def usage_strip(
    windows: Sequence[WindowStats], spark_cells: Sequence[Sequence[Optional[float]]], width: int, height: int = 4
) -> Table:
    """Four compact columns: label + mean, idle/busy stack, peak + busy share, per-minute chart."""
    t = Table.grid(expand=True, padding=(0, 2))
    for _ in windows:
        t.add_column(ratio=1, no_wrap=True)
    col_w = max(8, (width - 2 * (len(windows) - 1)) // max(1, len(windows)) - 1)
    chart_h = max(1, height - 3)
    heads, stacks, facts, sparks = [], [], [], []
    for w, cells in zip(windows, spark_cells):
        h = Text(f"{w.label.upper():<9}", Style(color=R.MUTED), no_wrap=True, overflow="ellipsis")
        h.append(R.fmt(w.util, 1, "%"), Style(color=R.INK, bold=True))
        heads.append(h)
        stacks.append(_stack(w, col_w))
        f = Text(f"peak {R.fmt(w.util_peak, 0, '%')}", Style(color=R.MUTED), no_wrap=True, overflow="ellipsis")
        busy = f"busy {(w.busy_share or 0) * 100:.0f}%" if w.util is not None else ""
        if busy and col_w >= len("peak 100% busy 100%"):  # same columns on every card, or none
            f.append(" " * (col_w - len(f.plain) - len(busy)) + busy, Style(color=R.MUTED))
        facts.append(f)
        sparks.append(
            R.sparkline(list(cells), col_w, R.GPU_GLOW)
            if chart_h == 1
            else R.area_chart(list(cells), col_w, chart_h, R.GPU, R.GPU_GLOW, axis=False)
        )
    for row in (heads, stacks, facts, sparks):
        t.add_row(*row)
    return t


def window_card(w: WindowStats, cells: Sequence[Optional[float]], width: int, height: int = 10) -> Group:
    big = Text(R.fmt(w.util, 1), Style(color=R.INK, bold=True))
    big.append(" % avg", Style(color=R.MUTED))
    idle, busy = w.idle_share or 0.0, w.busy_share or 0.0
    held = (w.held_idle_share or 0) * 100 if w.held_idle_share is not None else None
    return Group(
        big,
        Text(f"peak {R.fmt(w.util_peak)}% · VRAM {R.fmt(w.vram)}%", Style(color=R.MUTED), no_wrap=True),
        _stack(w, width),
        Text(f"idle {idle * 100:.0f}%".ljust(max(0, width - 9)) + f"busy {busy * 100:.0f}%", Style(color=R.MUTED)),
        kv("Avg power", f"{R.fmt(w.power)} W"),
        kv("Peak temp", f"{R.fmt(w.temp_peak)} °C"),
        kv("Held idle", f"{R.fmt(held)}%"),
        kv("CPU / RAM", f"{R.fmt(w.cpu)}% / {R.fmt(w.mem)}%"),
        R.sparkline(list(cells), width, R.GPU_GLOW)
        if height <= 10
        else R.area_chart(list(cells), width, height - 9, R.GPU, R.GPU_GLOW, axis=False),
        Text(f"{w.covered_minutes}/{w.minutes} min recorded", Style(color=R.DIM)),
    )


# ---- I/O -----------------------------------------------------------------------------------
def io_block(recent: Sequence[Sample], s: Sample, width: int, height: int) -> Group:
    """Disk and network throughput: header line + mirrored braille trace per device class."""
    groups = (
        ("DISK", "read", "write", [x.disk_read_bps for x in recent], [x.disk_write_bps for x in recent],
         s.disk_read_bps, s.disk_write_bps, (R.DISK, R.DISK_GLOW), (R.mix(R.DISK, R.BG, 0.2), R.DISK)),
        ("NET", "↓ rx", "↑ tx", [x.net_rx_bps for x in recent], [x.net_tx_bps for x in recent],
         s.net_rx_bps, s.net_tx_bps, (R.NET, R.NET_GLOW), (R.mix(R.NET, R.BG, 0.2), R.NET)),
    )  # fmt: skip
    per = max(1, height // 2)
    rows: list[RenderableType] = []
    for name, la, lb, a_vals, b_vals, a_now, b_now, ca, cb in groups:
        head = Text(f"{name:<5}", Style(color=R.INK, bold=True), no_wrap=True, overflow="ellipsis")
        head.append(f"{la} ", Style(color=ca[1]))
        head.append(f"{R.fmt_rate(a_now):>10}", Style(color=R.INK, bold=True))
        head.append(f"   {lb} ", Style(color=cb[1]))
        head.append(f"{R.fmt_rate(b_now):>10}", Style(color=R.INK, bold=True))
        both = [v for v in a_vals + b_vals if v is not None]
        if both and width >= 50:
            head.append(f"   peak {R.fmt_rate(max(both))}", Style(color=R.DIM))
        rows.append(head)
        if per >= 3:
            rows.append(anim.mirror(a_vals, b_vals, width, per - 1, ca, cb))
        elif per == 2:
            total = [None if x is None and y is None else (x or 0) + (y or 0) for x, y in zip(a_vals, b_vals)]
            rows.append(anim.trace(total, width, 1, ca[0], ca[1]))
    return Group(*rows)


# ---- CPU & memory --------------------------------------------------------------------------
def cpu_block(recent: Sequence[Sample], s: Sample, width: int, height: int, threads: Optional[int]) -> Group:
    extras = []
    if s.cpu_freq_mhz:
        extras.append(f"{s.cpu_freq_mhz / 1000:.2f} GHz")
    if s.cpu_temp_c is not None:
        extras.append(f"{s.cpu_temp_c:.0f}°C")
    head = legend(
        (R.CPU_GLOW, "CPU", f"{s.cpu_util:.0f}%"),
        (R.RAM_GLOW, "RAM", f"{s.mem_pct:.0f}%  {s.mem_used / 2**30:.1f}/{s.mem_total / 2**30:.1f} GB"),
    )
    if extras:
        head.append(" · ".join(extras), Style(color=R.MUTED))
    cores_h = 3 if height >= 14 else 2
    chart_h = max(2, height - cores_h - 4)
    per_core = [v for c in s.cpu_per_core for v in (c, c, None)][:-1]
    ram = Text(" RAM ", Style(color=R.MUTED), no_wrap=True)
    ram.append_text(R.bar(s.mem_pct / 100, max(4, width - 18), R.RAM, R.RAM_GLOW))
    ram.append(f"  swap {R.fmt(s.swap_pct, 0, '%')}", Style(color=R.MUTED))
    hot = max(s.cpu_per_core, default=0.0)
    return Group(
        head,
        R.area_chart([x.cpu_util for x in recent], width, chart_h, R.CPU, R.CPU_GLOW),
        Text(
            f"per-core load · {len(s.cpu_per_core)} logical"
            + (f" of {threads}" if threads else "")
            + f" · hottest {hot:.0f}%",
            Style(color=R.MUTED),
            no_wrap=True,
        ),
        R.area_chart(per_core, min(width, max(1, len(per_core))), cores_h, R.CPU, R.CPU_GLOW, axis=False),
        ram,
    )


# ---- GPU processes -------------------------------------------------------------------------
def procs_table(procs: Sequence[GpuProcess], total_mib: Optional[float], width: int, limit: int) -> RenderableType:
    t = Table(box=None, header_style=Style(color=R.MUTED), expand=True, pad_edge=False, padding=(0, 1))
    t.add_column("Process", ratio=3, no_wrap=True, overflow="ellipsis")
    t.add_column("PID", justify="right", no_wrap=True)
    t.add_column("VRAM", justify="right", no_wrap=True)
    bar_w = max(4, min(24, width // 4))
    t.add_column("share", no_wrap=True, width=bar_w)
    for p in sorted(procs, key=lambda p: -(p.used_mib or -1))[:limit]:
        frac = (p.used_mib or 0) / total_mib if total_mib and p.used_mib is not None else None
        t.add_row(
            p.name,
            str(p.pid),
            "n/a" if p.used_mib is None else f"{p.used_mib:,.0f} MiB",
            R.bar(frac, bar_w, R.VRAM, R.VRAM_GLOW) if frac is not None else Text("·" * bar_w, Style(color=R.DIM)),
        )
    return t


# ---- GPU architecture & live detail --------------------------------------------------------
def _f(v: Any, fmt: str = "{}", none: str = NA) -> str:
    return none if v is None else fmt.format(v)


def _onoff(v: Optional[bool]) -> str:
    return NA if v is None else ("on" if v else "off")


def arch_summary(d: Any, g: Optional[GpuSample]) -> Text:
    """One line: ``Blackwell · CC 12.0 · 4608 CUDA cores · 5th-gen Tensor · 448 GB/s · 23.7 TFLOPS``."""
    parts = [
        getattr(d, "architecture", None),
        _f(getattr(d, "compute_capability", None), "CC {}", ""),
        _f(getattr(d, "cuda_cores", None), "{:,} CUDA cores", ""),
        _f(getattr(d, "tensor_core_gen", None), "{} Tensor", ""),
        _f(getattr(d, "memory_bandwidth_gbs", None), "{:.0f} GB/s", ""),
        _f(getattr(d, "fp32_tflops", None), "{:.1f} TFLOPS FP32", ""),
    ]
    t = Text(no_wrap=True, overflow="ellipsis")
    shown = [p for p in parts if p]
    if not shown:
        t.append(g.name if g else "GPU details loading…", Style(color=R.MUTED))
        return t
    for i, p in enumerate(shown):
        if i:
            t.append(" · ", Style(color=R.DIM))
        t.append(p, Style(color=R.GPU_GLOW if i == 0 else R.INK, bold=i == 0))
    return t


def arch_card(d: Any, g: Optional[GpuSample], width: int) -> Group:
    """Grouped static facts; the PCIe link turns amber when running below its maximum."""
    lw = 14
    rows: list[RenderableType] = []
    blocks: list[list[RenderableType]] = []

    def section(title: str, items: list[tuple[str, str, str]]) -> None:
        items = [x for x in items if x[1] not in (NA, "")]
        if items:
            block: list[RenderableType] = [Text(title, Style(color=R.GPU_GLOW, bold=True))]
            block.extend(kv(a, b, lw, c) for a, b, c in items)
            blocks.append(block)

    name = getattr(d, "name", None) or (g.name if g else None)
    if name:
        rows.append(Text(name, Style(color=R.INK, bold=True)))
        part = getattr(d, "board_part", None)
        if part:
            rows.append(Text(f"board {part}", Style(color=R.MUTED)))
    section(
        "COMPUTE",
        [
            ("Architecture", _f(getattr(d, "architecture", None)), R.INK),
            ("Compute cap.", _f(getattr(d, "compute_capability", None)), R.INK),
            ("CUDA cores", _f(getattr(d, "cuda_cores", None), "{:,}"), R.INK),
            ("SMs", _f(getattr(d, "sm_count", None)), R.INK),
            ("Tensor cores", _f(getattr(d, "tensor_core_gen", None)), R.INK),
            ("FP32 peak", _f(getattr(d, "fp32_tflops", None), "{:.1f} TFLOPS"), R.INK),
            ("Max SM clock", _f(getattr(d, "max_sm_clock_mhz", None), "{:,.0f} MHz"), R.INK),
        ],
    )
    vram = getattr(d, "vram_total_mib", None) or (g.mem_total_mib if g else None)
    section(
        "MEMORY",
        [
            ("VRAM", _f(vram, "{:,.0f} MiB") if vram is None else f"{vram / 1024:.1f} GB", R.INK),
            ("Bus width", _f(getattr(d, "memory_bus_bits", None), "{}-bit"), R.INK),
            ("Bandwidth", _f(getattr(d, "memory_bandwidth_gbs", None), "{:.0f} GB/s"), R.INK),
            ("Max mem clock", _f(getattr(d, "max_mem_clock_mhz", None), "{:,.0f} MHz"), R.INK),
            ("BAR1", _f(getattr(d, "bar1_total_mib", None), "{:,.0f} MiB"), R.INK),
        ],
    )
    cur_gen, cur_w = (g.pcie_gen, g.pcie_width) if g else (None, None)
    max_gen, max_w = getattr(d, "pcie_max_gen", None), getattr(d, "pcie_max_width", None)
    below = bool(cur_gen and max_gen and cur_gen < max_gen) or bool(cur_w and max_w and cur_w < max_w)
    link = f"Gen{cur_gen} x{cur_w}" if cur_gen else NA
    section(
        "LINK",
        [
            (
                "PCIe now",
                link + ("  ▲ below max" if below else ""),
                R.WARN_GLOW if below else R.INK,
            ),
            ("PCIe max", f"Gen{max_gen} x{max_w}" if max_gen else NA, R.INK),
        ],
    )
    section(
        "POWER",
        [
            ("Current cap", _f(g.power_limit_w if g else None, "{:.0f} W"), R.INK),
            ("Default", _f(getattr(d, "power_default_w", None), "{:.0f} W"), R.INK),
            (
                "Range",
                f"{getattr(d, 'power_min_w', 0) or 0:.0f} – {getattr(d, 'power_max_w', 0) or 0:.0f} W"
                if getattr(d, "power_max_w", None)
                else NA,
                R.INK,
            ),
        ],
    )
    section(
        "BOARD",
        [
            ("VBIOS", _f(getattr(d, "vbios", None)), R.INK),
            ("Driver", _f(getattr(d, "driver", None)), R.INK),
            ("CUDA driver", _f(getattr(d, "cuda_driver", None)), R.INK),
            ("Fans", _f(getattr(d, "num_fans", None)), R.INK),
            ("Persistence", _onoff(getattr(d, "persistence_mode", None)), R.INK),
            ("Compute mode", _f(getattr(d, "compute_mode", None)), R.INK),
            ("ECC", _onoff(getattr(d, "ecc_enabled", None)), R.INK),
            ("MIG", _onoff(getattr(d, "mig_enabled", None)), R.INK),
            ("Display", _onoff(getattr(d, "display_active", None)), R.INK),
        ],
    )
    if d is None:
        rows.append(Text("Reading architecture details…", Style(color=R.MUTED)))
    if rows:
        rows.append(Text())
    if width >= 56 and len(blocks) > 1:  # two balanced columns of sections
        total = sum(len(b) + 1 for b in blocks)
        split, acc = 1, len(blocks[0]) + 1
        while split < len(blocks) - 1 and acc + len(blocks[split]) + 1 <= total / 2 + 2:
            acc += len(blocks[split]) + 1
            split += 1
        cols: list[list[RenderableType]] = [[], []]
        for i, b in enumerate(blocks):
            col = cols[0 if i < split else 1]
            col.extend(([Text()] if col else []) + b)
        grid = Table.grid(padding=(0, 4), expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(Group(*cols[0]), Group(*cols[1]))
        rows.append(grid)
    else:
        for i, b in enumerate(blocks):
            rows.extend(([Text()] if i else []) + b)
    return Group(*rows)


def gpu_live(g: Optional[GpuSample], width: int) -> Group:
    """Live clocks, buses, codecs and PCIe throughput as labelled gauges."""
    if g is None:
        return Group(Text("No supported GPU detected.", Style(color=R.MUTED)))
    bw = max(6, width - 24)

    def row(label: str, frac: Optional[float], value: str, color: str = R.GPU, glow: str = R.GPU_GLOW) -> Text:
        t = Text(f"{label:<9}", Style(color=R.MUTED), no_wrap=True, overflow="ellipsis")
        t.append_text(R.bar(frac, bw, color, glow))
        t.append(f" {value}", Style(color=R.INK, bold=True))
        return t

    rows: list[RenderableType] = [
        row("SM clock", (g.clock_mhz or 0) / (g.clock_max_mhz or 1), f"{R.fmt(g.clock_mhz)} MHz"),
        row("Mem bus", (g.mem_util or 0) / 100, f"{R.fmt(g.mem_util)}%"),
        row("Fan", (g.fan_pct or 0) / 100, f"{R.fmt(g.fan_pct)}%", R.TEMP, R.TEMP_GLOW),
        row("Encoder", (g.enc_util or 0) / 100, R.fmt(g.enc_util, 0, "%"), R.CPU, R.CPU_GLOW),
        row("Decoder", (g.dec_util or 0) / 100, R.fmt(g.dec_util, 0, "%"), R.CPU, R.CPU_GLOW),
    ]
    pcie = Text(f"{'PCIe':<9}", Style(color=R.MUTED), no_wrap=True, overflow="ellipsis")
    pcie.append("↑ tx ", Style(color=R.NET_GLOW))
    pcie.append(R.fmt_rate(g.pcie_tx_bps), Style(color=R.INK, bold=True))
    pcie.append("   ↓ rx ", Style(color=R.NET_GLOW))
    pcie.append(R.fmt_rate(g.pcie_rx_bps), Style(color=R.INK, bold=True))
    rows.append(pcie)
    facts = " · ".join(
        x
        for x in [
            g.pstate,
            f"PCIe Gen{g.pcie_gen} x{g.pcie_width}" if g.pcie_gen else None,
            f"mem {R.fmt(g.mem_clock_mhz)} MHz" if g.mem_clock_mhz else None,
            (f"{g.energy_j / 3.6e6:.2f} kWh" if g.energy_j >= 3.6e6 else f"{g.energy_j / 3600:.0f} Wh")
            + " since driver load"
            if g.energy_j
            else None,
        ]
        if x
    )
    rows.append(Text(facts, Style(color=R.MUTED), no_wrap=True, overflow="ellipsis"))
    return Group(*rows)


# ---- machine specs -------------------------------------------------------------------------
def specs_block(sp: Specs) -> Group:
    days, rem = divmod(int(sp.uptime_s), 86400)
    w = 9
    return Group(
        kv("Host", f"{sp.hostname} ({sp.arch})", w),
        kv("OS", sp.os, w),
        kv("CPU", sp.cpu_model, w),
        kv(
            "",
            f"{sp.cpu_cores} cores · {sp.cpu_threads} threads" + (f" · {sp.cpu_max_ghz} GHz" if sp.cpu_max_ghz else ""),
            w,
        ),
        kv("Memory", f"{sp.mem_total / 2**30:.1f} GB", w),
        *[kv("GPU", name, w) for name in sp.gpus],
        kv("Driver", " · ".join(x for x in [sp.driver, f"CUDA {sp.cuda}" if sp.cuda else None] if x) or NA, w),
        *[kv("Disk", d, w) for d in sp.disks],
        kv("Uptime", f"{days}d {rem // 3600}h {rem % 3600 // 60}m", w),
    )
