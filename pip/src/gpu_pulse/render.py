"""Rich renderables for the terminal: gradient bars, block-character charts, heatmap.

Shared by the one-shot ``status`` command and the live Textual dashboard.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime
from typing import Optional

from rich.color import Color
from rich.style import Style
from rich.text import Text

# Same validated series colors as the VS Code extension (dark theme).
GPU, GPU_GLOW = "#1e9ccc", "#5fd4ff"
VRAM, VRAM_GLOW = "#d27019", "#ffb066"
CPU, CPU_GLOW = "#8e74e4", "#b9a8ff"
RAM, RAM_GLOW = "#d24d8a", "#ff8fc0"
TEMP, TEMP_GLOW = "#1fa67a", "#74f0c0"  # mint while cool; shifts to WARN / HOT near the threshold
POWER, POWER_GLOW = "#93ad1f", "#dcf27a"  # lime; shifts to WARN while the power cap throttles clocks
TENSOR, TENSOR_GLOW = "#b04fd0", "#ec9dff"
DISK, DISK_GLOW = "#2f8f9d", "#7fe3ef"
NET, NET_GLOW = "#5a7be0", "#a9bcff"
WARN, WARN_GLOW = "#d9950b", "#ffc94d"
HOT, HOT_GLOW = "#c8302f", "#ff7a6e"
TRACK = "#252b35"
MUTED = "#7d8696"
DIM = "#4a5160"
INK = "#eef1f5"
BORDER = "#263040"
BG = "#080a0e"
PANEL_BG = "#0b0e13"
HEAT = ("#16222b", "#0f4c66", "#1e9ccc", "#8fe1ff")
EMPTY_CELL = "#151a21"
LEVEL_COLOR = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b", "info": "#5b9fc0"}
LEVEL_ICON = {"good": "✔", "warning": "▲", "serious": "●", "critical": "⬣", "info": "ℹ"}
GRADE_COLOR = {"idle": MUTED, "low": "#fab219", "good": "#0ca30c", "max": GPU_GLOW}

EIGHTHS_H = " ▏▎▍▌▋▊▉█"
EIGHTHS_V = " ▁▂▃▄▅▆▇█"


def _rgb(hex_: str) -> tuple[int, int, int]:
    c = Color.parse(hex_).get_truecolor()
    return c.red, c.green, c.blue


def mix(a: str, b: str, t: float) -> str:
    x, y = _rgb(a), _rgb(b)
    t = max(0.0, min(1.0, t))
    red, green, blue = (round(p + (q - p) * t) for p, q in zip(x, y))
    return f"#{red:02x}{green:02x}{blue:02x}"


def heat_color(v: float) -> str:
    x = max(0.0, min(1.0, v / 100)) * (len(HEAT) - 1)
    i = min(len(HEAT) - 2, int(x))
    return mix(HEAT[i], HEAT[i + 1], x - i)


def fmt(v: Optional[float], digits: int = 0, unit: str = "") -> str:
    return "–" if v is None else f"{v:.{digits}f}{unit}"


def fmt_rate(bps: Optional[float]) -> str:
    """Human throughput, e.g. ``"3.4 MB/s"``; ``"n/a"`` when unknown."""
    if bps is None:
        return "n/a"
    v = max(0.0, bps)
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if v < 1000 or unit == "GB/s":
            return f"{v:.0f} {unit}" if unit == "B/s" or v >= 100 else f"{v:.1f} {unit}"
        v /= 1000
    return f"{v:.1f} GB/s"  # pragma: no cover - loop always returns


def badge(label: str, color: str) -> Text:
    """A small filled tag, e.g. ``POWER CAP``."""
    return Text(f" {label.upper()} ", Style(color=BG, bgcolor=color, bold=True))


def score_color(score: Optional[float]) -> str:
    if score is None:
        return MUTED
    if score >= 70:
        return LEVEL_COLOR["good"]
    return LEVEL_COLOR["warning"] if score >= 40 else LEVEL_COLOR["critical"]


def bar(frac: Optional[float], width: int, color: str, glow: str) -> Text:
    """Smooth horizontal gauge: gradient fill with sub-character precision on a dim track."""
    f = max(0.0, min(1.0, frac or 0.0))
    cells = f * width
    full = int(cells)
    out = Text()
    for i in range(full):
        out.append("█", Style(color=mix(color, glow, i / max(1, width - 1))))
    rem = int((cells - full) * 8)
    if full < width:
        if rem:
            out.append(EIGHTHS_H[rem], Style(color=mix(color, glow, full / max(1, width - 1)), bgcolor=TRACK))
            full += 1
        out.append(" " * (width - full), Style(bgcolor=TRACK))
    return out


def resample(values: Sequence[Optional[float]], width: int) -> list[Optional[float]]:
    """Average ``values`` into ``width`` buckets (or stretch if shorter)."""
    if not values or width <= 0:
        return [None] * max(0, width)
    n = len(values)
    out: list[Optional[float]] = []
    for i in range(width):
        lo, hi = int(i * n / width), max(int(i * n / width) + 1, int((i + 1) * n / width))
        chunk = [v for v in values[lo:hi] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def area_chart(
    values: Sequence[Optional[float]],
    width: int,
    height: int,
    color: str,
    glow: str,
    max_v: float = 100.0,
    axis: bool = True,
) -> Text:
    """Filled area chart from vertical eighth-blocks; brighter toward the top like a glow."""
    label_w = 5 if axis else 0
    cols = resample(values, max(1, width - label_w))
    lines = Text()
    for row in range(height):
        top = height - row  # rows counted from the bottom, 1-based
        if axis:
            lab = {0: f"{max_v:.0f}%", height // 2: f"{max_v / 2:.0f}%", height - 1: "0%"}.get(row, "")
            lines.append(f"{lab:>4} ", Style(color=MUTED))
        shade = mix(color, glow, 1 - row / max(1, height - 1))
        for v in cols:
            if v is None:
                lines.append("·" if row == height - 1 else " ", Style(color=TRACK))
                continue
            level = max(0.0, min(1.0, v / max_v)) * height
            if level >= top:
                lines.append("█", Style(color=shade))
            elif level > top - 1:
                lines.append(EIGHTHS_V[max(1, int((level - (top - 1)) * 8))], Style(color=shade))
            else:
                lines.append(" ")
        if row < height - 1:
            lines.append("\n")
    return lines


def sparkline(values: Sequence[Optional[float]], width: int, color: str, max_v: float = 100.0) -> Text:
    out = Text()
    for v in resample(values, width):
        if v is None:
            out.append(" ")
        else:
            out.append(EIGHTHS_V[max(1, round(max(0.0, min(1.0, v / max_v)) * 8))], Style(color=color))
    return out


def heatmap(
    start_ms: int, cells: Sequence[Optional[float]], now: Optional[float] = None, col_w: int = 1, tall: bool = False
) -> Text:
    """24 clock-aligned hours × 60 minutes, two hours per text row via '▀' (fg = upper, bg = lower).

    ``tall`` draws one hour per row instead (24 rows), for terminals with height to spare.
    """
    now = now if now is not None else time.time()
    hour0 = datetime.fromtimestamp(now).replace(minute=0, second=0, microsecond=0).timestamp() - 23 * 3600

    def cell(t: float) -> Optional[str]:
        if t > now:
            return None
        i = int((t * 1000 - start_ms) // 60_000)
        v = cells[i] if 0 <= i < len(cells) else None
        return EMPTY_CELL if v is None else heat_color(v)

    out = Text()
    out.append("      ", Style(color=MUTED))
    for m in range(0, 60, 15):
        out.append(f":{m:02d}".ljust(15 * col_w), Style(color=MUTED))
    out.append("\n")
    if tall:
        for r in range(24):
            t_row = hour0 + r * 3600
            label = "now" if r == 23 else datetime.fromtimestamp(t_row).strftime("%H:00")
            out.append(f"{label:>5} ", Style(color=INK if r == 23 else MUTED))
            for c in range(60):
                color = cell(t_row + c * 60)
                out.append("▀" * col_w if color else " " * col_w, Style(color=color) if color else None)
            if r < 23:
                out.append("\n")
        return out
    for r in range(0, 24, 2):
        t_top, t_bot = hour0 + r * 3600, hour0 + (r + 1) * 3600
        label = "now" if r == 22 else datetime.fromtimestamp(t_top).strftime("%H:00")
        out.append(f"{label:>5} ", Style(color=INK if r == 22 else MUTED))
        for c in range(60):
            fg, bg = cell(t_top + c * 60), cell(t_bot + c * 60)
            out.append("▀" * col_w, Style(color=fg or "default", bgcolor=bg))
        if r < 22:
            out.append("\n")
    return out


def heat_legend() -> Text:
    t = Text("0% ", Style(color=MUTED))
    for i in range(12):
        t.append("█", Style(color=heat_color(i / 11 * 100)))
    t.append(" 100%   ", Style(color=MUTED))
    t.append("█", Style(color=EMPTY_CELL))
    t.append(" not recorded", Style(color=MUTED))
    return t
