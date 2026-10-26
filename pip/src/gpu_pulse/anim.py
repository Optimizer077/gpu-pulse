"""Braille-dot graphics for the animated dashboard: ring geometry, traces and the pulse wave.

Each terminal cell holds a 2×4 grid of braille dots, so a cell is drawn at sub-character
resolution. With the usual 1:2 cell aspect ratio the dots are square, which is what lets
the ring gauges come out round. Geometry is precomputed per size; the per-frame drawing of
rings and the pulse strip lives in :mod:`gpu_pulse.strips`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from rich.style import Style
from rich.text import Text

from . import render as R

# Braille bit for dot (x in 0..1, y in 0..3) inside one cell.
_BITS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))
BRAILLE_BITS = _BITS
START_DEG, SWEEP_DEG = 135.0, 270.0  # gauge opens at the bottom, like the VS Code version


def _style(color: str, bold: bool = False) -> Style:
    return R.style(color, bold)


# ---- big digits (3 rows) for the gauge centre --------------------------------------------
_FONT = {
    "0": ("╭─╮", "│ │", "╰─╯"),
    "1": ("╶┐ ", " │ ", "╶┴╴"),
    "2": ("╶─╮", "╭─╯", "╰─╴"),
    "3": ("╶─╮", " ─┤", "╶─╯"),
    "4": ("╷ ╷", "╰─┤", "  ╵"),
    "5": ("╭─╴", "╰─╮", "╶─╯"),
    "6": ("╭─╴", "├─╮", "╰─╯"),
    "7": ("╶─┐", "  │", "  ╵"),
    "8": ("╭─╮", "├─┤", "╰─╯"),
    "9": ("╭─╮", "╰─┤", "╶─╯"),
    "-": ("   ", "╶─╴", "   "),
    ".": (" ", " ", "."),
}


def big_text(s: str) -> tuple[str, str, str]:
    """Render digits in a 3-row box-drawing font; unknown characters are dropped."""
    rows = ["", "", ""]
    for ch in s:
        glyph = _FONT.get(ch)
        if glyph:
            for i in range(3):
                rows[i] += glyph[i] + " "
    return rows[0].rstrip(), rows[1].rstrip(), rows[2].rstrip()


# ---- ring gauge ----------------------------------------------------------------------------
@dataclass(frozen=True)
class _Cell:
    ring: tuple[tuple[int, float], ...]  # (bit, t along the sweep 0..1) for band dots
    ticks: tuple[tuple[int, float], ...]  # (bit, t) for tick-mark dots
    t: float  # mean t of the band dots, used for the cell's colour


@dataclass(frozen=True)
class RingGeometry:
    width: int
    height: int
    cells: tuple[tuple[Optional[_Cell], ...], ...]
    center_row: int
    inner_cols: int  # usable width (cells) inside the ring for centre text
    inner_rows: int  # usable height (rows) inside the ring
    cx: float  # ring centre, in dots
    cy: float
    r_tick: float  # radius of the tick ring, in dots (markers sit here)
    free: tuple[tuple[int, int], ...] = ()  # per row: [start, end) of the empty run around the centre column

    def fits(self, row: int, text_len: int, margin: int = 1) -> bool:
        """True if ``text_len`` centred cells fit in ``row`` without touching the ring (``margin`` cells clear)."""
        if not 0 <= row < self.height:
            return False
        a, b = self.free[row]
        x0 = (self.width - text_len) // 2
        lo, hi = (a + margin, b - margin) if (a, b) != (0, self.width) else (0, self.width)
        return lo <= x0 and x0 + text_len <= hi

    def marker_cell(self, frac: float) -> tuple[int, int]:
        """(row, col) of the tick-ring cell at position ``frac`` along the sweep."""
        a = math.radians(START_DEG + SWEEP_DEG * max(0.0, min(1.0, frac)))
        x, y = self.cx + math.cos(a) * self.r_tick, self.cy + math.sin(a) * self.r_tick
        return min(self.height - 1, max(0, int(y // 4))), min(self.width - 1, max(0, int(x // 2)))


@lru_cache(maxsize=32)
def ring_geometry(width: int, height: int) -> RingGeometry:
    dw, dh = width * 2, height * 4
    cx, cy = (dw - 1) / 2, (dh - 1) / 2 + dh * 0.04  # nudge down: the gap is at the bottom
    r_out = max(4.0, min(dw, dh) / 2 - 2.5)
    thick = max(2.0, r_out * 0.2)
    r_in = r_out - thick
    r_tick = r_out + 1.6
    ring: dict[tuple[int, int], list[tuple[int, float]]] = {}
    ticks: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for y in range(dh):
        for x in range(dw):
            dx, dy = x - cx, y - cy
            r = math.hypot(dx, dy)
            ang = (math.degrees(math.atan2(dy, dx)) - START_DEG) % 360
            if ang > SWEEP_DEG:
                continue
            t = ang / SWEEP_DEG
            key, bit = (y // 4, x // 2), _BITS[x % 2][y % 4]
            if r_in <= r <= r_out:
                ring.setdefault(key, []).append((bit, t))
            # Ticks every 10%, about one dot wide (angular tolerance shrinks with radius).
            elif abs(r - r_tick) < 0.7 and abs(t * 10 - round(t * 10)) < 1.06 / r_tick:
                ticks.setdefault(key, []).append((bit, t))
    rows = []
    free: list[tuple[int, int]] = []
    mid = width // 2
    for cy_ in range(height):
        row: list[Optional[_Cell]] = []
        for cx_ in range(width):
            rd, td = ring.get((cy_, cx_), []), ticks.get((cy_, cx_), [])
            dots = rd or td  # tick-only cells take their colour position from the ticks
            row.append(_Cell(tuple(rd), tuple(td), sum(t for _, t in dots) / len(dots)) if dots else None)
        rows.append(tuple(row))
        a = b = mid
        while a > 0 and row[a - 1] is None:
            a -= 1
        while b < width and row[b] is None:
            b += 1
        free.append((a, b) if row[mid] is None else (mid, mid))
    # Inner diameter is 2*r_in dots = r_in cells wide (2 dots/cell) and r_in/2 rows tall (4 dots/row).
    return RingGeometry(
        width,
        height,
        tuple(rows),
        int(cy // 4),
        max(3, int(r_in) - 1),
        max(1, int(r_in / 2)),
        cx,
        cy,
        r_tick,
        tuple(free),
    )


# ---- braille area trace (I/O, CPU history) -------------------------------------------------
def trace(
    values: list[Optional[float]],
    width: int,
    height: int,
    color: str,
    glow: str,
    max_v: Optional[float] = None,
) -> Text:
    """A filled braille area chart, 2 columns × 4 rows of dots per cell; newest on the right.

    ``max_v`` None auto-scales to the visible peak, so small I/O bursts are still visible.
    """
    width, height = max(1, width), max(1, height)
    dw, dh = width * 2, height * 4
    cols = R.resample(values, dw)
    peak = max_v if max_v else max((v for v in cols if v is not None), default=0.0) or 1.0
    grid = [[0] * width for _ in range(height)]
    for x, v in enumerate(cols):
        if v is None:
            continue
        lvl = max(0.0, min(1.0, v / peak))
        n = int(round(lvl * dh))
        if v > 0 and n == 0:
            n = 1  # any activity at all leaves a visible baseline
        for k in range(n):
            yy = dh - 1 - k
            grid[yy // 4][x // 2] |= _BITS[x % 2][yy % 4]
    out = Text(no_wrap=True, overflow="crop")
    for r, row in enumerate(grid):
        shade = R.mix(color, glow, 1 - r / max(1, height - 1)) if height > 1 else glow
        out.append(_braille_row(row), _style(shade))  # one span per row
        if r < height - 1:
            out.append("\n")
    return out


def _braille_row(row: list[int]) -> str:
    return "".join(chr(0x2800 | bits) if bits else " " for bits in row)


def mirror(
    up: list[Optional[float]],
    down: list[Optional[float]],
    width: int,
    height: int,
    up_colors: tuple[str, str],
    down_colors: tuple[str, str],
) -> Text:
    """Two series sharing one scale: ``up`` grows upward from the middle, ``down`` downward."""
    width, height = max(1, width), max(2, height)
    top_h = height // 2
    dw = width * 2
    a, b = R.resample(up, dw), R.resample(down, dw)
    peak = max((v for v in a + b if v is not None), default=0.0) or 1.0
    grid = [[0] * width for _ in range(height)]
    half_up, half_down = top_h * 4, (height - top_h) * 4
    for x in range(dw):
        for v, span, sign in ((a[x], half_up, -1), (b[x], half_down, 1)):
            if v is None:
                continue
            n = int(round(max(0.0, min(1.0, v / peak)) * span)) or (1 if v > 0 else 0)
            for k in range(n):
                yy = half_up - 1 - k if sign < 0 else half_up + k
                grid[yy // 4][x // 2] |= _BITS[x % 2][yy % 4]
    out = Text(no_wrap=True, overflow="crop")
    for r, row in enumerate(grid):
        if r < top_h:
            c = R.mix(up_colors[0], up_colors[1], 1 - r / max(1, top_h - 1) if top_h > 1 else 1)
        else:
            k = r - top_h
            c = R.mix(down_colors[0], down_colors[1], 1 - k / max(1, height - top_h - 1) if height - top_h > 1 else 1)
        out.append(_braille_row(row), _style(c))
        if r < height - 1:
            out.append("\n")
    return out


# ---- ECG-style pulse strip -----------------------------------------------------------------
class PulseWave:
    """A scrolling heartbeat trace. Beats get faster and taller as utilization rises."""

    def __init__(self, columns: int = 240) -> None:
        self.ys: deque[float] = deque([0.0] * columns, maxlen=columns)
        self.pos = 0.0  # position inside the current beat, in dot columns
        self.level = 0.0  # 0..1 smoothed utilization

    def resize(self, columns: int) -> None:
        if columns != self.ys.maxlen:
            old = list(self.ys)[-columns:]
            self.ys = deque([0.0] * (columns - len(old)) + old, maxlen=columns)

    def _beat_value(self, p: float, amp: float) -> float:
        """Shape of one beat at phase p (0..1): P bump, sharp QRS spike, T bump; 0 = baseline."""

        def bump(c: float, w: float, h: float) -> float:
            return h * math.exp(-(((p - c) / w) ** 2))

        return (
            bump(0.18, 0.035, 0.12 * amp)
            - bump(0.33, 0.012, 0.18 * amp)
            + bump(0.36, 0.014, 1.0 * amp)
            - bump(0.395, 0.014, 0.32 * amp)
            + bump(0.6, 0.06, 0.22 * amp)
        )

    def advance(self, util: Optional[float], steps: int) -> None:
        target = max(0.0, min(1.0, (util or 0.0) / 100))
        for _ in range(steps):
            self.level += (target - self.level) * 0.05
            beat_len = 90 - 66 * self.level  # dot columns per beat: slow when idle, fast when busy
            amp = 0.18 + 0.82 * self.level
            self.pos = (self.pos + 1) % beat_len
            self.ys.append(self._beat_value(self.pos / beat_len, amp))
