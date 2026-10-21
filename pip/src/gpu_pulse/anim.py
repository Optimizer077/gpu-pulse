"""Braille-dot graphics for the animated dashboard: ring gauges and an ECG-style pulse strip.

Each terminal cell holds a 2×4 grid of braille dots, so a cell is drawn at sub-character
resolution. With the usual 1:2 cell aspect ratio the dots are square, which is what lets
the ring gauges come out round. Geometry is precomputed per size, so a frame only does
per-cell bit masking.
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
START_DEG, SWEEP_DEG = 135.0, 270.0  # gauge opens at the bottom, like the VS Code version


@lru_cache(maxsize=4096)
def _style(color: str, bold: bool = False) -> Style:
    return Style(color=color, bold=bold)


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


def ring(
    width: int,
    height: int,
    frac: float,
    color: str,
    glow: str,
    shimmer: Optional[float] = None,
    lines: tuple[tuple[str, str, bool], ...] = (),
    marker: Optional[float] = None,
    top: Optional[int] = None,
) -> Text:
    """Draw a 270° ring gauge filled to ``frac`` with centred text ``lines`` of (text, colour, bold).

    ``shimmer`` (0..1) is the position of a highlight that travels along the filled arc;
    ``marker`` (0..1) places a ◆ on the tick ring (e.g. the 1-hour average); ``top`` is the
    row of the first text line (default: block centred on the ring centre).
    """
    g = ring_geometry(width, height)
    frac = max(0.0, min(1.0, frac))
    if top is None:
        top = g.center_row - len(lines) // 2
    overlay: dict[int, tuple[int, str, str, bool]] = {}
    for i, (s, c, b) in enumerate(lines):
        s = s[: max(1, width)]
        overlay[top + i] = ((width - len(s)) // 2, s, c, b)
    mark = g.marker_cell(marker) if marker is not None else None
    out = Text(no_wrap=True, overflow="crop")
    for y, row in enumerate(g.cells):
        ov = overlay.get(y)
        x = 0
        while x < width:
            if ov and ov[0] <= x < ov[0] + len(ov[1]):
                out.append(ov[1], _style(ov[2], ov[3]))
                x += len(ov[1])
                continue
            cell = row[x]
            x += 1
            if mark == (y, x - 1):
                out.append("◆", _style(R.INK, True))
                continue
            if cell is None:
                out.append(" ")
                continue
            lit = 0
            for bit, t in cell.ring:
                if t <= frac:
                    lit |= bit
            tick_bits = 0
            tick_lit = False
            for bit, t in cell.ticks:
                tick_bits |= bit
                tick_lit = tick_lit or t <= frac
            if lit:
                c = R.mix(color, glow, cell.t)
                if frac - 0.025 <= cell.t <= frac + 0.01:  # bright head at the arc's tip
                    c = R.mix(glow, "#ffffff", 0.65)
                elif shimmer is not None and abs(cell.t - shimmer) < 0.05 and cell.t <= frac:
                    c = R.mix(c, "#ffffff", 0.55 * (1 - abs(cell.t - shimmer) / 0.05))
                out.append(chr(0x2800 | lit | tick_bits), _style(c))
            elif cell.ring:
                out.append(chr(0x2800 | sum({b for b, _ in cell.ring}) | tick_bits), _style(R.TRACK))
            else:
                out.append(chr(0x2800 | tick_bits), _style(glow if tick_lit else "#3a4250"))
        if y < len(g.cells) - 1:
            out.append("\n")
    return out


def beside(left: Text, right: tuple[Text, ...], gap: int = 2, width: int = 0) -> Text:
    """Place ``right`` lines next to ``left`` (a multi-line Text), vertically centred.

    Every output row is padded/cropped to ``width`` cells when given, so it never wraps.
    """
    rows = left.split("\n", allow_blank=True)
    top = max(0, (len(rows) - len(right)) // 2)
    out = Text(no_wrap=True, overflow="crop")
    lw = max((r.cell_len for r in rows), default=0)
    for i, row in enumerate(rows):
        line = row.copy()
        line.pad_right(lw - line.cell_len)
        j = i - top
        if 0 <= j < len(right):
            line.append(" " * gap)
            line.append_text(right[j])
        if width:
            line.truncate(width, overflow="crop", pad=True)
        out.append_text(line)
        if i < len(rows) - 1:
            out.append("\n")
    return out


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
        for bits in row:
            out.append(chr(0x2800 | bits) if bits else " ", _style(shade))
        if r < height - 1:
            out.append("\n")
    return out


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
        for bits in row:
            out.append(chr(0x2800 | bits) if bits else " ", _style(c))
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


def pulse(wave: PulseWave, width: int, height: int, color: str, glow: str) -> Text:
    """Render the trace with a fading trail (left = old/dim, right = new/bright)."""
    dw, dh = width * 2, height * 4
    wave.resize(dw)
    base = dh * 0.68
    scale = dh * 0.62
    grid = [[0] * width for _ in range(height)]
    prev: Optional[int] = None
    for x, v in enumerate(wave.ys):
        y = int(round(base - v * scale))
        y = max(0, min(dh - 1, y))
        lo, hi = (y, y) if prev is None else (min(prev, y), max(prev, y))
        for yy in range(lo, hi + 1):  # vertical connection keeps the line continuous
            grid[yy // 4][x // 2] |= _BITS[x % 2][yy % 4]
        prev = y
    out = Text(no_wrap=True, overflow="crop")
    for row_i, row in enumerate(grid):
        for x, bits in enumerate(row):
            if not bits:
                out.append(" ")
                continue
            age = x / max(1, width - 1)
            c = R.mix(R.TRACK, color, min(1.0, age * 1.6))
            if age > 0.6:
                c = R.mix(color, glow, (age - 0.6) / 0.4)
            if x >= width - 2:
                c = R.mix(glow, "#ffffff", 0.6)
            out.append(chr(0x2800 | bits), _style(c))
        if row_i < height - 1:
            out.append("\n")
    return out
