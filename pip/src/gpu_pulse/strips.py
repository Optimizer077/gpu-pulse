"""Line-API renderers for the animated widgets: ring-gauge rows and the pulse strip.

Textual's ``render()`` path turns a Rich ``Text`` into ``Content`` and then into strips on every
refresh; for six braille rings at 20 fps that was most of the dashboard's CPU time. These helpers
build ``Segment`` rows straight from precomputed geometry with cached styles, merge runs of equal
style, and give every ring row a cheap *key* so a widget repaints only the rows that changed.
"""

from __future__ import annotations

import bisect
from collections.abc import Hashable
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from rich.segment import Segment
from rich.style import Style

from . import anim
from . import render as R

HEAD_BEHIND, HEAD_AHEAD = 0.025, 0.01  # bright tip: cells with t in [frac - 0.025, frac + 0.01]
SHIMMER_HALF = 0.05  # half-width of the travelling highlight, in sweep fractions
SHIMMER_STEPS = 200  # shimmer position is quantised so a row's key repeats between nearby frames
TICK_DIM = "#3a4250"
WHITE = "#ffffff"

Overlay = tuple[int, str, str, bool]  # (x, text, colour, bold)


@dataclass(frozen=True)
class CellPlan:
    ts: tuple[float, ...]  # band dot positions along the sweep, sorted
    cum: tuple[int, ...]  # cum[i] = OR of the bits of the first i + 1 band dots
    all_bits: int  # every band dot (drawn dim as the unlit track)
    tick_bits: int
    tick_t: float  # position of the cell's first tick dot (lit once frac passes it); 2.0 = no ticks
    t: float  # colour position (mean t of the band dots)


@dataclass(frozen=True)
class RowPlan:
    cells: tuple[Optional[CellPlan], ...]  # one per column; None = empty
    dot_ts: tuple[float, ...]  # every position where a dot of this row lights up, sorted
    band_ts: tuple[float, ...]  # colour positions of the row's band cells, sorted


@lru_cache(maxsize=32)
def ring_plan(width: int, height: int) -> tuple[RowPlan, ...]:
    """Per-row lookup tables for a ring of this size (built once, from ``anim.ring_geometry``)."""
    rows: list[RowPlan] = []
    for row in anim.ring_geometry(width, height).cells:
        cells: list[Optional[CellPlan]] = []
        dots: list[float] = []
        band: list[float] = []
        for c in row:
            if c is None:
                cells.append(None)
                continue
            ring = sorted(c.ring, key=lambda d: d[1])
            cum: list[int] = []
            acc = 0
            for bit, _ in ring:
                acc |= bit
                cum.append(acc)
            tick_bits = 0
            for bit, _ in c.ticks:
                tick_bits |= bit
            tick_t = min((t for _, t in c.ticks), default=2.0)
            cells.append(CellPlan(tuple(t for _, t in ring), tuple(cum), acc, tick_bits, tick_t, c.t))
            dots.extend(t for _, t in ring)
            if c.ticks:
                dots.append(tick_t)
            if ring:
                band.append(c.t)
        rows.append(RowPlan(tuple(cells), tuple(sorted(dots)), tuple(sorted(band))))
    return tuple(rows)


def quantize_shimmer(shimmer: Optional[float]) -> Optional[float]:
    return None if shimmer is None else round(shimmer * SHIMMER_STEPS) / SHIMMER_STEPS


def row_key(rp: RowPlan, frac: float, shimmer: Optional[float]) -> Hashable:
    """Everything about ``frac`` / ``shimmer`` that changes this row's pixels, as a small tuple.

    Two frames with equal keys draw identical rows, so easing between dot positions and a
    shimmer that is not over this row cost nothing.
    """
    bt = rp.band_ts
    lit = bisect.bisect_right(rp.dot_ts, frac)
    head = (bisect.bisect_left(bt, frac - HEAD_BEHIND), bisect.bisect_right(bt, frac + HEAD_AHEAD))
    sk: Optional[float] = None
    if shimmer is not None:
        i = bisect.bisect_right(bt, shimmer - SHIMMER_HALF)
        if i < len(bt) and bt[i] < shimmer + SHIMMER_HALF and bt[i] <= frac:
            sk = shimmer
    return lit, head, sk


def _lit(c: CellPlan, frac: float) -> int:
    i = bisect.bisect_right(c.ts, frac)
    return c.cum[i - 1] if i else 0


def ring_row(
    rp: RowPlan,
    frac: float,
    color: str,
    glow: str,
    shimmer: Optional[float],
    overlay: Optional[Overlay],
    mark_x: Optional[int],
) -> list[Segment]:
    """One row of a ring gauge (same drawing rules as before, without the Text round trip)."""
    out: list[Segment] = []
    run: list[str] = []
    run_style: Optional[Style] = None
    head = R.style(R.mix(glow, WHITE, 0.65))
    track = R.style(R.TRACK)
    ov_x, ov_end = (overlay[0], overlay[0] + len(overlay[1])) if overlay else (-1, -1)
    cells = rp.cells
    width = len(cells)
    x = 0
    while x < width:
        if ov_x <= x < ov_end and overlay is not None:
            if run:
                out.append(Segment("".join(run), run_style))
                run = []
            out.append(Segment(overlay[1], R.style(overlay[2], overlay[3])))
            run_style = None
            x = ov_end
            continue
        c = cells[x]
        if x == mark_x:
            ch, st = "◆", R.style(R.INK, True)
        elif c is None:
            run.append(" ")  # a space looks the same in any foreground colour: extend the current run
            x += 1
            continue
        else:
            lit = _lit(c, frac) if c.ts else 0
            if lit:
                if frac - HEAD_BEHIND <= c.t <= frac + HEAD_AHEAD:
                    st = head
                elif shimmer is not None and abs(c.t - shimmer) < SHIMMER_HALF and c.t <= frac:
                    amount = round(0.55 * (1 - abs(c.t - shimmer) / SHIMMER_HALF) * 16) / 16
                    st = R.style(R.mix(R.mix(color, glow, c.t), WHITE, amount))
                else:
                    st = R.style(R.mix(color, glow, c.t))
                ch = chr(0x2800 | lit | c.tick_bits)
            elif c.ts:
                ch, st = chr(0x2800 | c.all_bits | c.tick_bits), track
            else:
                ch, st = chr(0x2800 | c.tick_bits), R.style(glow if c.tick_t <= frac else TICK_DIM)
        if st is not run_style and run and run_style is not None:
            out.append(Segment("".join(run), run_style))
            run = []
        run_style = st
        run.append(ch)
        x += 1
    if run:
        out.append(Segment("".join(run), run_style))
    return out


# ---- pulse strip ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def pulse_styles(width: int, color: str, glow: str) -> tuple[Style, ...]:
    """Per-column style of the fading trail, quantised so neighbouring columns share a style."""
    out: list[Style] = []
    for x in range(width):
        age = round(x / max(1, width - 1) * 48) / 48
        c = R.mix(R.TRACK, color, min(1.0, age * 1.6))
        if age > 0.6:
            c = R.mix(color, glow, (age - 0.6) / 0.4)
        if x >= width - 2:
            c = R.mix(glow, WHITE, 0.6)
        out.append(R.style(c))
    return tuple(out)


def pulse_rows(wave: anim.PulseWave, width: int, height: int, color: str, glow: str) -> list[list[Segment]]:
    """The scrolling ECG trace as segment rows (left = old/dim, right = new/bright)."""
    dw, dh = width * 2, height * 4
    wave.resize(dw)
    base, scale = dh * 0.68, dh * 0.62
    grid = [[0] * width for _ in range(height)]
    bits = anim.BRAILLE_BITS
    prev: Optional[int] = None
    for x, v in enumerate(wave.ys):
        y = max(0, min(dh - 1, int(round(base - v * scale))))
        lo, hi = (y, y) if prev is None else (min(prev, y), max(prev, y))
        col = bits[x % 2]
        cx = x // 2
        for yy in range(lo, hi + 1):  # vertical connection keeps the line continuous
            grid[yy // 4][cx] |= col[yy % 4]
        prev = y
    styles = pulse_styles(width, color, glow)
    rows: list[list[Segment]] = []
    for row in grid:
        segs: list[Segment] = []
        run: list[str] = []
        run_style: Optional[Style] = None
        for x, b in enumerate(row):
            if not b:
                run.append(" ")
                continue
            st = styles[x]
            if st is not run_style and run and run_style is not None:
                segs.append(Segment("".join(run), run_style))
                run = []
            run_style = st
            run.append(chr(0x2800 | b))
        if run:
            segs.append(Segment("".join(run), run_style))
        rows.append(segs)
    return rows
