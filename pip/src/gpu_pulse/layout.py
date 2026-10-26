"""Responsive layout planning: pure functions from terminal size to panel sizes.

Kept free of Textual so the sizing rules are unit-testable. Braille dots are square when a
cell is 1:2 (w:h), so a round ring of content height ``h`` rows needs about ``2h`` columns.
"""

from __future__ import annotations

from dataclasses import dataclass

RING_COUNT = 6
# Fraction of the body height the ring block may take, by number of ring rows.
RING_SHARE = {1: 0.36, 2: 0.46, 3: 0.54}
RING_MIN, RING_MAX = 5, 22  # content rows of a drawn ring; below RING_MIN rings become meters
CHROME = 2  # border rows/cols per panel
PAD = 0  # horizontal padding inside a ring panel (the tick ring already leaves a margin)
MIDDLE_MIN = 12  # pulse/timeline/diagnosis block
BOTTOM_H = 6  # usage-window strip + I/O on the Overview


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    compact: bool  # small terminal: drop non-essential panels
    ring_cols: int
    ring_rows: int
    ring_h: int  # outer height of one ring panel (content + border)
    meters: bool  # rings too small to draw: one-line meters instead
    middle_h: int  # Overview middle block (pulse + timeline | diagnosis + health)
    show_pulse: bool
    show_bottom: bool  # Overview usage strip + I/O
    bottom_h: int
    columns: int  # Overview middle: 3 = timeline | true usage + health | diagnosis; 2 = no health column; 1
    fleet_h: int = 0  # outer height of the ALL GPUS panel (multi-GPU only)

    @property
    def show_left(self) -> bool:
        return self.columns >= 2


def ring_content_h(width: int, height: int, cols: int, rows: int, fleet_h: int = 0) -> int:
    """Largest round-ring content height for a ``cols`` × ``rows`` grid in ``width`` × ``height``."""
    body = max(1, height - 2 - fleet_h)  # header/tab bar + footer (+ the ALL GPUS panel)
    content_w = width // cols - CHROME - PAD
    by_width = content_w // 2
    by_height = int(body * RING_SHARE[rows]) // rows - CHROME
    return min(by_width, by_height, RING_MAX)


def plan(width: int, height: int, fleet_h: int = 0, ring_count: int = RING_COUNT) -> Layout:
    """Panel sizes for a terminal; ``fleet_h`` is the outer height of the ALL GPUS panel (0 = one GPU)."""
    body = max(1, height - 2 - fleet_h)
    compact = width < 80 or height < 24
    grids = ((4, 1), (2, 2)) if ring_count == 4 else ((6, 1), (3, 2), (2, 3))
    best = (-1, ring_count, 1)
    for cols, rows in grids:
        h = ring_content_h(width, height, cols, rows, fleet_h)
        if h > best[0]:
            best = (h, cols, rows)
    h, cols, rows = best
    meters = h < RING_MIN
    if meters:
        # One-line meters need ~18 columns each; 1 or 2 content rows depending on height.
        cols = (4 if width >= 4 * 22 else 2) if ring_count == 4 else (
            6 if width >= 6 * 22 else 3 if width >= 3 * 20 else 2
        )
        rows = ring_count // cols
        h = 2 if body - rows * 4 >= 12 else 1
    ring_h = h + CHROME
    rest = body - rows * ring_h
    bottom_h = max(BOTTOM_H, min(12, int(rest * 0.26)))
    show_bottom = not compact and rest - bottom_h >= MIDDLE_MIN + 4
    middle = rest - (bottom_h if show_bottom else 0)
    return Layout(
        width=width,
        height=height,
        fleet_h=fleet_h,
        compact=compact,
        ring_cols=cols,
        ring_rows=rows,
        ring_h=ring_h,
        meters=meters,
        middle_h=max(MIDDLE_MIN if not compact else 8, middle),
        show_pulse=not compact and middle >= 16,
        show_bottom=show_bottom,
        bottom_h=bottom_h,
        columns=1 if compact or width < 90 else 3 if width >= 120 else 2,
    )
