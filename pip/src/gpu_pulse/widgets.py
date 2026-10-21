"""Animated Textual widgets: ring gauges and the pulse strip.

The app sets each widget's *target* when a sample arrives; ``step()`` is called every
animation frame to ease the displayed value toward it, so motion stays smooth even
though hardware is only sampled a few times per second.

``RingGauge`` adapts its content to whatever size the layout gives it:

* roomy  – ring on the left, a column of extra readings (``side``) on the right
* large  – big 3-row digits + caption + sub-line inside the ring
* medium – value / caption / sub-line
* small  – value only
* meter  – fewer than 3 rows: a one-line gradient meter instead of a ring
"""

from __future__ import annotations

import time
from typing import Optional

from rich.style import Style
from rich.text import Text
from textual.widget import Widget
from textual.widgets import Static

from . import anim
from . import render as R

BOOT_S = 0.9  # startup sweep: gauges swing to full, then settle on the real value
SIDE_MIN = 16  # columns needed before extra readings are shown beside the ring

Line = tuple[str, str, bool]  # (text, colour, bold)


def ring_lines(
    g: anim.RingGeometry, num: str, unit: str, caption: str, sub: str, accent: str
) -> tuple[tuple[Line, ...], int]:
    """Pick the richest centre text that fits inside the ring without touching it.

    Returns (lines, top row). Tried in order: big digits + caption + sub-line, big + caption,
    big alone, value + caption + sub-line, value + caption, value alone. Big digits sit on the
    ring centre; text below them uses the opening at the bottom of the ring.
    """
    t3, m3, b3 = anim.big_text(num if num != "–" else "-")
    bw = max(len(t3), len(m3), len(b3))
    big: tuple[Line, ...] = tuple(
        (r, accent, True) for r in (t3.ljust(bw + len(unit)), m3.ljust(bw + len(unit)), b3.ljust(bw) + unit)
    )
    value: Line = (f"{num}{unit}", R.INK, True)
    cap: Line = (caption, R.MUTED, False)
    subl: Line = (sub, R.INK, False)
    c = g.center_row
    candidates: list[tuple[tuple[Line, ...], int]] = [
        (big + (cap, subl), c - 1),
        (big + (cap,), c - 1),
        (big, c - 1),
        ((value, cap, subl), c - 1),
        ((value, cap), c),
        ((value,), c),
    ]
    # Big digits are judged at two-digit width at least, so a 7% ring doesn't switch style next to a 70% one.
    canon = 7 + len(unit)
    for lines, top in candidates:
        lines = tuple(x for x in lines if x[0])
        is_big = lines[0] is big[0]
        if all(g.fits(top + i, max(len(x[0]), canon) if is_big and i < 3 else len(x[0])) for i, x in enumerate(lines)):
            return lines, top
    return (value,), c


class RingGauge(Widget):
    DEFAULT_CSS = """
    RingGauge { height: 13; width: 1fr; border: round #263040; border-title-color: #9aa4b4;
                border-subtitle-color: #7d8696; padding: 0 0; }
    """

    def __init__(self, title: str, color: str, glow: str, unit: str = "%", max_value: float = 100.0, **kw) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kw)
        self.border_title = title
        self.base = (color, glow)
        self.color, self.glow, self.unit, self.max_value = color, glow, unit, max_value
        self.value = 0.0  # displayed (eased)
        self.target: Optional[float] = None
        self.caption = ""
        self.sub = ""
        self.marker: Optional[float] = None  # 0..1 position of the ◆ (e.g. 1-hour average)
        self.side: tuple[tuple[str, str], ...] = ()  # (label, value) readings shown beside a roomy ring
        self.fit_width: Optional[int] = None  # fit text as if this wide, so sibling rings pick the same tier
        self.shimmer = 0.0
        self.animate_motion = True
        self._boot_until = time.monotonic() + BOOT_S
        self._border_color = ""

    def set_target(
        self,
        value: Optional[float],
        caption: str = "",
        sub: str = "",
        max_value: Optional[float] = None,
        marker: Optional[float] = None,
        side: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.target = value
        self.caption, self.sub, self.marker, self.side = caption, sub, marker, side
        if max_value:
            self.max_value = max_value
        if not self.animate_motion:
            self.value = value or 0.0
        self.refresh()

    def set_state(self, level: str = "normal", note: str = "") -> None:
        """Recolour for a warning state: ``normal``, ``warn`` (amber) or ``hot`` (red)."""
        self.color, self.glow = {"warn": (R.WARN, R.WARN_GLOW), "hot": (R.HOT, R.HOT_GLOW)}.get(level, self.base)
        border = {"warn": R.WARN, "hot": R.HOT}.get(level, R.BORDER)
        if border != self._border_color:
            self._border_color = border
            self.styles.border = ("round", border)
            self.styles.border_subtitle_color = R.mix(border, "#ffffff", 0.25) if level != "normal" else R.MUTED
        self.border_subtitle = note

    def step(self, dt: float) -> None:
        """Advance one animation frame."""
        booting = time.monotonic() < self._boot_until
        goal = self.max_value if booting else (self.target or 0.0)
        # Critically-damped-ish easing: ~90% of the way in 0.35 s regardless of frame rate.
        k = 1 - pow(0.001, dt / 1.0)
        self.value += (goal - self.value) * min(1.0, k * (2.2 if booting else 1.6))
        self.shimmer = (self.shimmer + dt * 0.45) % 1.25  # pause beyond 1.0 between passes
        self.refresh()

    # ---- rendering ----
    def _num(self) -> str:
        return "–" if self.target is None else f"{self.target:.0f}"

    def _meter(self, w: int, h: int) -> Text:
        frac = self.value / self.max_value if self.max_value else 0.0
        value = f"{self._num()}{self.unit}"
        t = Text(no_wrap=True, overflow="crop")
        t.append(f"{value:>5} ", Style(color=R.INK, bold=True))
        t.append_text(R.bar(frac, max(1, w - 6), self.color, self.glow))
        if h >= 2:
            t.append("\n")
            t.append(f"{self.caption}  {self.sub}".strip()[:w], Style(color=R.MUTED))
        return t

    def render(self) -> Text:
        w, h = max(4, self.content_size.width), max(1, self.content_size.height)
        if h < 3 or w < 8:
            return self._meter(w, h)
        frac = self.value / self.max_value if self.max_value else 0.0
        ring_w = min(w, 2 * h + 1)
        roomy = bool(self.side) and w - ring_w - 2 >= SIDE_MIN
        rw = ring_w if roomy else w
        g = anim.ring_geometry(rw, h)
        accent = R.mix(self.color, self.glow, 0.8)
        fit = anim.ring_geometry(min(rw, self.fit_width), h) if self.fit_width and self.fit_width < rw else g
        lines, top = ring_lines(fit, self._num(), self.unit, self.caption, self.sub, accent)
        top += g.center_row - fit.center_row
        shimmer = self.shimmer if self.animate_motion and self.shimmer <= 1.0 else None
        ring = anim.ring(rw, h, frac, self.color, self.glow, shimmer, lines, self.marker, top)
        if not roomy:
            return ring
        side_w = w - rw - 2
        rows: list[Text] = []
        label_w = max((len(lbl) for lbl, _ in self.side), default=0)
        for label, value in self.side[:h]:
            t = Text(no_wrap=True, overflow="ellipsis")
            t.append(f"{label:<{label_w}} ", Style(color=R.MUTED))
            t.append(value, Style(color=R.INK, bold=True))
            if t.cell_len > side_w:
                t.truncate(side_w, overflow="ellipsis")
            rows.append(t)
        return anim.beside(ring, tuple(rows), gap=2, width=w)


class Panel(Static):
    """A bordered panel whose content is re-rendered from data."""

    DEFAULT_CSS = """
    Panel { border: round #263040; border-title-color: #9aa4b4; border-subtitle-color: #7d8696;
            padding: 0 1; height: auto; }
    """

    def __init__(self, title: str, **kw) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kw)
        self.border_title = title

    def on_resize(self) -> None:
        # Content is sized to the panel, so ask the app to rebuild the page once layout settles.
        request = getattr(self.app, "request_rerender", None)
        if request is not None:
            request()

    # ``size`` is already the content area (border and padding excluded).
    @property
    def inner_width(self) -> int:
        return max(20, self.size.width)

    @property
    def inner_height(self) -> int:
        return max(1, self.size.height)


class PulseStrip(Widget):
    DEFAULT_CSS = """
    PulseStrip { height: 5; width: 1fr; border: round #263040; border-title-color: #9aa4b4;
                 border-subtitle-color: #7d8696; padding: 0 1; }
    """

    COLUMNS_PER_S = 36  # scroll speed in dot columns per second

    def __init__(self, **kw) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kw)
        self.border_title = "PULSE · beats faster as the GPU works harder"
        self.wave = anim.PulseWave()
        self.util: Optional[float] = None
        self._carry = 0.0

    def step(self, dt: float) -> None:
        self._carry += dt * self.COLUMNS_PER_S
        n = int(self._carry)
        if n:
            self._carry -= n
            self.wave.advance(self.util, n)
            self.refresh()

    def render(self) -> Text:
        w, h = max(10, self.content_size.width), max(1, self.content_size.height)
        return anim.pulse(self.wave, w, h, R.GPU, R.GPU_GLOW)
