"""Animated Textual widgets (ring gauges, pulse strip) and change-aware panels.

The app sets each gauge's *target* when a sample arrives; ``step()`` is called every animation
frame to ease the displayed value toward it, so motion stays smooth even though hardware is
only sampled a few times per second.

Cost model (a monitor must stay cheap): gauges and the pulse strip use Textual's Line API
(``render_line``) and draw rows straight from cached geometry and styles. Each ring row has a
key built from what changes its pixels; a frame repaints only rows whose key changed, a gauge
that reached its target reports itself settled so the app can stop the frame timer, and panels
skip ``update`` when their new content is identical to what is already shown.

``RingGauge`` adapts its content to whatever size the layout gives it:

* roomy  – ring on the left, a column of extra readings (``side``) on the right
* large  – big 3-row digits + caption + sub-line inside the ring
* medium – value / caption / sub-line
* small  – value only
* meter  – fewer than 3 rows: a one-line gradient meter instead of a ring
"""

from __future__ import annotations

import time
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Optional

from rich.cells import cell_len
from rich.console import Group
from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual.geometry import Region
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Static

from . import anim
from . import render as R
from . import strips as S

BOOT_S = 0.9  # startup sweep: gauges swing to full, then settle on the real value
SIDE_MIN = 16  # columns needed before extra readings are shown beside the ring
SIDE_GAP = 2
SETTLE = 0.002  # snap to the target within 0.2% of the range (well under one braille dot)
SHIMMER_MOVE = 2  # shown units (%, °C, W) a value must move between two shimmer passes on its ring
SNAP = 0.03  # target changes up to 3% of the range jump instead of easing (about one ring cell)

Line = tuple[str, str, bool]  # (text, colour, bold)


def _shown(value: Optional[float]) -> Optional[int]:
    """The integer a ring prints for ``value`` (None = "–")."""
    return None if value is None else round(value)


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


def text_strips(widget: Widget, text: Text, width: int) -> list[Strip]:
    """Render a Rich ``Text`` into content-width strips (used for rarely drawn fallbacks)."""
    console = widget.app.console
    lines = console.render_lines(text, console.options.update_width(max(1, width)), pad=True)
    return [Strip(line, width) for line in lines]


@dataclass
class _RingLayout:
    """Per-sample state of a ring gauge: everything except the eased value and the shimmer."""

    meter: bool
    rw: int = 0  # ring width (cells)
    overlay: dict[int, S.Overlay] = field(default_factory=dict)
    mark: Optional[tuple[int, int]] = None
    side: list[list[Segment]] = field(default_factory=list)  # per row, extra readings beside the ring
    side_keys: list[Hashable] = field(default_factory=list)


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
        self.numeric_only = False  # absolute readings with no meaningful capacity, e.g. unified GPU allocations
        self.precision = 0
        self.caption = ""
        self.sub = ""
        self.marker: Optional[float] = None  # 0..1 position of the ◆ (e.g. 1-hour average)
        self.side: tuple[tuple[str, str], ...] = ()  # (label, value) readings shown beside a roomy ring
        self.fit_width: Optional[int] = None  # fit text as if this wide, so sibling rings pick the same tier
        self.shimmer: Optional[float] = None  # 0..1 position of the travelling highlight, None = off
        self.animate_motion = True
        self.rows_painted = 0  # rows rebuilt so far (cost accounting for tests and benchmarks)
        self._changed = True
        self._shimmer_ref: Optional[int] = None  # shown value at the last shimmer pass
        self._boot_until = time.monotonic() + BOOT_S
        self._border_color = ""
        self._lay_key: Optional[tuple[Any, ...]] = None
        self._lay = _RingLayout(meter=True)
        self._painted_size = (0, 0)
        self._keys: list[Hashable] = []
        self._strips: dict[int, Strip] = {}

    # ---- state ----
    def set_target(
        self,
        value: Optional[float],
        caption: str = "",
        sub: str = "",
        max_value: Optional[float] = None,
        marker: Optional[float] = None,
        side: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if _shown(value) != _shown(self.target):  # the number on screen changes, not just its decimals
            self._changed = True
        self.target = value
        self.caption, self.sub, self.marker, self.side = caption, sub, marker, side
        if max_value:
            self.max_value = max_value
        goal = value or 0.0
        if not self.animate_motion:
            self.value = goal
        elif time.monotonic() >= self._boot_until and abs(goal - self.value) <= SNAP * max(1.0, self.max_value):
            # A small step (CPU load wobbling a few percent) is not visible as motion: jump now, so it
            # repaints with this sample's update instead of easing through several animation frames.
            self.value = goal
        self.redraw()

    def set_state(self, level: str = "normal", note: str = "") -> None:
        """Recolour for a warning state: ``normal``, ``warn`` (amber) or ``hot`` (red)."""
        self.color, self.glow = {"warn": (R.WARN, R.WARN_GLOW), "hot": (R.HOT, R.HOT_GLOW)}.get(level, self.base)
        border = {"warn": R.WARN, "hot": R.HOT}.get(level, R.BORDER)
        if border != self._border_color:
            self._border_color = border
            self.styles.border = ("round", border)
            self.styles.border_subtitle_color = R.mix(border, "#ffffff", 0.25) if level != "normal" else R.MUTED
        set_border(self, subtitle=note)
        self.redraw()

    def consume_change(self) -> bool:
        """True (once) if the shown value moved by SHIMMER_MOVE or more since the last True: gates the shimmer.

        A reading that wobbles by a unit around one level (CPU load, power draw) never earns a pass.
        """
        changed, self._changed = self._changed, False
        if not changed:
            return False
        now, ref = _shown(self.target), self._shimmer_ref
        if now is not None and ref is not None and abs(now - ref) < SHIMMER_MOVE:
            return False
        self._shimmer_ref = now
        return True

    @property
    def settled(self) -> bool:
        """True once the boot sweep is over and the displayed value has reached the target."""
        return self.numeric_only or (time.monotonic() >= self._boot_until and self.value == (self.target or 0.0))

    def step(self, dt: float, shimmer: Optional[float] = None) -> bool:
        """Advance one animation frame; returns True while the value is still moving."""
        if self.numeric_only:
            return False
        booting = time.monotonic() < self._boot_until
        goal = self.max_value if booting else (self.target or 0.0)
        diff = goal - self.value
        moving = True
        before = (self.value, self.shimmer)
        if not booting and abs(diff) <= SETTLE * max(1.0, self.max_value):
            self.value, moving = goal, False
        else:
            # Critically-damped-ish easing: ~90% of the way in 0.35 s regardless of frame rate.
            k = 1 - pow(0.001, dt / 1.0)
            self.value += diff * min(1.0, k * (2.2 if booting else 1.6))
        self.shimmer = S.quantize_shimmer(shimmer) if self.animate_motion else None
        if (self.value, self.shimmer) != before:  # settled and no shimmer: nothing to recompute
            self.redraw()
        return moving

    # ---- change tracking ----
    def _num(self) -> str:
        return "–" if self.target is None else f"{self.target:.{self.precision}f}"

    def _frac(self) -> float:
        if self.numeric_only:
            return 0.0
        return max(0.0, min(1.0, self.value / self.max_value if self.max_value else 0.0))

    def _layout(self, w: int, h: int) -> _RingLayout:
        key: tuple[Any, ...] = (w, h, self.fit_width, self._num(), self.unit, self.caption, self.sub)
        key += (self.marker, self.side, self.color, self.glow, self.numeric_only)
        if key == self._lay_key:
            return self._lay
        self._lay_key = key
        if self.numeric_only:
            big = anim.big_text(self._num())
            lines: tuple[Line, ...] = ((f"{self._num()} {self.unit}", self.glow, True),)
            if h >= 6 and max(map(len, big)) + len(self.unit) + 1 <= w:
                lines = tuple((line, self.glow, True) for line in big[:-1]) + (
                    (big[-1] + " " + self.unit, self.glow, True),
                )
            if h >= len(lines) + 2:
                lines += ((self.caption, R.INK, False), (self.sub, R.MUTED, False))
            top = max(0, (h - len(lines)) // 2)
            absolute_overlay = {top + i: (max(0, (w - len(s)) // 2), s[:w], c, b)
                                for i, (s, c, b) in enumerate(lines)}
            self._lay = _RingLayout(meter=False, rw=w, overlay=absolute_overlay)
            return self._lay
        if h < 3 or w < 8:
            self._lay = _RingLayout(meter=True)
            return self._lay
        ring_w = min(w, 2 * h + 1)
        roomy = bool(self.side) and w - ring_w - SIDE_GAP >= SIDE_MIN
        rw = ring_w if roomy else w
        g = anim.ring_geometry(rw, h)
        accent = R.mix(self.color, self.glow, 0.8)
        fit = anim.ring_geometry(min(rw, self.fit_width), h) if self.fit_width and self.fit_width < rw else g
        lines, top = ring_lines(fit, self._num(), self.unit, self.caption, self.sub, accent)
        top += g.center_row - fit.center_row
        overlay: dict[int, S.Overlay] = {}
        for i, (s, c, b) in enumerate(lines):
            s = s[: max(1, rw)]
            overlay[top + i] = ((rw - len(s)) // 2, s, c, b)
        lay = _RingLayout(meter=False, rw=rw, overlay=overlay)
        lay.mark = g.marker_cell(self.marker) if self.marker is not None else None
        lay.side = [[] for _ in range(h)]
        lay.side_keys = [None] * h
        if roomy:
            side_w = w - rw - SIDE_GAP
            shown = self.side[:h]
            label_w = max((len(lbl) for lbl, _ in shown), default=0)
            first = max(0, (h - len(shown)) // 2)
            for j, (label, value) in enumerate(shown):
                head = f"{label:<{label_w}} "
                room = side_w - cell_len(head)
                if cell_len(value) > room:
                    value = value[: max(0, room - 1)] + "…" if room > 0 else ""
                lay.side[first + j] = [
                    Segment(" " * SIDE_GAP),
                    Segment(head[:side_w], R.style(R.MUTED)),
                    Segment(value, R.style(R.INK, True)),
                ]
                lay.side_keys[first + j] = (head, value)
        self._lay = lay
        return lay

    def _row_keys(self, w: int, h: int) -> list[Hashable]:
        lay = self._layout(w, h)
        if self.numeric_only:
            return [("absolute", lay.overlay.get(y)) for y in range(h)]
        if lay.meter:
            k = ("meter", round(self._frac() * w * 8), self._num(), self.caption, self.sub, self.color, self.glow)
            return [k] * h
        plan = S.ring_plan(lay.rw, h)
        frac, sh = self._frac(), S.quantize_shimmer(self.shimmer)
        mark = lay.mark
        keys: list[Hashable] = []
        for y in range(h):
            mx = mark[1] if mark is not None and mark[0] == y else None
            ring = S.row_key(plan[y], frac, sh)
            keys.append((ring, lay.overlay.get(y), mx, lay.side_keys[y], self.color, self.glow, lay.rw))
        return keys

    def _sync(self) -> list[int]:
        """Recompute row keys; drop and return the rows that need repainting."""
        w, h = self.content_size
        if w <= 0 or h <= 0:
            return []
        keys = self._row_keys(w, h)
        if (w, h) != self._painted_size or len(keys) != len(self._keys):
            self._painted_size, self._keys = (w, h), keys
            self._strips.clear()
            return list(range(h))
        changed = [y for y, (a, b) in enumerate(zip(keys, self._keys)) if a != b]
        self._keys = keys
        for y in changed:
            self._strips.pop(y, None)
        return changed

    def redraw(self) -> None:
        """Repaint only the rows whose pixels changed since the last frame."""
        if not self.is_mounted:
            return
        changed = self._sync()
        if not changed:
            return
        w = self._painted_size[0]
        if len(changed) == self._painted_size[1]:
            self.refresh(Region(0, 0, w, len(changed)))  # content rows only: keep the cached border lines
        else:
            self.refresh(*(Region(0, y, w, 1) for y in changed))

    # ---- rendering (Line API) ----
    def _meter(self, w: int, h: int) -> Text:
        value = f"{self._num()}{self.unit}"
        t = Text(no_wrap=True, overflow="crop")
        t.append(f"{value:>5} ", Style(color=R.INK, bold=True))
        t.append_text(R.bar(self._frac(), max(1, w - 6), self.color, self.glow))
        if h >= 2:
            t.append("\n")
            t.append(f"{self.caption}  {self.sub}".strip()[:w], Style(color=R.MUTED))
        return t

    def _build(self, y: int, w: int, h: int) -> Strip:
        self.rows_painted += 1
        lay = self._layout(w, h)
        base = self.rich_style
        if self.numeric_only:
            col, text, color, bold = lay.overlay.get(y, (0, "", R.INK, False))
            return Strip([Segment(" " * col), Segment(text, Style(color=color, bold=bold))]).adjust_cell_length(
                w, base,
            ).apply_style(base)
        if lay.meter:
            rows = text_strips(self, self._meter(w, h), w)
            strip = rows[y] if y < len(rows) else Strip.blank(w)
            return strip.apply_style(base)
        plan = S.ring_plan(lay.rw, h)
        mark = lay.mark
        mx = mark[1] if mark is not None and mark[0] == y else None
        segs = S.ring_row(
            plan[y], self._frac(), self.color, self.glow, S.quantize_shimmer(self.shimmer), lay.overlay.get(y), mx
        )
        side = lay.side[y] if y < len(lay.side) else []
        return Strip(Segment.apply_style(segs + side, base)).adjust_cell_length(w, base)

    def render_line(self, y: int) -> Strip:
        w, h = self.content_size
        if (w, h) != self._painted_size:
            self._sync()
        strip = self._strips.get(y)
        if strip is None:
            strip = self._strips[y] = self._build(y, w, h)
        return strip

    def on_resize(self) -> None:
        self.redraw()


# ---- panels --------------------------------------------------------------------------------
_UNSET = object()


def set_border(widget: Widget, title: Any = _UNSET, subtitle: Any = _UNSET) -> None:
    """Set a border title / subtitle only when it changed (each assignment repaints the widget)."""
    seen: dict[str, Any] = widget.__dict__.setdefault("_gp_border", {})
    if title is not _UNSET and seen.get("title", _UNSET) != title:
        seen["title"] = title
        widget.border_title = title
    if subtitle is not _UNSET and seen.get("subtitle", _UNSET) != subtitle:
        seen["subtitle"] = subtitle
        widget.border_subtitle = subtitle


def fingerprint(r: Any) -> Optional[Hashable]:
    """A hashable summary of a renderable's content, or None when it can't be summarised.

    Equal fingerprints mean the panel would draw exactly the same, so ``update`` can be skipped.
    """
    if isinstance(r, str):
        return r
    if isinstance(r, Text):
        return ("T", r.plain, tuple(r.spans), r.style, r.justify, r.overflow, r.no_wrap, r.end)
    if isinstance(r, Group):
        parts = tuple(fingerprint(x) for x in r.renderables)
        return None if any(p is None for p in parts) else ("G", parts)
    if isinstance(r, Table):
        try:
            cols = tuple(
                (fingerprint(c.header), c.ratio, c.width, c.justify, c.no_wrap, c.overflow, c.style)
                + tuple(fingerprint(x) for x in c._cells)
                for c in r.columns
            )
        except AttributeError:  # Rich internals changed: always update
            return None
        if any(p is None for col in cols for p in col[7:]):
            return None
        rows = tuple((row.style, row.end_section) for row in r.rows)
        return ("Tb", cols, rows, r.expand, r.show_header, r.box, r.padding, r.pad_edge, r.width)
    return None


class QuietStatic(Static):
    """A ``Static`` whose ``update`` is skipped when the content is unchanged.

    A layout pass is only requested when the widget's height depends on its content
    (``height: auto``); fixed-size panels just repaint.
    """

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._fp: Optional[Hashable] = None
        self.updates = 0  # content changes applied (cost accounting for tests and benchmarks)

    def update(self, content: Any = "", *, layout: Optional[bool] = None) -> None:  # type: ignore[override]
        fp = fingerprint(content)
        if fp is not None and fp == self._fp:
            return
        self._fp = fp
        self.updates += 1
        if layout is None:
            h = self.styles.height
            layout = h is None or h.is_auto
        super().update(content, layout=layout)


class Panel(QuietStatic):
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
        self._fp = None
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

    # Scroll speed in dot columns per second. An idle GPU gets a slow "resting pulse": the trace
    # scrolls (and repaints) ~4 times a second instead of every frame, which is most of the
    # Overview's cost on an idle machine. Speed follows the smoothed utilization up to full rate.
    COLUMNS_PER_S = 36
    RESTING_COLUMNS_PER_S = 4

    def __init__(self, **kw) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kw)
        self.border_title = "PULSE · beats faster as the GPU works harder"
        self.wave = anim.PulseWave()
        self.util: Optional[float] = None
        self._carry = 0.0
        self._rows: Optional[list[Strip]] = None
        self._rows_size = (0, 0)

    def step(self, dt: float) -> bool:
        """Scroll the trace; returns True when it moved (and a repaint was requested)."""
        self._carry += dt * self.columns_per_s()
        n = int(self._carry)
        if not n:
            return False
        self._carry -= n
        self.wave.advance(self.util, n)
        self._rows = None
        w, h = self.content_size
        self.refresh(Region(0, 0, w, h))  # content rows only: the border's cached lines stay valid
        return True

    def columns_per_s(self) -> float:
        # Busy-ness from the live reading or the wave's smoothed level, whichever is higher, so a
        # job starting speeds the trace up at once instead of after the smoothing catches up.
        busy = max(self.wave.level, min(1.0, (self.util or 0.0) / 100))
        rest = self.RESTING_COLUMNS_PER_S
        return rest + (self.COLUMNS_PER_S - rest) * min(1.0, busy / 0.3)

    def render_line(self, y: int) -> Strip:
        w, h = self.content_size
        if self._rows is None or self._rows_size != (w, h):
            base = self.rich_style
            rows = S.pulse_rows(self.wave, max(1, w), max(1, h), R.GPU, R.GPU_GLOW)
            self._rows = [Strip(Segment.apply_style(r, base)).adjust_cell_length(w, base) for r in rows]
            self._rows_size = (w, h)
        return self._rows[y] if y < len(self._rows) else Strip.blank(w)
