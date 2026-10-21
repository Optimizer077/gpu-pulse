"""``gpu-pulse``: full-screen live dashboard (Textual).

Five pages in a ContentSwitcher (Overview · GPU · System · History · Diagnosis). Layout is
recomputed on every resize by :func:`gpu_pulse.layout.plan`. Work is split by cost:

* every animation frame – only the ring gauges and pulse strip of the visible Overview
* every sample (``-i``)  – live readouts of the visible page
* every ~2 s             – window stats, diagnosis, heatmap, sessions, processes
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import ContentSwitcher, Footer, Static

from . import pages
from . import panels as P
from . import render as R
from .intel import Diagnosis
from .layout import Layout, plan
from .models import GpuSample, Sample, Specs
from .monitor import Analysis, Monitor
from .screens import HelpScreen
from .widgets import Panel, PulseStrip, RingGauge

logger = logging.getLogger(__name__)

RANGES = (60, 300, 900, 3600)
SLOW_PANELS_S = 2.0  # windows / heatmap / processes / diagnosis change slowly
TABS = (
    ("overview", "Overview"),
    ("gpu", "GPU"),
    ("system", "System"),
    ("history", "History"),
    ("diagnosis", "Diagnosis"),
)


class TabBar(Static):
    """Header row: brand, numbered tabs (clickable) and live status on the right."""

    DEFAULT_CSS = "TabBar { height: 1; background: #0d1117; }"

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.spans: list[tuple[int, int, str]] = []

    def on_click(self, event: events.Click) -> None:
        for a, b, tab in self.spans:
            if a <= event.x < b:
                self.app.action_show_tab(tab)  # type: ignore[attr-defined]


class GpuDashboard(App[None]):
    TITLE = "GPU Pulse"
    CSS = """
    Screen { background: #080a0e; }
    ContentSwitcher { height: 1fr; }
    ContentSwitcher > VerticalScroll { height: 1fr; scrollbar-size-vertical: 1; }
    #rings { grid-size: 6 1; grid-gutter: 0; height: auto; grid-rows: 13; }
    #ov-arch { height: 1; padding: 0 2; }
    #ov-mid { height: 22; }
    #ov-left { width: 3fr; }
    #ov-center { width: 2fr; }
    #ov-right { width: 2fr; }
    #timeline { height: 1fr; }
    #diag { height: 1fr; }
    #live-o { height: 1fr; }
    #ov-bottom { height: 6; }
    #usage { width: 3fr; height: 6; }
    #io { width: 2fr; height: 6; }
    .row { height: auto; }
    .half { width: 1fr; }
    #wins { grid-size: 4; grid-gutter: 0; height: auto; grid-rows: auto; }
    #heat { width: 70; }
    #sessions { width: 1fr; }
    #arch { width: 1fr; }
    #gpu-side { width: 1fr; height: auto; }
    #sys-top { height: 18; }
    #diag-side { height: auto; }
    #gpu-bottom { height: 16; }
    #gpu-trace { width: 3fr; height: 1fr; }
    #procs { width: 2fr; height: 1fr; }
    #cpu, #io-s { height: 1fr; }
    """
    BINDINGS = [
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("g", "next_gpu", "Next GPU"),
        Binding("r", "next_range", "Range"),
        Binding("e", "export", "Export CSV"),
        Binding("q", "quit", "Quit"),
        *[Binding(str(i + 1), f"show_tab('{tab}')", label, show=False) for i, (tab, label) in enumerate(TABS)],
        Binding("right,right_square_bracket", "cycle_tab(1)", "Next tab", show=False),
        Binding("left,left_square_bracket", "cycle_tab(-1)", "Prev tab", show=False),
    ]

    def __init__(self, monitor: Monitor, fps: float = 20.0) -> None:
        super().__init__()
        self.mon = monitor
        self.fps = max(0.0, min(60.0, fps))
        self.gpu_idx = 0
        self.range_i = 1
        self.tab = "overview"
        self.specs: Optional[Specs] = None
        self.details: list[Any] = []
        self.layout_plan: Layout = plan(120, 40)
        self.diag: Diagnosis = Diagnosis()
        self.analysis: Optional[Analysis] = None
        self._diag_failed = False
        self._last_slow = 0.0
        self._last_frame = time.monotonic()
        self.gauges: list[RingGauge] = []
        self._rerender_pending = False

    # ---- composition ----
    def compose(self) -> ComposeResult:
        yield TabBar(id="header")
        with ContentSwitcher(initial="overview", id="pages"):
            with VerticalScroll(id="overview"):
                with Grid(id="rings"):
                    yield RingGauge("GPU", R.GPU, R.GPU_GLOW, id="ring-gpu")
                    yield RingGauge("VRAM", R.VRAM, R.VRAM_GLOW, id="ring-vram")
                    yield RingGauge("TEMP", R.TEMP, R.TEMP_GLOW, unit="°C", id="ring-temp")
                    yield RingGauge("POWER", R.POWER, R.POWER_GLOW, unit="W", id="ring-power")
                    yield RingGauge("CPU", R.CPU, R.CPU_GLOW, id="ring-cpu")
                    yield RingGauge("RAM", R.RAM, R.RAM_GLOW, id="ring-ram")
                yield Static(id="ov-arch")
                with Horizontal(id="ov-mid"):
                    with Vertical(id="ov-left"):
                        yield PulseStrip(id="pulse")
                        yield Panel("GPU TIMELINE", id="timeline")
                    with Vertical(id="ov-center"):
                        yield Panel("TRUE USAGE", id="truth")
                        yield Panel("SYSTEM HEALTH", id="health")
                        yield Panel("GPU LIVE", id="live-o")
                    with Vertical(id="ov-right"):
                        yield Panel("TRUE USAGE", id="truth-r")
                        yield Panel("DIAGNOSIS", id="diag")
                        yield Panel("SYSTEM HEALTH", id="health-r")
                with Horizontal(id="ov-bottom"):
                    yield Panel("USAGE WINDOWS", id="usage")
                    yield Panel("DISK & NETWORK", id="io")
            with VerticalScroll(id="gpu"):
                with Horizontal(classes="row"):
                    yield Panel("GPU ARCHITECTURE", id="arch")
                    with Vertical(id="gpu-side"):
                        yield Panel("LIVE", id="gpu-live")
                        yield Panel("TRUE USAGE", id="truth-g")
                with Horizontal(classes="row", id="gpu-bottom"):
                    yield Panel("TRACES", id="gpu-trace")
                    yield Panel("GPU PROCESSES", id="procs")
            with VerticalScroll(id="system"):
                with Horizontal(classes="row", id="sys-top"):
                    yield Panel("CPU & MEMORY", id="cpu", classes="half")
                    yield Panel("DISK & NETWORK", id="io-s", classes="half")
                yield Panel("MACHINE", id="specs")
            with VerticalScroll(id="history"):
                with Grid(id="wins"):
                    for i, label in enumerate(("5 MIN", "30 MIN", "1 HOUR", "24 HOURS")):
                        yield Panel(label, id=f"win{i}")
                with Horizontal(classes="row", id="heat-row"):
                    yield Panel("24-HOUR ACTIVITY · one cell per minute", id="heat")
                    yield Panel("SESSIONS · last 24 h", id="sessions")
            with VerticalScroll(id="diagnosis"):
                with Horizontal(classes="row"):
                    yield Panel("DIAGNOSIS", id="diag-full", classes="half")
                    with Vertical(classes="half", id="diag-side"):
                        yield Panel("SYSTEM HEALTH", id="health-full")
                        yield Panel("TRUE USAGE · evidence", id="truth-d")
                yield Panel("INSIGHTS · rules + engine findings", id="insights")
        yield Footer()

    def on_mount(self) -> None:
        self.gauges = list(self.query(RingGauge))
        self.pulse = self.query_one(PulseStrip)
        if self.fps:
            self.set_interval(1 / self.fps, self._frame)
        else:  # --fps 0: values snap, no shimmer; the pulse still advances once per sample
            for gauge in self.gauges:
                gauge.animate_motion = False
        self._apply_layout()
        self.refresh_data()
        self.set_interval(self.mon.cfg.interval_s, self.refresh_data)
        self.run_worker(self._load_specs, thread=True)

    def on_resize(self, event: events.Resize) -> None:
        if not self.gauges:  # resize can arrive before on_mount
            return
        self._apply_layout()

    def _apply_layout(self) -> None:
        w, h = self.size.width or 120, self.size.height or 40
        L = self.layout_plan = plan(w, h)
        rings = self.query_one("#rings", Grid)
        rings.styles.grid_size_columns = L.ring_cols
        rings.styles.grid_size_rows = L.ring_rows
        rings.styles.grid_rows = str(L.ring_h)
        fit = w // L.ring_cols - 2  # narrowest ring's content width (1fr columns differ by one cell)
        for gauge in self.gauges:
            gauge.styles.height = L.ring_h
            gauge.fit_width = fit
        self.query_one("#ov-bottom").styles.height = L.bottom_h
        for pid in ("#usage", "#io"):
            self.query_one(pid).styles.height = L.bottom_h
        self._show("#ov-arch", not L.compact and L.middle_h >= 16)
        self.query_one("#ov-mid").styles.height = L.middle_h - (1 if not L.compact and L.middle_h >= 16 else 0)
        self._show("#ov-bottom", L.show_bottom)
        self._show("#pulse", L.show_pulse)
        self._show("#ov-left", L.columns >= 2)
        self._show("#ov-center", L.columns == 3)
        self._show("#truth-r", L.columns < 3)
        sys_h = max(16, h - 2 - 16)  # leave room for the MACHINE panel below
        self.query_one("#sys-top").styles.height = sys_h if w >= 100 else 2 * max(14, sys_h // 2)
        self.query_one("#gpu-bottom").styles.height = max(14, h - 2 - 22) if w >= 100 else 34
        card_rows = 1 if w >= 120 else 2 if w >= 60 else 4
        card_h = max(12, min(20, h - 2 - 17)) if card_rows == 1 else 12
        wins = self.query_one("#wins", Grid)
        wins.styles.grid_size_columns = 4 // card_rows
        wins.styles.grid_rows = str(card_h)
        for i in range(4):
            self.query_one(f"#win{i}").styles.height = card_h
        narrow = w < 100
        for row in self.query(".row"):
            row.styles.layout = "vertical" if narrow else "horizontal"
        heat_w = 130 if w >= 175 else 70
        self.query_one("#heat").styles.width = heat_w if w >= 74 else "1fr"
        self._show("#heat", w >= 74)
        self.query_one("#heat-row").styles.layout = "vertical" if w < heat_w + 34 else "horizontal"

    def request_rerender(self) -> None:
        """Debounced full redraw of the visible page after panels change size."""
        if not self._rerender_pending:
            self._rerender_pending = True
            self.set_timer(0.05, self._rerender)

    def _rerender(self) -> None:
        self._rerender_pending = False
        self.refresh_data(sample=False, slow=True)

    def _show(self, selector: str, on: bool) -> None:
        self.query_one(selector).display = on

    def _frame(self) -> None:
        now = time.monotonic()
        dt, self._last_frame = min(0.25, now - self._last_frame), now
        if self._shutting_down() or self.tab != "overview" or self.screen is not self.screen_stack[0]:
            return
        for gauge in self.gauges:
            gauge.step(dt)
        if self.layout_plan.show_pulse:
            self.pulse.step(dt)

    def _load_specs(self) -> None:
        # psutil / NVML probes can fail in restricted containers; never let that close the dashboard.
        try:
            details = list(getattr(self.mon, "gpu_details", lambda: [])() or [])
        except Exception as e:  # noqa: BLE001
            logger.warning("GPU details unavailable: %s", e)
            details = []
        self._post(self._set_details, details)
        try:
            specs = self.mon.specs()
        except Exception as e:  # noqa: BLE001
            self._post(self._specs_failed, str(e))
            return
        self._post(self._show_specs, specs)

    def _post(self, fn: Any, *args: Any) -> None:
        """Run ``fn`` on the UI thread from the specs worker, unless the app already closed.

        Spec probes take seconds on Windows; quitting before they finish must not raise
        (the screen stack is gone by then).
        """
        if not self.is_running:
            return

        def safe() -> None:
            try:
                fn(*args)
            except (NoMatches, ScreenStackError):
                pass

        try:
            self.call_from_thread(safe)
        except RuntimeError:  # event loop already stopped
            pass

    def _specs_failed(self, msg: str) -> None:
        self.query_one("#specs", Panel).update(Text(f"Specs unavailable: {msg}", Style(color=R.MUTED)))

    def _set_details(self, details: list[Any]) -> None:
        self.details = details
        self.refresh_data(sample=False, slow=True)

    # ---- actions ----
    def action_show_tab(self, tab: str) -> None:
        if tab == self.tab:
            return
        self.tab = tab
        self.query_one("#pages", ContentSwitcher).current = tab
        self.refresh_data(sample=False, slow=True)  # panels re-render again via request_rerender once sized

    def action_cycle_tab(self, step: int) -> None:
        ids = [t for t, _ in TABS]
        self.action_show_tab(ids[(ids.index(self.tab) + step) % len(ids)])

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_next_gpu(self) -> None:
        n = len(self.mon.latest.gpus) if self.mon.latest else 0
        self.gpu_idx = (self.gpu_idx + 1) % max(1, n)
        self.refresh_data(sample=False, slow=True)

    def action_next_range(self) -> None:
        self.range_i = (self.range_i + 1) % len(RANGES)
        self.refresh_data(sample=False)

    def action_export(self) -> None:
        name = f"gpu-pulse-history-{time.strftime('%Y%m%d-%H%M%S')}.csv"
        for folder in (Path.cwd(), Path.home()):  # cwd may be read-only (/, System32)
            try:
                (folder / name).write_text(self.mon.history.to_csv(), encoding="utf-8")
            except OSError:
                continue
            self.notify(f"Exported to {folder / name}")
            return
        self.notify("Export failed: no writable folder (tried current and home directory)", severity="error")

    # ---- refresh ----
    def _gpu(self, s: Sample) -> Optional[GpuSample]:
        return s.gpus[self.gpu_idx] if self.gpu_idx < len(s.gpus) else (s.gpus[0] if s.gpus else None)

    def detail_for(self, g: Optional[GpuSample]) -> Any:
        if g is None:
            return None
        return next((d for d in self.details if getattr(d, "uuid", None) == g.uuid), None) or (
            self.details[g.index] if g.index < len(self.details) else None
        )

    def _shutting_down(self) -> bool:  # not `_closing`: Textual's MessagePump already uses that name
        """True once the app is shutting down (timers can still fire once while screens unmount)."""
        return not self.is_running or not self.screen_stack

    def refresh_data(self, sample: bool = True, slow: bool = False) -> None:
        if self._shutting_down():
            return
        try:
            self._refresh(sample, slow)
        except (NoMatches, ScreenStackError):
            if not self._shutting_down():  # a real layout bug, not teardown: surface it
                raise

    def _refresh(self, sample: bool, slow: bool) -> None:
        s = self.mon.tick() if sample else self.mon.latest
        if s is None:
            return
        g = self._gpu(s)
        now = time.monotonic()
        if slow or self.analysis is None or now - self._last_slow >= SLOW_PANELS_S:
            self._last_slow = now
            self._slow(s, g)
        a = self.analysis
        assert a is not None
        self._header(s, g)
        self._rings(s, g, a)
        pages.live(self, s, g, a)
        if not self.fps and self.tab == "overview":
            self.pulse.step(self.mon.cfg.interval_s)

    def _slow(self, s: Sample, g: Optional[GpuSample]) -> None:
        uid = g.uuid if g else None
        self.analysis = self.mon.analysis(uid)
        try:
            self.diag = self.mon.diagnose(uid)
        except Exception as e:  # noqa: BLE001 - a diagnosis bug must not take the dashboard down
            if not self._diag_failed:
                logger.warning("diagnose() failed: %s", e)
            self._diag_failed = True
            self.diag = Diagnosis()
        pages.slow(self, s, g, self.analysis)

    def _header(self, s: Sample, g: Optional[GpuSample]) -> None:
        bar = self.query_one("#header", TabBar)
        width = self.size.width or 120
        t = Text(no_wrap=True, overflow="crop")
        t.append(" GPU PULSE ", Style(color=R.BG, bgcolor=R.GPU_GLOW, bold=True))
        t.append(" ")
        spans: list[tuple[int, int, str]] = []
        short = width < 100
        for i, (tab, label) in enumerate(TABS):
            text = f" {i + 1} {label[:4] if short and len(label) > 5 else label} "
            start = t.cell_len
            if tab == self.tab:
                t.append(text, Style(color=R.INK, bgcolor="#1b2a3a", bold=True))
            else:
                t.append(f" {i + 1}", Style(color=R.GPU_GLOW))
                t.append(text[2:], Style(color=R.MUTED))
            spans.append((start, t.cell_len, tab))
        bar.spans = spans
        right = Text(no_wrap=True)
        if self.mon.cfg.demo:
            right.append(" DEMO ", Style(color=R.BG, bgcolor="#fab219", bold=True))
            right.append(" ")
        if len(s.gpus) > 1 and g:
            right.append(f"GPU {g.index + 1}/{len(s.gpus)} ", Style(color=R.INK))
        if self.specs and width >= 150:
            right.append(f"{self.specs.hostname} · ", Style(color=R.MUTED))
        right.append("● ", Style(color=R.LEVEL_COLOR["good"]))
        right.append(f"{self.mon.cfg.interval_s:g}s", Style(color=R.MUTED))
        if self.fps and width >= 110:
            right.append(f" · {self.fps:g} fps", Style(color=R.MUTED))
        right.append("  ? help ", Style(color=R.DIM))
        gap = width - t.cell_len - right.cell_len
        if gap >= 1:
            t.append(" " * gap)
            t.append_text(right)
        bar.update(t)

    def _rings(self, s: Sample, g: Optional[GpuSample], a: Analysis) -> None:
        if self.tab == "overview":
            pages.rings(self, s, g, a)

    def _show_specs(self, sp: Specs) -> None:
        self.specs = sp
        self.query_one("#specs", Panel).update(P.specs_block(sp))
