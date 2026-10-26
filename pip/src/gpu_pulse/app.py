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
from typing import Any, Literal, Optional

from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import ContentSwitcher, Footer

from . import pages
from . import panels as P
from . import render as R
from .alerts import Alert, AlertConfig
from .alertview import AlertCenter, alerts_list
from .fleet import Fleet
from .fleetview import Hit, fleet_lines
from .intel import Diagnosis
from .layout import Layout, plan
from .models import GpuSample, Sample, Specs
from .monitor import Analysis, Monitor
from .screens import HelpScreen
from .widgets import Panel, PulseStrip, QuietStatic, RingGauge

logger = logging.getLogger(__name__)

RANGES = (60, 300, 900, 3600)
SLOW_PANELS_S = 2.0  # windows / heatmap / processes / diagnosis change slowly
HISTORY_SLOW_S = 5.0  # the History tab shows per-minute data: redraw its heatmap less often
GPU_PREFIX_S = 1.5  # `g` then a digit within this time jumps to that GPU
SHIMMER_EVERY_S = 9.0  # at most one highlight pass along the rings every 9 s ...
SHIMMER_SPEED = 0.45  # ... travelling 45% of the sweep per second (a pass takes ~2.2 s)
TABS = (
    ("overview", "Overview"),
    ("gpu", "GPU"),
    ("system", "System"),
    ("history", "History"),
    ("diagnosis", "Diagnosis"),
)


class TabBar(QuietStatic):
    """Header row: brand, numbered tabs (clickable) and live status on the right."""

    DEFAULT_CSS = "TabBar { height: 1; background: #0d1117; }"

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.spans: list[tuple[int, int, str]] = []

    def on_click(self, event: events.Click) -> None:
        for a, b, tab in self.spans:
            if a <= event.x < b:
                self.app.action_show_tab(tab)  # type: ignore[attr-defined]


class FleetPanel(Panel):
    """The ALL GPUS panel; clicking a GPU's line selects it."""

    def __init__(self, title: str, **kw: Any) -> None:
        super().__init__(title, **kw)
        self.hits: list[Hit] = []

    def on_click(self, event: events.Click) -> None:
        off = event.get_content_offset(self)
        if off is None:
            return
        for hit in self.hits:
            if hit.line == off.y and hit.x0 <= off.x < hit.x1:
                self.app.select_gpu(hit.pos)  # type: ignore[attr-defined]
                return


class MainScreen(Screen[None]):
    """The dashboard's base screen (the pages live on it; Help is pushed on top)."""

    def on_screen_resume(self) -> None:
        # Back from the help overlay: restart the animation now, not at the next sample.
        self.app.wake()  # type: ignore[attr-defined]

    def on_key(self, event: events.Key) -> None:
        # Runs before the app's bindings: any key but `g` or a digit ends the "g then digit" prefix.
        if event.key != "g" and not event.key.isdigit():
            self.app.end_gpu_prefix()  # type: ignore[attr-defined]


class GpuDashboard(App[None]):
    TITLE = "GPU Pulse"
    CSS = """
    Screen { background: #080a0e; }
    ContentSwitcher { height: 1fr; }
    ContentSwitcher > VerticalScroll { height: 1fr; scrollbar-size-vertical: 1; }
    #rings { grid-size: 6 1; grid-gutter: 0; height: auto; grid-rows: 13; }
    #ov-arch { height: 1; padding: 0 2; }
    #fleet { height: 6; }
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
        Binding("G,shift+g", "next_gpu(-1)", "Prev GPU", show=False),
        Binding("r", "next_range", "Range"),
        Binding("e", "export", "Export CSV"),
        Binding("m", "mute_alerts", "Mute alerts"),
        Binding("q", "quit", "Quit"),
        *[Binding(str(d), f"digit({d})", "", show=False) for d in range(10)],
        Binding("right,right_square_bracket", "cycle_tab(1)", "Next tab", show=False),
        Binding("left,left_square_bracket", "cycle_tab(-1)", "Prev tab", show=False),
    ]

    def __init__(
        self,
        monitor: Monitor,
        fps: float = 12.0,
        alerts: bool = True,
        alert_config: Optional[AlertConfig] = None,
        bell: bool = False,
    ) -> None:
        super().__init__()
        self.mon = monitor
        self.fps = max(0.0, min(60.0, fps))
        cfg = alert_config or AlertConfig(temp_warn=monitor.cfg.temp_warn, interval_s=monitor.cfg.interval_s)
        self.alert_center: Optional[AlertCenter] = AlertCenter(cfg) if alerts else None
        self.bell_on_alert = bell
        self._alerts_drawn = -1  # AlertCenter.version last rendered into the ALERTS panel
        self.gpu_idx = 0  # position of the shown GPU in the latest sample (resolved from gpu_uuid every sample)
        self.gpu_uuid: Optional[str] = None  # the selected GPU; survives other GPUs dropping out
        self._shown_uuid: Optional[str] = None  # GPU the per-GPU panels were last built for
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
        self._t0 = time.monotonic()
        self._frame_timer: Optional[Timer] = None
        self._wake_timer: Optional[Timer] = None
        self.frames_paused = False
        self.frames = 0  # frame callbacks run (cost accounting for tests and benchmarks)
        self._shimmer_cycle = -1
        self._shimmer_on = True
        self.n_gpus = 0  # GPUs in the latest sample; the ALL GPUS panel appears for more than one
        self.fleet_lines = 0
        self.fleet_width = 116  # content width the ALL GPUS panel is planned and drawn at
        self.fleet: Optional[Fleet] = None
        self._gpu_prefix_until = 0.0  # after `g`, digits select that GPU index instead of a tab
        self._gpu_digits = ""  # digits typed since `g` (two for GPU 10+)
        self._shimmer_mask: list[bool] = []  # per gauge: its value moved enough to earn the current pass

    def get_default_screen(self) -> Screen:
        return MainScreen(id="_default")

    @property
    def apple_layout(self) -> bool:
        return self.mon.vendor == "apple" and not self.mon.cfg.demo

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
                yield QuietStatic(id="ov-arch")
                yield FleetPanel("ALL GPUS · g / G or click to select", id="fleet")
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
                        yield Panel("ALERTS · newest first · m mutes", id="alerts")
                        yield Panel("SYSTEM HEALTH", id="health-full")
                        yield Panel("TRUE USAGE · evidence", id="truth-d")
                yield Panel("INSIGHTS · rules + engine findings", id="insights")
        yield Footer()

    def on_mount(self) -> None:
        self.gauges = list(self.query(RingGauge))
        self.pulse = self.query_one(PulseStrip)
        if self.fps:
            self._frame_timer = self.set_interval(1 / self.fps, self._frame)
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

    def _overview_scrollbar(self) -> bool:
        try:
            return bool(self.query_one("#overview", VerticalScroll).show_vertical_scrollbar)
        except NoMatches:
            return False

    def fit_fleet(self, width: int) -> None:
        """Re-plan the ALL GPUS panel at its real content width (the Overview's scrollbar came or went)."""
        if width != self.fleet_width:
            self._apply_layout(width)

    def _apply_layout(self, fleet_width: Optional[int] = None) -> None:
        w, h = self.size.width or 120, self.size.height or 40
        # Panel content width = width - border - padding (- the Overview's scrollbar while it shows).
        # pages.live checks it against the panel's real width and calls fit_fleet when they disagree.
        self.fleet_width = fleet_width or max(20, w - 4 - (1 if self._overview_scrollbar() else 0))
        self.fleet_lines = fleet_lines(self.n_gpus, self.fleet_width, h)
        fleet_h = self.fleet_lines + 3 if self.fleet_lines else 0
        apple = self.apple_layout
        L = self.layout_plan = plan(w, h, fleet_h, ring_count=4 if apple else 6)
        fleet_panel = self.query_one("#fleet")
        fleet_panel.display = fleet_h > 0
        if fleet_h:
            fleet_panel.styles.height = fleet_h
        rings = self.query_one("#rings", Grid)
        rings.styles.grid_size_columns = L.ring_cols
        rings.styles.grid_size_rows = L.ring_rows
        rings.styles.grid_rows = str(L.ring_h)
        fit = w // L.ring_cols - 2  # narrowest ring's content width (1fr columns differ by one cell)
        for gauge in self.gauges:
            gauge.styles.height = L.ring_h
            gauge.fit_width = fit
        self._show("#ring-temp", not apple)
        self._show("#ring-power", not apple)
        for pid in ("truth", "truth-g", "truth-d", "gpu-side", "procs"):
            self._show(f"#{pid}", not apple)
        self.query_one("#arch").styles.height = 12 if apple else "auto"
        self.query_one("#ov-bottom").styles.height = L.bottom_h
        for pid in ("#usage", "#io"):
            self.query_one(pid).styles.height = L.bottom_h
        show_arch = not L.compact and (apple or L.middle_h >= 16)
        self._show("#ov-arch", show_arch)
        self.query_one("#ov-mid").styles.height = L.middle_h - int(show_arch)
        self._show("#ov-bottom", L.show_bottom)
        self._show("#pulse", L.show_pulse)
        self._show("#ov-left", L.columns >= 2)
        self._show("#ov-center", L.columns == 3)
        self._show("#truth-r", L.columns < 3 and not apple)
        sys_h = max(16, h - 2 - 16)  # leave room for the MACHINE panel below
        self.query_one("#sys-top").styles.height = sys_h if w >= 100 else 2 * max(14, sys_h // 2)
        self.query_one("#gpu-bottom").styles.height = (
            max(10, h - 2 - 12) if apple else max(14, h - 2 - 22) if w >= 100 else 34
        )
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
        self.wake()

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

    # ---- animation clock ----
    def _shimmer_at(self, now: float) -> Optional[float]:
        """Position (0..1) of the shimmer pass, or None between passes.

        A pass runs at most every SHIMMER_EVERY_S and only on rings whose shown value moved
        noticeably since their last pass (``_shimmer_mask``): a frozen or jittering reading
        (idle machine, stalled job, CPU wobbling by a percent) leaves the rings still.
        """
        cycle = int((now - self._t0) // SHIMMER_EVERY_S)
        if cycle != self._shimmer_cycle:
            self._shimmer_cycle = cycle
            self._shimmer_mask = [g.consume_change() for g in self.gauges]  # every flag, no short-circuit
            self._shimmer_on = any(self._shimmer_mask)
        pos = ((now - self._t0) % SHIMMER_EVERY_S) * SHIMMER_SPEED
        return pos if self._shimmer_on and pos <= 1.0 else None

    def _animating_here(self) -> bool:
        return (
            bool(self.fps)
            and self.tab == "overview"
            and bool(self.screen_stack)
            and (self.screen is self.screen_stack[0])
        )

    def wake(self) -> None:
        """Restart the frame timer (new sample, tab switch, resize, shimmer due)."""
        timer = self._frame_timer
        if timer is None or not self.frames_paused or not self._animating_here():
            return
        self.frames_paused = False
        self._last_frame = time.monotonic()
        timer.resume()

    def _sleep(self, now: float) -> None:
        """Nothing is moving: stop frames until the next sample or shimmer pass."""
        if self._frame_timer is None or self.frames_paused:
            return
        self.frames_paused = True
        self._frame_timer.pause()
        if self._wake_timer is not None:
            self._wake_timer.stop()
        due = SHIMMER_EVERY_S - (now - self._t0) % SHIMMER_EVERY_S
        self._wake_timer = self.set_timer(due + 0.01, self.wake)

    def _frame(self) -> None:
        now = time.monotonic()
        self.frames += 1
        if self._shutting_down():
            return
        if not self._animating_here():
            self._sleep(now)
            return
        dt, self._last_frame = min(0.25, now - self._last_frame), now
        shimmer = self._shimmer_at(now)
        moving = shimmer is not None
        mask = self._shimmer_mask
        for i, gauge in enumerate(self.gauges):
            moving = gauge.step(dt, shimmer if i < len(mask) and mask[i] else None) or moving
        if self.layout_plan.show_pulse and self.pulse.display:
            self.pulse.step(dt)
            moving = True
        if not moving:
            self._sleep(now)

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
        self.end_gpu_prefix()
        if tab == self.tab:
            return
        self.tab = tab
        self.query_one("#pages", ContentSwitcher).current = tab
        self.refresh_data(sample=False, slow=True)  # panels re-render again via request_rerender once sized

    def action_cycle_tab(self, step: int) -> None:
        ids = [t for t, _ in TABS]
        self.action_show_tab(ids[(ids.index(self.tab) + step) % len(ids)])

    def action_help(self) -> None:
        self.push_screen(HelpScreen(shared_memory=self.apple_layout))

    def end_gpu_prefix(self) -> None:
        """Forget a pending ``g``: the next digit switches tabs again."""
        self._gpu_prefix_until = 0.0
        self._gpu_digits = ""

    def action_next_gpu(self, step: int = 1) -> None:
        """``g`` / ``G``: next / previous GPU. For ~1.5 s after ``g``, digits select GPU <digits>."""
        s = self.mon.latest
        n = len(s.gpus) if s else 0
        self.end_gpu_prefix()
        if step > 0 and n > 1:
            self._gpu_prefix_until = time.monotonic() + GPU_PREFIX_S
        pos = self._resolve(s) if s else self.gpu_idx
        self.select_gpu((pos + step) % max(1, n))

    def select_gpu(self, pos: int) -> None:
        """Show GPU at position ``pos`` of the latest sample on every per-GPU panel (remembered by uuid)."""
        gpus = self.mon.latest.gpus if self.mon.latest else ()
        pos = max(0, min(pos, len(gpus) - 1)) if gpus else 0
        uuid = gpus[pos].uuid if gpus else None
        if pos == self.gpu_idx and uuid == self.gpu_uuid and self.analysis is not None:
            return
        self.gpu_idx, self.gpu_uuid = pos, uuid
        self.refresh_data(sample=False, slow=True)

    def action_digit(self, d: int) -> None:
        """1-5 switch tabs; right after ``g`` digits pick the GPU with that NVML index instead.

        While the prefix is active every digit is taken as part of a GPU index (never a tab): a
        second digit within the window extends it (``g 1 2`` → GPU 12 on machines with 10+ GPUs).
        """
        gpus = self.mon.latest.gpus if self.mon.latest else ()
        if time.monotonic() < self._gpu_prefix_until and len(gpus) > 1:
            digits = self._gpu_digits + str(d)
            indices = [str(g.index) for g in gpus]
            longer = any(i != digits and i.startswith(digits) for i in indices)
            if longer:  # e.g. "1" on a 12-GPU machine: GPU 1 now, GPU 1x if another digit follows
                self._gpu_digits = digits
                self._gpu_prefix_until = time.monotonic() + GPU_PREFIX_S
            else:
                self.end_gpu_prefix()
            pos = next((i for i, g in enumerate(gpus) if str(g.index) == digits), None)
            if pos is not None:
                self.select_gpu(pos)
            elif not longer:
                self.notify(f"No GPU {digits} on this machine", severity="warning", timeout=2)
            return
        if 1 <= d <= len(TABS):
            self.action_show_tab(TABS[d - 1][0])

    def action_next_range(self) -> None:
        self.range_i = (self.range_i + 1) % len(RANGES)
        self.refresh_data(sample=False)

    def action_mute_alerts(self) -> None:
        center = self.alert_center
        if center is None:
            self.notify("Alerts are off (started with --no-alerts)", timeout=3)
            return
        muted = center.toggle_mute()
        self.notify("Alerts muted for 1 hour (m to unmute)" if muted else "Alerts unmuted", timeout=3)
        self._alerts_drawn = -1
        self.refresh_data(sample=False)

    def _alerts(self, s: Sample) -> None:
        """Run the alert rules on a new sample (sample path only, never per frame); toast what fired."""
        center = self.alert_center
        if center is None:
            return
        new = center.process(s)
        if not new or center.muted():
            return
        for a in new:
            self._toast(a)

    def _toast(self, a: Alert) -> None:
        severity: Literal["warning", "information"] = "warning" if a.level == "warning" else "information"
        self.notify(a.detail, title=a.title, severity=severity, timeout=10 if a.level == "warning" else 8, markup=False)
        if self.bell_on_alert:
            self.bell()

    def draw_alerts(self, force: bool = False) -> None:
        """Redraw the ALERTS panel (Diagnosis tab) when the list changed."""
        center = self.alert_center
        version = center.version if center else 0
        if self.tab != "diagnosis" or (version == self._alerts_drawn and not force):
            return
        self._alerts_drawn = version
        panel = self.query_one("#alerts", Panel)
        panel.update(alerts_list(center, panel.inner_width, shared_memory=self.apple_layout))

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
    def _resolve(self, s: Sample) -> int:
        """Position of the selected GPU in ``s``: found by uuid; only if it is gone, the old position clamped."""
        pos = next((i for i, g in enumerate(s.gpus) if g.uuid == self.gpu_uuid), None)
        if pos is None:  # first sample, or the selected GPU dropped out (driver reset)
            pos = max(0, min(self.gpu_idx, len(s.gpus) - 1))
        return pos

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
        if sample:
            self._alerts(s)
        if len(s.gpus) != self.n_gpus:  # first sample, or a GPU appeared / dropped out
            self.n_gpus = len(s.gpus)
            self._apply_layout()
            slow = True
        self.gpu_idx = self._resolve(s)
        g = self._gpu(s)
        if g is not None and g.uuid != self._shown_uuid:  # a different physical GPU: rebuild its panels
            self._shown_uuid = g.uuid
            slow = True
        if self.gpu_uuid is None and g is not None:
            self.gpu_uuid = g.uuid
        now = time.monotonic()
        period = HISTORY_SLOW_S if self.tab == "history" else SLOW_PANELS_S
        if slow or self.analysis is None or now - self._last_slow >= period:
            self._last_slow = now
            self._slow(s, g)
        a = self.analysis
        assert a is not None
        self._header(s, g)
        self._rings(s, g, a)
        pages.live(self, s, g, a)
        self.draw_alerts()
        if not self.fps and self.tab == "overview":
            self.pulse.step(self.mon.cfg.interval_s)
        self.wake()

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
        self.fleet = self.mon.fleet() if len(s.gpus) > 1 else None
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
        if len(s.gpus) > 1 and g:  # NVML index (as nvidia-smi / CUDA show it), then position
            right.append(f"GPU {g.index} ", Style(color=R.INK, bold=True))
            right.append(f"{min(self.gpu_idx, len(s.gpus) - 1) + 1}/{len(s.gpus)} ", Style(color=R.MUTED))
        if self.specs and width >= 150:
            right.append(f"{self.specs.hostname} · ", Style(color=R.MUTED))
        center = self.alert_center
        if center is not None and center.muted():
            right.append("alerts muted ", Style(color=R.MUTED))
        elif center is not None and (n := center.active_count()):
            right.append(f"▲ {n} alert{'s' if n > 1 else ''} ", Style(color=R.LEVEL_COLOR["warning"], bold=True))
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
