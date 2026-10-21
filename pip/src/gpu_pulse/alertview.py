"""Dashboard side of alerts: recent list, mute state and the ALERTS panel (Rich only, no Textual)."""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text

from . import render as R
from .alerts import Alert, AlertConfig, AlertEngine
from .models import Sample

KEEP = 20  # alerts listed on the Diagnosis tab
MUTE_S = 3600.0


class AlertCenter:
    """Runs the engine on each sample and keeps what the dashboard shows (recent alerts, mute)."""

    def __init__(self, cfg: AlertConfig) -> None:
        self.engine = AlertEngine(cfg)
        self.recent: deque[Alert] = deque(maxlen=KEEP)  # newest first
        self.muted_until = 0.0  # time.monotonic() deadline; 0 = not muted
        self.version = 0  # bumped when the list changes, so the panel redraws only then

    def process(self, sample: Sample) -> list[Alert]:
        """Alerts fired by ``sample``, already recorded in :attr:`recent`."""
        new = self.engine.update(sample)
        for a in new:
            self.recent.appendleft(a)
        if new:
            self.version += 1
        return new

    def muted(self, now: Optional[float] = None) -> bool:
        return (time.monotonic() if now is None else now) < self.muted_until

    def toggle_mute(self, now: Optional[float] = None) -> bool:
        """Mute for an hour, or unmute; returns the new muted state."""
        now = time.monotonic() if now is None else now
        self.muted_until = 0.0 if self.muted(now) else now + MUTE_S
        return self.muted(now)

    def active_count(self) -> int:
        return len(self.engine.active())


def rules_summary(cfg: AlertConfig) -> str:
    return (
        f"Watching: ≥ {cfg.temp_warn:g} °C for {cfg.hot_s:g} s · VRAM ≥ {cfg.vram_full_pct:g}% for "
        f"{cfg.vram_full_s:g} s · idle while holding VRAM {cfg.idle_held_min:g} min · "
        f"job finished (≥ {cfg.job_min:g} min)"
    )


def alerts_list(center: Optional[AlertCenter], width: int) -> RenderableType:
    """The ALERTS panel body: newest first, one line plus a dim detail line per alert."""
    if center is None:
        return Text("Alerts are off (--no-alerts).", Style(color=R.MUTED))
    rows: list[RenderableType] = []
    for i, a in enumerate(center.recent):
        color = R.LEVEL_COLOR.get(a.level, R.MUTED)
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(time.strftime("%H:%M:%S ", time.localtime(a.t)), Style(color=R.DIM))
        line.append(f"{R.LEVEL_ICON.get(a.level, '·')} ", Style(color=color, bold=True))
        line.append(a.title, Style(color=R.INK, bold=True))
        rows.append(line)
        if i == 0 and width >= 40:  # detail for the newest only: 20 alerts stay ~20 lines
            rows.append(Text(f"           {a.detail}", Style(color=R.MUTED), overflow="fold"))
    if not rows:
        rows.append(Text("✔ No alerts yet.", Style(color=R.LEVEL_COLOR["good"])))
    rows.append(Text(rules_summary(center.engine.cfg), Style(color=R.DIM), overflow="fold"))
    if center.muted():
        rows.append(Text("Muted: no toasts or bell for up to an hour (m to unmute).", Style(color=R.MUTED)))
    return Group(*rows)
