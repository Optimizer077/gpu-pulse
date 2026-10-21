"""Alert rules evaluated on every sample (pure logic: no UI, no I/O).

Four rules per GPU, timed with sample timestamps (the VS Code extension implements the same ones):

* ``hot``          temp >= temp_warn continuously for 30 s; clears below temp_warn - 3
* ``vram_full``    VRAM >= 95% continuously for 20 s; clears below 90%
* ``idle_held``    util < 10% and VRAM >= 50% continuously for ``idle_held_min``; clears at util >= 15% or VRAM < 40%
* ``job_finished`` a busy stretch (util >= 15%, gaps < 60 s bridged) of >= ``job_min`` ends with util < 10% for 60 s

Common behaviour: an alert fires once when its condition becomes true and re-arms only after the
clear condition (hysteresis); each (rule, GPU uuid) has a 30-minute cooldown, and a condition that
becomes true inside the cooldown is latched silently (it does not fire later when the cooldown
ends). A sample gap over ``gap_factor`` × the interval restarts the "continuously" timers. ``None``
readings satisfy neither a condition nor its clear condition. Work per sample is O(GPUs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from .models import GpuSample, Sample

AlertLevel = Literal["warning", "info"]
RULES = ("hot", "vram_full", "idle_held", "job_finished")
MIB_PER_GB = 1024.0


@dataclass(frozen=True)
class AlertConfig:
    temp_warn: float = 80.0
    interval_s: float = 1.0  # expected seconds between samples (for gap detection)
    idle_held_min: float = 15.0
    job_min: float = 10.0
    hot_s: float = 30.0
    hot_clear_delta: float = 3.0
    vram_full_pct: float = 95.0
    vram_full_s: float = 20.0
    vram_clear_pct: float = 90.0
    idle_util: float = 10.0  # below = idle / quiet
    busy_util: float = 15.0  # at or above = busy
    idle_vram_pct: float = 50.0
    idle_vram_clear_pct: float = 40.0
    job_bridge_s: float = 60.0  # non-busy gaps shorter than this stay inside one busy stretch
    job_quiet_s: float = 60.0  # util < idle_util this long ends the stretch
    cooldown_s: float = 1800.0
    gap_factor: float = 3.0


@dataclass(frozen=True)
class Alert:
    rule: str
    gpu_uuid: str
    gpu_index: int
    level: AlertLevel
    title: str
    detail: str
    t: float  # epoch seconds of the sample that fired it


@dataclass
class _Timer:
    """A latched "condition held continuously for N seconds" rule."""

    since: Optional[float] = None  # start of the current true run
    latched: bool = False  # fired (or suppressed by cooldown) and not cleared yet


@dataclass
class _Job:
    start: Optional[float] = None  # first busy sample of the open stretch
    last_busy: float = 0.0
    util_sum: float = 0.0  # over the stretch's samples up to last_busy
    n: int = 0
    pend_sum: float = 0.0  # samples since last_busy (folded in if the stretch continues)
    pend_n: int = 0
    quiet_since: Optional[float] = None

    def reset(self) -> None:
        self.start, self.util_sum, self.n, self.pend_sum, self.pend_n, self.quiet_since = None, 0.0, 0, 0.0, 0, None


@dataclass
class _GpuState:
    last_t: Optional[float] = None
    hot: _Timer = field(default_factory=_Timer)
    vram: _Timer = field(default_factory=_Timer)
    idle: _Timer = field(default_factory=_Timer)
    job: _Job = field(default_factory=_Job)
    fired_at: dict[str, float] = field(default_factory=dict)  # rule -> last fire time (cooldown)


def _ge(v: Optional[float], x: float) -> bool:
    return v is not None and v >= x


def _lt(v: Optional[float], x: float) -> bool:
    return v is not None and v < x


def _gb(mib: Optional[float]) -> float:
    return (mib or 0.0) / MIB_PER_GB


class AlertEngine:
    """Feed every sample to :meth:`update`; it returns the alerts that fire on that sample."""

    def __init__(self, cfg: Optional[AlertConfig] = None) -> None:
        self.cfg = cfg or AlertConfig()
        self._gpus: dict[str, _GpuState] = {}
        self._present: tuple[str, ...] = ()

    def update(self, sample: Sample) -> list[Alert]:
        out: list[Alert] = []
        self._present = tuple(g.uuid for g in sample.gpus)
        for g in sample.gpus:
            st = self._gpus.setdefault(g.uuid, _GpuState())
            self._step(st, g, sample.t, out)
        return out

    def active(self) -> list[tuple[str, str]]:
        """(rule, gpu uuid) of the latched conditions on GPUs in the latest sample."""
        out = []
        for uuid in self._present:
            st = self._gpus.get(uuid)
            if st is not None:
                out += [(rule, uuid) for rule, tm in (("hot", st.hot), ("vram_full", st.vram), ("idle_held", st.idle))
                        if tm.latched]  # fmt: skip
        return out

    # ---- per GPU ----
    def _step(self, st: _GpuState, g: GpuSample, t: float, out: list[Alert]) -> None:
        c = self.cfg
        gap = None if st.last_t is None else t - st.last_t
        st.last_t = t
        if gap is not None and (gap < 0 or gap > c.gap_factor * c.interval_s):
            for tm in (st.hot, st.vram, st.idle):
                tm.since = None
            st.job.quiet_since = None
            if gap < 0 or gap >= c.job_bridge_s:  # can't know what ran meanwhile: drop the stretch
                st.job.reset()
        vram = g.vram_pct
        self._timer(st, "hot", st.hot, g, t, _ge(g.temp_c, c.temp_warn), _lt(g.temp_c, c.temp_warn - c.hot_clear_delta),
                    c.hot_s, out)  # fmt: skip
        self._timer(st, "vram_full", st.vram, g, t, _ge(vram, c.vram_full_pct), _lt(vram, c.vram_clear_pct),
                    c.vram_full_s, out)  # fmt: skip
        idle_on = _lt(g.util, c.idle_util) and _ge(vram, c.idle_vram_pct)
        idle_off = _ge(g.util, c.busy_util) or _lt(vram, c.idle_vram_clear_pct)
        self._timer(st, "idle_held", st.idle, g, t, idle_on, idle_off, c.idle_held_min * 60, out)
        self._job(st, g, t, out)

    def _timer(
        self, st: _GpuState, rule: str, tm: _Timer, g: GpuSample, t: float, on: bool, off: bool, hold: float,
        out: list[Alert],
    ) -> None:  # fmt: skip
        if tm.latched:
            if off:
                tm.latched, tm.since = False, None
            return
        if not on:
            tm.since = None
            return
        if tm.since is None:
            tm.since = t
        if t - tm.since >= hold:
            tm.latched = True
            self._fire(st, rule, g, t, t - tm.since, out)

    def _job(self, st: _GpuState, g: GpuSample, t: float, out: list[Alert]) -> None:
        c, j, u = self.cfg, st.job, g.util
        if _ge(u, c.busy_util):
            assert u is not None
            if j.start is None or t - j.last_busy >= c.job_bridge_s:
                j.reset()
                j.start = t
            j.util_sum += j.pend_sum + u
            j.n += j.pend_n + 1
            j.pend_sum, j.pend_n, j.last_busy, j.quiet_since = 0.0, 0, t, None
            return
        if j.start is None:
            return
        if not _lt(u, c.idle_util):
            j.quiet_since = None
        elif j.quiet_since is None:
            j.quiet_since = t
        if u is not None:
            j.pend_sum += u
            j.pend_n += 1
        if j.quiet_since is None:
            if t - j.last_busy >= c.job_bridge_s:  # no quiet spell started within the bridge: not a clean finish
                j.reset()
            return
        if t - j.quiet_since < c.job_quiet_s:
            return
        dur, avg = j.last_busy - j.start, j.util_sum / max(1, j.n)
        j.reset()
        if dur >= c.job_min * 60:
            self._fire(st, "job_finished", g, t, dur, out, avg=avg)

    def _fire(
        self, st: _GpuState, rule: str, g: GpuSample, t: float, held: float, out: list[Alert], avg: float = 0.0
    ) -> None:
        last = st.fired_at.get(rule)
        if last is not None and t - last < self.cfg.cooldown_s:
            return  # cooldown: stays latched (if a timer rule) without notifying
        st.fired_at[rule] = t
        out.append(self._make(rule, g, t, held, avg))

    def _make(self, rule: str, g: GpuSample, t: float, held: float, avg: float) -> Alert:
        c, i = self.cfg, g.index
        if rule == "hot":
            title = f"GPU {i} running hot: {g.temp_c:.0f} °C"
            detail = f"At or above the {c.temp_warn:g} °C warning threshold for {held:.0f} s; check airflow and fans."
            level: AlertLevel = "warning"
        elif rule == "vram_full":
            title = f"GPU {i} VRAM nearly full ({_gb(g.mem_used_mib):.1f}/{_gb(g.mem_total_mib):.0f} GB)"
            detail = f"Above {c.vram_full_pct:g}% for {held:.0f} s; the next allocation may fail with out-of-memory."
            level = "warning"
        elif rule == "idle_held":
            title = f"GPU {i} idle for {held / 60:.0f} min while holding {_gb(g.mem_used_mib):.1f} GB of VRAM"
            detail = "Utilization under 10%; a stopped notebook or finished job may still hold the memory."
            level = "info"
        else:
            title = f"Job on GPU {i} finished after {held / 60:.0f} min (avg {avg:.0f}%)"
            detail = "Utilization has stayed under 10% for a minute after the busy stretch."
            level = "info"
        return Alert(rule, g.uuid, i, level, title, detail, t)


def format_line(a: Alert) -> str:
    """One log line: ``2026-10-21 14:03:07  WARNING  GPU 2 running hot: 86 °C``."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a.t))
    return f"{stamp}  {a.level.upper():7s}  {a.title}"
