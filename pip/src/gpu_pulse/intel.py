"""Usage intelligence: bottleneck diagnosis, sessions, forecasts and an efficiency score.

CONTRACT (shared by the analysis engine and the UI; keep these names/types stable):

    diagnose(history, gpu_uuid, now=None, temp_warn=80.0, details=None) -> Diagnosis

``history`` is a HistoryStore (raw samples in ``history.recent`` for the last hour, per-minute
buckets via ``history.stats`` / ``history.heatmap``). ``details`` is the matching ``GpuDetails``
(optional; enables PCIe-link findings). The UI calls this every ~2 s, so it must be cheap (well
under 5 ms) and must never raise on sparse/empty data, a missing GPU, or ``None`` fields.

How the call is made (rules live in ``_intel_gpu`` / ``_intel_system``):

* The recent window is the last 3 minutes of raw samples (per-minute stats if there are fewer
  than 20). Every rule that fires proposes a ``bottleneck`` with a confidence; the highest wins
  and the runner-up is quoted in the evidence.
* ``utilization.gpu`` only says *a kernel was running*. When NVML GPM is available, SM / Tensor /
  DRAM activity ("true usage") is used instead, and the evidence says which basis was used.
"""

from __future__ import annotations

import bisect
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, cast
from weakref import WeakKeyDictionary

from ._intel_findings import gpm_findings, gpu_findings, gpu_health, vram_forecast
from ._intel_gpu import (
    ACTIVE_UTIL,
    Call,
    GRow,
    gpu_calls,
    gpu_features,
    hour_extras,
    input_strength,
    minute_calls,
)
from ._intel_system import p95, system_calls, system_features, system_findings, system_health
from .history import MINUTE_MS, RECENT_S, HistoryStore
from .models import GpuDetails, GpuSample, Insight, InsightLevel, Sample, WindowStats

logger = logging.getLogger(__name__)

Bottleneck = Literal[
    "none",  # GPU busy and unconstrained
    "idle",  # nothing running
    "input",  # GPU waits on data loading / CPU work (bursty utilization, high CPU)
    "cpu",  # CPU saturated (or one core pegged) while GPU underused
    "power",  # clocks held down by the power cap
    "thermal",  # clocks held down by temperature
    "memory",  # memory-bandwidth bound (GPM DRAM bandwidth, else memory-controller heuristic)
    "vram_held",  # VRAM allocated, GPU idle (e.g. an open notebook)
    "ram",  # system memory pressure / swapping
    "disk",  # bursty GPU while disk reads run near their observed peak
    "network",  # bursty GPU while network receive runs near its observed peak
    "launch",  # utilization high but SMs mostly idle (GPM): low SM parallelism (small grids, waiting kernels)
    "unknown",
]

WINDOW_S = 180.0  # classification window (3 min: long enough for dips, short enough to follow phases)
FINDINGS_S = 480.0  # raw look-back per call: anomalies need the busy stretch before a collapse
LIVE_S = 10.0  # a sample older than this is not "now"
SESSION_HOURS = 25
SESSION_GAP_MIN = 2  # inactive minutes bridged inside one session
SM_GOOD = 80.0  # SM activity treated as full use (DCGM guidance: >=80% is well used)

_LABELS = {
    "none": "unconstrained",
    "idle": "idle",
    "input": "input-bound",
    "cpu": "CPU-bound",
    "power": "power-capped",
    "thermal": "thermal-limited",
    "memory": "memory-bound",
    "vram_held": "VRAM held idle",
    "ram": "RAM pressure",
    "disk": "disk-bound",
    "network": "network-bound",
    "launch": "low SM parallelism",
    "unknown": "unclear",
}


@dataclass(frozen=True)
class Session:
    """A contiguous stretch of meaningful GPU activity (e.g. one training run)."""

    start: float  # epoch seconds
    end: Optional[float]  # None while still running
    minutes: int
    mean_util: float
    peak_vram_pct: float
    energy_wh: Optional[float] = None  # GPU energy over the session (counter, else minute power means)


@dataclass(frozen=True)
class Health:
    """One row of the system health panel."""

    name: str  # "GPU", "VRAM", "CPU", "CPU temp", "RAM", "Disk", "Network"
    level: InsightLevel
    value: str  # short reading, e.g. "87%", "412 MB/s read · 3 MB/s write"
    detail: str  # one-line judgement, e.g. "Healthy", "Swapping: training may stall"


@dataclass(frozen=True)
class Diagnosis:
    score: Optional[int] = None  # 0..100 efficiency score over the last hour; None = not enough data
    verdict: str = "Collecting data"  # short headline, e.g. "Input-bound: GPU waits on data"
    bottleneck: Bottleneck = "unknown"
    confidence: float = 0.0  # 0..1
    evidence: tuple[str, ...] = ()  # short bullet strings explaining the call
    actions: tuple[str, ...] = ()  # concrete suggestions, e.g. "Raise DataLoader num_workers"
    throttle: tuple[str, ...] = ()  # active throttle reasons now, e.g. ("power cap",)
    session: Optional[Session] = None  # current or most recent session
    vram_forecast: Optional[str] = None  # e.g. "VRAM full in ~6 min at current rate"
    findings: tuple[Insight, ...] = field(default_factory=tuple)  # extra insights to merge into the list
    health: tuple[Health, ...] = ()  # GPU, VRAM, CPU, CPU temp (if known), RAM, Disk, Network
    runner_up: Optional[Bottleneck] = None  # second-best explanation, if any
    vram_eta_min: Optional[float] = None  # numeric companion of vram_forecast
    score_basis: Optional[str] = None  # "sm_active" (GPM) or "utilization"


def diagnose(
    history: HistoryStore,
    gpu_uuid: Optional[str],
    now: Optional[float] = None,
    temp_warn: float = 80.0,
    details: Optional[GpuDetails] = None,
) -> Diagnosis:
    """Analyse recent history for one GPU (or the first GPU when ``gpu_uuid`` is None)."""
    try:
        return _diagnose(history, gpu_uuid, time.time() if now is None else now, temp_warn, details)
    except Exception:  # noqa: BLE001 - the dashboard must keep running; the bug is logged
        logger.warning("diagnose() failed", exc_info=True)
        return Diagnosis(verdict="Analysis unavailable")


def _pick(gpus: tuple[GpuSample, ...], uuid: Optional[str]) -> Optional[GpuSample]:
    if uuid is None:
        return gpus[0] if gpus else None
    for g in gpus:
        if g.uuid == uuid:
            return g
    return None


def _collect(history: HistoryStore, uuid: Optional[str], now: float, span: float) -> tuple[list[Sample], list[GRow]]:
    """Raw samples of the last ``span`` seconds (oldest first) and the (sample, gpu) pairs with a reading."""
    samples: list[Sample] = []
    rows: list[GRow] = []
    lo, hi = now - span, now + 1
    for s in reversed(history.recent):
        t = s.t
        if t < lo:
            break
        if t > hi:
            continue
        samples.append(s)
        gs = s.gpus
        g = gs[0] if len(gs) == 1 and (uuid is None or gs[0].uuid == uuid) else _pick(gs, uuid)
        if g is not None and g.util is not None:
            rows.append((s, g))
    samples.reverse()
    rows.reverse()
    return samples, rows


@dataclass(frozen=True)
class _Hourly:
    """Hour-wide results that change slowly; recomputed at most every HOURLY_TTL_S."""

    at: float
    cleared_at: int
    has_gpu: bool
    score: Optional[int]
    basis: Optional[str]
    session: Optional[Session]
    disk_peak: Optional[float]
    net_peak: Optional[float]


HOURLY_TTL_S = 10.0
# Weak keys: a discarded HistoryStore drops its cache, and a new one at the same address never hits it.
_HOURLY: WeakKeyDictionary[HistoryStore, dict[Optional[str], _Hourly]] = WeakKeyDictionary()


def _cached_hourly(history: HistoryStore, uuid: Optional[str], now: float) -> Optional[_Hourly]:
    cached = _HOURLY.get(history, {}).get(uuid)
    if cached is not None and 0 <= now - cached.at < HOURLY_TTL_S and cached.cleared_at == history.cleared_at:
        return cached
    return None


def _hourly(history: HistoryStore, uuid: Optional[str], now: float, samples: list[Sample], rows: list[GRow]) -> _Hourly:
    """Compute (and cache) the hour-wide results from the last hour of raw samples."""
    hour = history.stats("1 hour", 60, uuid, now)
    has_gpu = bool(rows) or hour.util is not None
    score, basis = _score(hour, rows)
    disk = [v for v in (s.disk_read_bps for s in samples) if v is not None]
    net = [v for v in (s.net_rx_bps for s in samples) if v is not None]
    result = _Hourly(
        at=now,
        cleared_at=history.cleared_at,
        has_gpu=has_gpu,
        score=score,
        basis=basis,
        session=_session(history, uuid, now, rows) if has_gpu else None,
        # 95th percentile, not max: one short burst must not disable the disk/network rules for an hour.
        disk_peak=p95(disk),
        net_peak=p95(net),
    )
    per = _HOURLY.get(history)
    if per is None:
        per = _HOURLY[history] = {}
    per[uuid] = result
    return result


def _diagnose(
    history: HistoryStore, uuid: Optional[str], now: float, temp_warn: float, details: Optional[GpuDetails]
) -> Diagnosis:
    hourly = _cached_hourly(history, uuid, now)
    if hourly is None:  # cold path (every HOURLY_TTL_S): one pass over the hour, then keep the recent part
        samples, rows = _collect(history, uuid, now, RECENT_S)
        hourly = _hourly(history, uuid, now, samples, rows)
        samples = samples[bisect.bisect_left([s.t for s in samples], now - FINDINGS_S) :]
        rows = rows[bisect.bisect_left([s.t for s, _ in rows], now - FINDINGS_S) :]
    else:
        samples, rows = _collect(history, uuid, now, FINDINGS_S)
    win_s = samples[bisect.bisect_left([s.t for s in samples], now - WINDOW_S) :]
    win_g = rows[bisect.bisect_left([s.t for s, _ in rows], now - WINDOW_S) :]
    gf = gpu_features(win_g)
    sf = system_features(win_s, hourly.disk_peak, hourly.net_peak)
    has_gpu = bool(rows) or hourly.has_gpu
    live = rows[-1][1] if rows and rows[-1][0].t >= now - LIVE_S else None
    input_conf = input_strength(gf) if gf else 0.0

    calls: list[Call] = gpu_calls(gf, temp_warn) if gf else []
    if gf is None and has_gpu:
        calls += minute_calls(history.stats("5 min", 5, uuid, now))
    sys_calls = system_calls(sf, gf, input_conf)
    if any(c.bottleneck == "cpu" for c in sys_calls):
        calls = [c for c in calls if c.bottleneck != "cpu"]  # raw-sample CPU rule beats the minute one
    calls = sorted(calls + sys_calls, key=lambda c: -c.confidence)
    best = calls[0] if calls else _fallback(gf, sf is not None, has_gpu)
    runner = next((c for c in calls[1:] if c.confidence >= 0.3 and c.bottleneck != best.bottleneck), None)

    evidence = list(best.evidence)
    if runner is not None:
        evidence.append(f"Runner-up: {_LABELS.get(runner.bottleneck, runner.bottleneck)} ({runner.confidence:.0%})")
    if hourly.score is not None:
        evidence.append(
            "Score basis: SM activity (GPM), last hour"
            if hourly.basis == "sm_active"
            else "Score basis: utilization.gpu (kernel-running time), last hour"
        )
    forecast, eta = vram_forecast(win_g, now)
    findings = (
        gpu_findings(rows, now, temp_warn)
        + gpm_findings(
            gf,
            best.bottleneck,
            input_conf,
            details.pcie_max_gen if details else None,
            details.pcie_max_width if details else None,
        )
        + system_findings(sf, input_conf, best.bottleneck)
    )
    if live is not None or not has_gpu:
        health_rows = gpu_health(live, gf, best.bottleneck, forecast, temp_warn)
    else:
        health_rows = [("GPU", "info", "n/a", "No live reading yet")]
    health_rows += system_health(sf, input_conf)
    return Diagnosis(
        score=hourly.score,
        verdict=best.verdict,
        bottleneck=cast(Bottleneck, best.bottleneck),
        confidence=round(best.confidence, 2),
        evidence=tuple(evidence),
        actions=best.actions,
        throttle=live.throttle if live else (),
        session=hourly.session,
        vram_forecast=forecast,
        findings=tuple(findings),
        health=tuple(Health(n, cast(InsightLevel, lv), v, d) for n, lv, v, d in health_rows),
        runner_up=cast(Bottleneck, runner.bottleneck) if runner else None,
        vram_eta_min=round(eta, 1) if eta is not None else None,
        score_basis=hourly.basis,
    )


def _fallback(gf: object, has_system: bool, has_gpu: bool) -> Call:
    if gf is not None:
        return Call("unknown", 0.25, "Mixed GPU load", ("No single pattern dominates the last 3 min",))
    if not has_gpu and has_system:
        return Call("unknown", 0.5, "No GPU; system view only", ("No supported GPU detected",))
    return Call("unknown", 0.0, "Collecting data", ())


def _score(hour: WindowStats, rows: Sequence[GRow]) -> tuple[Optional[int], Optional[str]]:
    """Efficiency score 0-100 over the last hour, and its basis.

    activity = mean SM activity / 80% capped at 1 when GPM data covers >= 80% of the hour's samples
               ("sm_active"), else mean utilization.gpu / 100 ("utilization")
    base     = 100 × (0.6 × activity + 0.4 × busy_share)        busy_share = time at >= 70% util
    penalty  = 30 × held_idle_share                             VRAM held while idle
             + 10 × power_throttle_share + 20 × thermal_share   of active raw samples
             + 15 × min(1, burstiness / 0.3)                    short-dip share of running time
    score    = clamp(round(base - penalty), 0, 100); None with < 5 recorded minutes.
    Throttle and burstiness come from raw samples, so they count as 0 until those exist.
    """
    if hour.covered_minutes < 5 or hour.util is None:
        return None, None
    power, thermal, burst = hour_extras(rows)
    # Use SM activity only when it describes (nearly) the whole hour, not a few GPM-era minutes
    # inside an hour otherwise recorded without it (e.g. minutes written by the VS Code extension).
    if hour.sm is not None and (hour.sm_coverage or 0.0) >= 0.8:
        activity, basis = min(1.0, hour.sm / SM_GOOD), "sm_active"
    else:
        activity, basis = hour.util / 100, "utilization"
    base = 100 * (0.6 * activity + 0.4 * (hour.busy_share or 0.0))
    penalty = 30 * (hour.held_idle_share or 0.0) + 10 * power + 20 * thermal + 15 * min(1.0, burst / 0.3)
    return max(0, min(100, round(base - penalty))), basis


def _bucket_gpu(b: dict[str, Any], uuid: Optional[str]) -> Optional[dict[str, Any]]:
    gpus = b["gpus"]
    return gpus.get(uuid) if uuid else next(iter(gpus.values()), None)


def _session(history: HistoryStore, uuid: Optional[str], now: float, rows: Sequence[GRow]) -> Optional[Session]:
    """Latest stretch of minutes with mean util >= 15%, bridging gaps of up to 2 minutes."""
    start_ms, cells = history.heatmap(uuid, SESSION_HOURS, now)
    last = len(cells) - 1
    end_i = last
    while end_i >= 0 and (cells[end_i] is None or (cells[end_i] or 0) < ACTIVE_UTIL):
        end_i -= 1
    if end_i < 0:
        return None
    first, gap = end_i, 0
    for j in range(end_i - 1, -1, -1):
        c = cells[j]
        if c is not None and c >= ACTIVE_UTIL:
            first, gap = j, 0
        else:
            gap += 1
            if gap > SESSION_GAP_MIN:
                break
    vals = [c for c in cells[first : end_i + 1] if c is not None]
    start = (start_ms + first * MINUTE_MS) / 1000
    running = last - end_i <= SESSION_GAP_MIN
    # Raw samples are finer than minutes: a job quiet for the last 30-60 s has ended.
    times = [s.t for s, _ in rows]
    tail = [(s.t, g.util or 0.0) for s, g in rows[bisect.bisect_left(times, now - 60) :]]
    if running and len(tail) >= 5 and tail[0][0] <= now - 30 and all(u < ACTIVE_UTIL for _, u in tail):
        running = False
    end = None if running else min(now, (start_ms + (end_i + 1) * MINUTE_MS) / 1000)
    span_end = now if end is None else end
    in_span = rows[bisect.bisect_left(times, start) : bisect.bisect_right(times, span_end)]
    total = in_span[-1][1].mem_total_mib if in_span else None
    used = max((g.mem_used_mib or 0.0 for _, g in in_span), default=0.0)
    peak = used / total * 100 if total else 0.0
    minute_power = 0.0
    flushed_to = start * 1000
    for b in reversed(history.minutes):
        if b["t"] < start_ms + first * MINUTE_MS:
            break
        g = _bucket_gpu(b, uuid)
        if g and b["t"] < span_end * 1000:
            peak = max(peak, g["vram"])
            minute_power += g["power"] / 60  # W × 1 min -> Wh
            flushed_to = max(flushed_to, b["t"] + MINUTE_MS)
    energy = _energy(in_span, start, flushed_to / 1000, minute_power)
    minutes = (end_i if end is not None else last) - first + 1
    return Session(
        start=start,
        end=end,
        minutes=max(1, minutes),
        mean_util=round(sum(vals) / len(vals), 1) if vals else 0.0,
        peak_vram_pct=round(peak, 1),
        energy_wh=energy,
    )


def _energy(rows: Sequence[GRow], start: float, flushed_to: float, minute_wh: float) -> Optional[float]:
    """Session energy: NVML's energy counter when raw samples cover the start, else minute power means
    plus raw power for the not-yet-flushed minute."""
    first = next(((s.t, g.energy_j) for s, g in rows if g.energy_j is not None), None)
    last = next(((s.t, g.energy_j) for s, g in reversed(rows) if g.energy_j is not None), None)
    if first and last and first[0] < last[0] and first[0] <= start + 90 and last[1] >= first[1]:
        return round((last[1] - first[1]) / 3600, 2)
    times = [s.t for s, _ in rows]
    tail = [(s.t, g.power_w) for s, g in rows[bisect.bisect_left(times, flushed_to) :] if g.power_w is not None]
    extra = 0.0
    for (t0, p0), (t1, _) in zip(tail, tail[1:]):
        extra += p0 * (t1 - t0) / 3600
    total = minute_wh + extra
    return round(total, 2) if total > 0 else None
