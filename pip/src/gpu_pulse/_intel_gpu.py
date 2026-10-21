"""GPU-side analysis for :mod:`gpu_pulse.intel`: window features, bottleneck rules, anomalies.

Private module. Everything here works on raw samples (``Sample``/``GpuSample`` pairs, oldest
first) and returns plain values so ``intel.py`` owns the public contract.

A caveat that shapes every rule: NVML ``utilization.gpu`` is the share of time *any* kernel was
running, not how many SMs were busy. Dips mean the GPU had nothing queued; a steady 100% does
not prove the GPU is used efficiently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from .history import HELD_VRAM, IDLE_UTIL
from .models import GpuSample, Sample, WindowStats

ACTIVE_UTIL = 15.0  # a job is running
DIP_UTIL = 30.0  # below this inside a running job = the GPU ran out of work
HIGH_UTIL = 60.0  # a busy plateau
BUSY_UTIL = 50.0  # samples used for clock / memory-controller comparisons
QUIET_S = 45.0  # trailing idle time that overrides the window's older pattern
DIP_MAX_S = 15.0  # longer low stretches are pauses (eval, checkpoint), not data-loader dips
MIN_RAW = 20  # raw samples needed before trusting the raw window
POWER_REASONS = frozenset({"power cap", "hw power brake"})
THERMAL_REASONS = frozenset({"thermal", "hw thermal"})
HW_SLOWDOWN = "hw slowdown"  # NVML: temperature *or* external power brake; split by temperature
UTIL_NOTE = "Utilization = share of time a kernel ran, not how many cores were busy"

GRow = tuple[Sample, GpuSample]


@dataclass(frozen=True)
class Call:
    """One candidate explanation for the recent window."""

    bottleneck: str
    confidence: float
    verdict: str  # <= 40 chars
    evidence: tuple[str, ...]
    actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class GpuFeat:
    n: int
    span_s: float
    util_mean: float
    idle_share: float  # util < 10
    held_share: float  # util < 10 with VRAM >= 50%
    run_share: float  # samples not inside long pauses
    run_mean: float  # mean util while running (short dips included)
    run_cv: float  # coefficient of variation while running
    high_share: float  # running samples >= HIGH_UTIL
    dip_share: float  # short-dip samples / running samples
    episodes: int  # completed short dips (busy -> <30% -> busy within DIP_MAX_S)
    dips_per_min: float
    vram_mean: Optional[float]
    vram_now: Optional[float]
    busy_util: Optional[float]
    busy_mem_util: Optional[float]
    power_share: float  # active samples with a power-type throttle reason
    thermal_share: float
    limit_share: float  # active samples with any power/thermal/hw reason
    near_cap_share: Optional[float]  # busy samples at >=95% power cap with clocks <95% of max
    clock_ratio: Optional[float]  # busy samples: clock / max clock
    clock_drop: Optional[float]  # first-third minus last-third busy clock ratio
    power_mean: Optional[float]
    power_cap: Optional[float]
    temp_now: Optional[float]
    temp_max: Optional[float]
    # GPM "true usage" means over active samples (None without GPM) and PCIe state.
    sm_mean: Optional[float] = None
    occupancy_mean: Optional[float] = None
    tensor_mean: Optional[float] = None
    dram_mean: Optional[float] = None
    fp32_mean: Optional[float] = None
    pcie_rx_mean: Optional[float] = None
    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None
    quiet_s: float = 0.0  # seconds since the last sample at >= 10% utilization


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def pct(x: float) -> str:
    return f"{round(x * 100)}%"


def fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares (slope, r²); (0, 0) when undefined."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sxy = syy = 0.0
    for x, y in zip(xs, ys):
        dx, dy = x - mx, y - my
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy
    if sxx <= 0:
        return 0.0, 0.0
    return sxy / sxx, (sxy * sxy / (sxx * syy) if syy > 0 else 0.0)


def dip_runs(ts: Sequence[float], us: Sequence[float]) -> tuple[int, int, list[bool]]:
    """Split low (<30%) runs into short dips and pauses.

    A dip is bracketed by busy samples (>=60% within 3 samples on both sides) and lasts at most
    DIP_MAX_S. Everything else that is low (window edges, long idle) is a pause.
    Returns (dip episodes, dip samples, pause mask).
    """
    n = len(us)
    pause = [False] * n
    episodes = dip_n = 0
    i = 0
    while i < n:
        if us[i] >= DIP_UTIL:
            i += 1
            continue
        j = i
        while j < n and us[j] < DIP_UTIL:
            j += 1
        bracketed = i > 0 and j < n and max(us[max(0, i - 3) : i]) >= HIGH_UTIL and max(us[j : j + 3]) >= HIGH_UTIL
        if bracketed and ts[j] - ts[i - 1] <= DIP_MAX_S + 2:
            episodes += 1
            dip_n += j - i
        else:
            for k in range(i, j):
                pause[k] = True
        i = j
    return episodes, dip_n, pause


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def gpu_features(rows: Sequence[GRow]) -> Optional[GpuFeat]:
    """Features of the recent window; None when there are too few samples."""
    n = len(rows)
    if n < MIN_RAW:
        return None
    ts = [s.t for s, _ in rows]
    us = [g.util or 0.0 for _, g in rows]
    episodes, dip_n, pause = dip_runs(ts, us)
    run = [u for u, p in zip(us, pause) if not p]
    run_mean = _mean(run) or 0.0
    run_cv = (
        math.sqrt(sum((u - run_mean) ** 2 for u in run) / len(run)) / run_mean if len(run) >= 10 and run_mean else 0.0
    )
    span = max(ts[-1] - ts[0], 1e-9)
    dt = span / (n - 1)
    run_min = len(run) * dt / 60

    vrams = [v for v in (g.vram_pct for _, g in rows) if v is not None]
    idle = held = active = power = thermal = limit = 0
    busy_u: list[float] = []
    busy_m: list[float] = []
    ratios: list[float] = []
    near: list[bool] = []
    powers: list[float] = []
    cap: Optional[float] = None
    temps = [g.temp_c for _, g in rows if g.temp_c is not None]
    temp_max = max(temps) if temps else None
    for _, g in rows:
        u = g.util or 0.0
        v = g.vram_pct
        if u < IDLE_UTIL:
            idle += 1
            held += v is not None and v >= HELD_VRAM
        if g.power_w is not None:
            powers.append(g.power_w)
        if g.power_limit_w:
            cap = g.power_limit_w
        if u >= ACTIVE_UTIL:
            active += 1
            r = g.throttle
            if r:
                hot = temp_max is not None and g.temp_c is not None and g.temp_c >= 75
                p = any(x in POWER_REASONS for x in r) or (HW_SLOWDOWN in r and not hot)
                th = any(x in THERMAL_REASONS for x in r) or (HW_SLOWDOWN in r and hot)
                power += p
                thermal += th
                limit += p or th
        if u >= BUSY_UTIL:
            busy_u.append(u)
            if g.mem_util is not None:
                busy_m.append(g.mem_util)
            if g.clock_mhz is not None and g.clock_max_mhz:
                ratio = g.clock_mhz / g.clock_max_mhz
                ratios.append(ratio)
                if g.power_w is not None and g.power_limit_w:
                    near.append(g.power_w >= 0.95 * g.power_limit_w and ratio < 0.95)
    third = len(ratios) // 3
    drop = (sum(ratios[:third]) - sum(ratios[-third:])) / third if third >= 5 else None
    act = active if active >= 10 else 0
    on = [g for _, g in rows if (g.util or 0) >= ACTIVE_UTIL]
    last = rows[-1][1]

    def gpm(attr: str) -> Optional[float]:
        vals = [v for v in (getattr(g, attr) for g in on) if v is not None]
        return sum(vals) / len(vals) if len(vals) >= 10 else None

    rx = [g.pcie_rx_bps for g in on if g.pcie_rx_bps is not None]
    return GpuFeat(
        n=n,
        span_s=span,
        util_mean=sum(us) / n,
        idle_share=idle / n,
        held_share=held / n,
        run_share=len(run) / n,
        run_mean=run_mean,
        run_cv=run_cv,
        high_share=sum(u >= HIGH_UTIL for u in run) / len(run) if run else 0.0,
        dip_share=dip_n / len(run) if run else 0.0,
        episodes=episodes,
        dips_per_min=episodes / run_min if run_min > 0 else 0.0,
        vram_mean=_mean(vrams),
        vram_now=rows[-1][1].vram_pct,
        busy_util=_mean(busy_u),
        busy_mem_util=_mean(busy_m),
        power_share=power / act if act else 0.0,
        thermal_share=thermal / act if act else 0.0,
        limit_share=limit / act if act else 0.0,
        near_cap_share=sum(near) / len(near) if len(near) >= 10 else None,
        clock_ratio=_mean(ratios),
        clock_drop=drop,
        power_mean=_mean(powers),
        power_cap=cap,
        temp_now=last.temp_c,
        temp_max=temp_max,
        sm_mean=gpm("sm_active"),
        occupancy_mean=gpm("sm_occupancy"),
        tensor_mean=gpm("tensor_active"),
        dram_mean=gpm("dram_bw_util"),
        fp32_mean=gpm("fp32_active"),
        pcie_rx_mean=sum(rx) / len(rx) if len(rx) >= 10 else None,
        pcie_gen=(on[-1] if on else last).pcie_gen,
        pcie_width=(on[-1] if on else last).pcie_width,
        quiet_s=ts[-1] - next((t for t, u in zip(reversed(ts), reversed(us)) if u >= IDLE_UTIL), ts[0]),
    )


def input_strength(f: GpuFeat) -> float:
    """Confidence (0.4..0.85) that a running job is busy-but-bursty; 0 when the pattern is absent."""
    run_min = f.run_share * f.span_s / 60
    if f.run_share < 0.5 or f.run_mean < 20 or f.high_share < 0.2 or run_min < 1:
        return 0.0
    if f.episodes < 3 or f.dips_per_min < 1.5 or (f.run_cv < 0.3 and f.dip_share < 0.1):
        return 0.0
    s = (clamp01((f.run_cv - 0.25) / 0.35) + clamp01(f.dips_per_min / 6) + clamp01(f.dip_share / 0.3)) / 3
    return min(0.85, 0.4 + 0.45 * s)


def input_evidence(f: GpuFeat) -> tuple[str, ...]:
    return (
        f"{f.dips_per_min:.1f} short dips/min below 30% between busy plateaus",
        f"Busy-time mean {f.run_mean:.0f}%, variation (CV) {f.run_cv:.2f}",
        f"GPU had no work queued for ~{pct(f.dip_share)} of running time",
    )


INPUT_ACTIONS = (
    "Raise DataLoader num_workers (try CPU cores / 2)",
    "Set pin_memory=True and persistent_workers=True",
    "Prefetch or cache decoded samples (prefetch_factor, faster storage)",
    "Confirm with torch.profiler: look for gaps between steps",
)
HELD_ACTIONS = (
    "Free it: restart the idle kernel, or del tensors + torch.cuda.empty_cache()",
    "Check the process list for the holder (Windows may hide per-process VRAM)",
)
CPU_ACTIONS = (
    "Move preprocessing to DataLoader workers or the GPU",
    "Check for a single busy core (single-threaded loader / Python GIL)",
)


def gpu_calls(f: GpuFeat, temp_warn: float) -> list[Call]:
    """Every GPU-side rule that fires on the window."""
    out: list[Call] = []
    win = f"{f.span_s / 60:.0f} min"
    if f.quiet_s >= QUIET_S and f.idle_share < 0.8:  # just went quiet: say so before older patterns
        held = (f.vram_now or 0) >= HELD_VRAM
        out.append(
            Call(
                "vram_held" if held and f.quiet_s >= 120 else "idle",
                0.92,
                "VRAM held while GPU idles" if held and f.quiet_s >= 120 else "GPU idle now",
                (
                    f"Utilization under 10% for the last {f.quiet_s:.0f} s",
                    f"VRAM still {f.vram_now:.0f}% allocated" if held else "VRAM released",
                ),
            )
        )
    elif f.held_share >= 0.5:
        out.append(
            Call(
                "vram_held",
                0.55 + 0.4 * f.held_share,
                "VRAM held while GPU idles",
                (
                    f"{f.vram_mean or 0:.0f}% VRAM allocated, utilization <10% for {pct(f.held_share)} of {win}",
                    "Typical of an open notebook or a finished script that kept its tensors",
                ),
                HELD_ACTIONS,
            )
        )
    elif f.idle_share >= 0.8:
        out.append(
            Call(
                "idle",
                0.6 + 0.35 * f.idle_share,
                "GPU idle",
                (f"Utilization under 10% for {pct(f.idle_share)} of the last {win}",),
            )
        )
    strength = input_strength(f)
    if strength:
        out.append(
            Call(
                "input",
                strength,
                "Input-bound: GPU waits on data",
                (*input_evidence(f), UTIL_NOTE),
                INPUT_ACTIONS,
            )
        )
    if f.power_share >= 0.3:
        out.append(
            _power_call(f, 0.45 + 0.45 * f.power_share, f"'power cap' reason in {pct(f.power_share)} of busy samples")
        )
    elif f.near_cap_share is not None and f.near_cap_share >= 0.4:
        out.append(
            _power_call(
                f, 0.35 + 0.3 * f.near_cap_share, f"At >=95% of power cap in {pct(f.near_cap_share)} of busy samples"
            )
        )
    hot = f.temp_max is not None and f.temp_max >= temp_warn - 3
    if f.thermal_share >= 0.15:
        out.append(
            _thermal_call(
                f, 0.5 + 0.45 * f.thermal_share, f"Thermal throttle reason in {pct(f.thermal_share)} of busy samples"
            )
        )
    elif hot and f.clock_drop is not None and f.clock_drop >= 0.05 and (f.clock_ratio or 1) < 0.9:
        out.append(
            _thermal_call(f, 0.45, f"Peak {f.temp_max:.0f} °C while clocks fell {f.clock_drop * 100:.0f} points of max")
        )
    if f.dram_mean is not None:  # measured DRAM bandwidth replaces the memory-controller heuristic
        if f.dram_mean >= 80:
            out.append(
                Call(
                    "memory",
                    0.55 + 0.35 * clamp01((f.dram_mean - 80) / 20),
                    "Memory-bound: DRAM bandwidth maxed",
                    (
                        f"DRAM bandwidth at {f.dram_mean:.0f}% of peak while active (GPM measurement)",
                        f"SMs {f.sm_mean or 0:.0f}% active; compute is waiting on memory",
                    ),
                    (
                        "Use mixed precision / smaller dtypes to move fewer bytes",
                        "Fuse elementwise ops (torch.compile) to cut memory traffic",
                        "Larger batches can raise arithmetic intensity for GEMM-heavy models",
                    ),
                )
            )
    elif (
        f.util_mean >= 50
        and f.busy_util
        and f.busy_mem_util is not None
        and f.busy_mem_util >= 60
        and f.busy_mem_util >= 0.75 * f.busy_util
    ):
        out.append(
            Call(
                "memory",
                0.3 + 0.2 * clamp01((f.busy_mem_util - 60) / 30),
                "Possibly memory-bandwidth bound",
                (
                    f"Memory controller busy {f.busy_mem_util:.0f}% vs kernel-active {f.busy_util:.0f}%",
                    "A hint from time-based counters, not a bandwidth measurement",
                ),
                (
                    "Use mixed precision / smaller dtypes to move fewer bytes",
                    "Fuse elementwise ops (torch.compile) to cut memory traffic",
                    "Confirm with Nsight Compute (dram throughput vs SM throughput)",
                ),
            )
        )
    launch = launch_bound(f)
    if launch:
        out.append(
            Call(
                "launch",
                launch,
                "Low SM parallelism",
                (
                    f"Utilization {f.run_mean:.0f}% but SMs only {f.sm_mean or 0:.0f}% active (GPM)",
                    "Kernels are running, but few SMs have work: small grids, a waiting/spinning kernel "
                    "(e.g. NCCL), or latency-bound code",
                    UTIL_NOTE,
                ),
                (
                    "Increase the batch size so each kernel has more parallel work",
                    "Profile with Nsight Systems / torch.profiler to find the low-occupancy kernels",
                    "If multi-GPU, check whether communication kernels are waiting on a peer",
                ),
            )
        )
    if (
        f.run_mean >= 80
        and f.run_share >= 0.75
        and f.run_cv < 0.25
        and f.limit_share < 0.3
        and not strength
        and not launch
        and (f.near_cap_share or 0.0) < 0.4  # pinned at the power cap is a limit even without a reason bit
        and (f.dram_mean is None or f.dram_mean < 80)  # saturated DRAM bandwidth is a limit (memory-bound)
    ):
        ev = [f"Mean {f.run_mean:.0f}% while running, {pct(f.high_share)} of samples >=60%"]
        if f.sm_mean is not None:
            ev.append(f"SMs {f.sm_mean:.0f}% active, Tensor Cores {f.tensor_mean or 0:.0f}% (GPM)")
        else:
            ev.append(UTIL_NOTE)
        ev.append("No power or thermal throttle reasons reported")
        out.append(
            Call(
                "none",
                0.55 + 0.35 * clamp01((f.run_mean - 80) / 15),
                "GPU kept busy, no limits seen",
                tuple(ev),
                ("Larger batches or mixed precision may still raise throughput",),
            )
        )
    return out


def launch_bound(f: GpuFeat) -> float:
    """Confidence that utilization is high while SMs are mostly idle (needs GPM); 0 otherwise.

    Kept modest (≤0.5): several causes look alike from counters alone. Busy DRAM points to a
    memory-bound workload instead, so the rule stays out of that case.
    """
    sm = f.sm_mean
    if sm is None or f.run_mean < 70 or sm >= 30 or sm > 0.4 * f.run_mean:
        return 0.0
    if f.dram_mean is not None and f.dram_mean >= 60:
        return 0.0
    return 0.35 + 0.15 * clamp01((30 - sm) / 25)


def _power_call(f: GpuFeat, conf: float, first: str) -> Call:
    ev = [first]
    if f.power_mean is not None and f.power_cap:
        ev.append(f"Drawing {f.power_mean:.0f} W of {f.power_cap:.0f} W cap")
    if f.clock_ratio is not None:
        ev.append(f"Clocks at {pct(f.clock_ratio)} of max while busy")
    return Call(
        "power",
        conf,
        "Power-capped: clocks held down",
        tuple(ev),
        (
            "Expected at full load: the card is at its power budget, not stalled",
            "If your card allows it, raise the power limit (nvidia-smi -pl) or improve cooling",
            "Mixed precision (AMP / bf16) usually does more work per watt",
        ),
    )


def _thermal_call(f: GpuFeat, conf: float, first: str) -> Call:
    ev = [first]
    if f.temp_now is not None:
        ev.append(f"Now {f.temp_now:.0f} °C, peak {f.temp_max or f.temp_now:.0f} °C")
    if f.clock_ratio is not None:
        ev.append(f"Clocks at {pct(f.clock_ratio)} of max while busy")
    return Call(
        "thermal",
        conf,
        "Thermal-limited: clocks dropping",
        tuple(ev),
        (
            "Improve airflow: clean dust filters, raise the fan curve",
            "Lower the power limit a little; it often costs little speed and cuts heat",
            "Check for a blocked intake or a hot room",
        ),
    )


def minute_calls(st: WindowStats) -> list[Call]:
    """Coarse rules from per-minute averages when raw samples are missing (e.g. just started)."""
    if st.util is None:
        return []
    note = "Based on per-minute averages; raw samples not yet available"
    out: list[Call] = []
    held = st.held_idle_share or 0.0
    if held >= 0.5:
        out.append(
            Call(
                "vram_held",
                0.5 + 0.3 * held,
                "VRAM held while GPU idles",
                (f"Held idle {pct(held)}", note),
                HELD_ACTIONS,
            )
        )
    elif (st.idle_share or 0) >= 0.8:
        out.append(Call("idle", 0.7, "GPU idle", (f"Mean utilization {st.util:.0f}%", note)))
    elif st.util >= 80 and (st.busy_share or 0) >= 0.75:
        out.append(Call("none", 0.5, "GPU kept busy, no limits seen", (f"Mean utilization {st.util:.0f}%", note)))
    elif 20 <= st.util < 70 and (st.util_peak or 0) >= 85:
        out.append(
            Call(
                "input",
                0.3,
                "Input-bound: GPU waits on data",
                (f"Mean {st.util:.0f}% but peaks of {st.util_peak:.0f}%", note),
                INPUT_ACTIONS,
            )
        )
    if st.cpu is not None and st.cpu >= 85 and 10 <= st.util < 50:
        out.append(
            Call(
                "cpu",
                0.45,
                "CPU-bound: GPU waits on CPU",
                (f"CPU {st.cpu:.0f}% vs GPU {st.util:.0f}%", note),
                CPU_ACTIONS,
            )
        )
    return out


def hour_extras(rows: Sequence[GRow]) -> tuple[float, float, float]:
    """(power-throttle share, thermal-throttle share, burstiness) over the raw hour.

    Shares are over active samples (util >= 15%). Burstiness = short-dip samples / running samples.
    """
    if len(rows) < MIN_RAW:
        return 0.0, 0.0, 0.0
    active = power = thermal = 0
    us: list[float] = []
    for _, g in rows:
        u = g.util or 0.0
        us.append(u)
        if u >= ACTIVE_UTIL:
            active += 1
            r = g.throttle
            if r:
                hot = g.temp_c is not None and g.temp_c >= 75
                power += any(x in POWER_REASONS for x in r) or (HW_SLOWDOWN in r and not hot)
                thermal += any(x in THERMAL_REASONS for x in r) or (HW_SLOWDOWN in r and hot)
    _, dip_n, pause = dip_runs([s.t for s, _ in rows], us)
    running = len(pause) - sum(pause)
    return (
        power / active if active else 0.0,
        thermal / active if active else 0.0,
        dip_n / running if running >= MIN_RAW else 0.0,
    )
