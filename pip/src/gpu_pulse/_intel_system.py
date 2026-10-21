"""Whole-system analysis for :mod:`gpu_pulse.intel`: CPU, RAM/swap, disk and network.

Private module. Works on raw ``Sample`` objects (oldest first) and on the GPU features from
``_intel_gpu`` when a GPU is present, so it also runs on machines without a supported GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from ._intel_gpu import Call, GpuFeat, clamp01, fit
from .models import Insight, Sample

DISK_MIN_BPS = 20e6  # sustained reads below this are unlikely to pace a GPU
NET_MIN_BPS = 10e6
CORE_PEGGED = 95.0
RAM_HIGH = 90.0
SWAP_RISE = 0.5  # %/min


@dataclass(frozen=True)
class SysFeat:
    n: int
    span_s: float
    cpu_mean: float
    cpu_now: float
    core_max_mean: Optional[float]  # mean over samples of the busiest core (the core may change)
    cores: int
    mem_mean: float
    mem_now: float
    swap_now: Optional[float]
    swap_slope: Optional[float]  # % of swap per minute
    cpu_temp: Optional[float]
    cpu_freq: Optional[float]
    disk_r_mean: Optional[float]
    disk_w_mean: Optional[float]
    disk_r_now: Optional[float]
    disk_w_now: Optional[float]
    disk_r_peak: Optional[float]  # highest reading in the last hour
    net_rx_mean: Optional[float]
    net_tx_mean: Optional[float]
    net_rx_now: Optional[float]
    net_tx_now: Optional[float]
    net_rx_peak: Optional[float]


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _vals(rows: Sequence[Sample], attr: str) -> list[float]:
    return [v for v in (getattr(s, attr) for s in rows) if v is not None]


def rate(bps: Optional[float]) -> str:
    """Human byte rate: '412 MB/s', '1.2 GB/s', '35 KB/s'."""
    if bps is None:
        return "n/a"
    if bps >= 1e9:
        return f"{bps / 1e9:.1f} GB/s"
    if bps >= 1e6:
        return f"{bps / 1e6:.0f} MB/s"
    return f"{bps / 1e3:.0f} KB/s"


def p95(vals: Sequence[float]) -> Optional[float]:
    """95th percentile (nearest rank); robust "peak" that ignores a single burst."""
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(0.95 * (len(s) - 1) + 0.5))]


def system_features(
    win: Sequence[Sample], disk_peak: Optional[float] = None, net_peak: Optional[float] = None
) -> Optional[SysFeat]:
    """Window features; ``disk_peak`` / ``net_peak`` are the hour's highest readings (cached upstream)."""
    if not win:
        return None
    n = len(win)
    last = win[-1]
    cores = [max(s.cpu_per_core) for s in win if s.cpu_per_core]
    swaps = [(s.t, s.swap_pct) for s in win if s.swap_pct is not None]
    slope: Optional[float] = None
    if len(swaps) >= 10 and swaps[-1][0] - swaps[0][0] >= 30:
        slope = fit([t - swaps[0][0] for t, _ in swaps], [v for _, v in swaps])[0] * 60
    dr = _vals(win, "disk_read_bps") + ([disk_peak] if disk_peak is not None else [])
    nr = _vals(win, "net_rx_bps") + ([net_peak] if net_peak is not None else [])
    return SysFeat(
        n=n,
        span_s=last.t - win[0].t,
        cpu_mean=sum(s.cpu_util for s in win) / n,
        cpu_now=last.cpu_util,
        core_max_mean=_mean(cores),
        cores=len(last.cpu_per_core),
        mem_mean=sum(s.mem_pct for s in win) / n,
        mem_now=last.mem_pct,
        swap_now=last.swap_pct,
        swap_slope=slope,
        cpu_temp=last.cpu_temp_c,
        cpu_freq=last.cpu_freq_mhz,
        disk_r_mean=_mean(_vals(win, "disk_read_bps")),
        disk_w_mean=_mean(_vals(win, "disk_write_bps")),
        disk_r_now=last.disk_read_bps,
        disk_w_now=last.disk_write_bps,
        disk_r_peak=p95(dr),
        net_rx_mean=_mean(_vals(win, "net_rx_bps")),
        net_tx_mean=_mean(_vals(win, "net_tx_bps")),
        net_rx_now=last.net_rx_bps,
        net_tx_now=last.net_tx_bps,
        net_rx_peak=p95(nr),
    )


def swapping(f: SysFeat) -> bool:
    return f.swap_slope is not None and f.swap_slope >= SWAP_RISE and (f.swap_now or 0) >= 2


def disk_heavy(f: SysFeat) -> bool:
    m, p = f.disk_r_mean, f.disk_r_peak
    return m is not None and p is not None and m >= DISK_MIN_BPS and m >= 0.5 * p


def net_heavy(f: SysFeat) -> bool:
    m, p = f.net_rx_mean, f.net_rx_peak
    return m is not None and p is not None and m >= NET_MIN_BPS and m >= 0.5 * p


def pegged(f: SysFeat) -> bool:
    """One core near 100% while the average is low: a single-threaded step (GIL, num_workers=0)."""
    return f.core_max_mean is not None and f.core_max_mean >= CORE_PEGGED and f.cpu_mean < 60 and f.cores >= 4


def system_calls(f: Optional[SysFeat], g: Optional[GpuFeat], input_conf: float) -> list[Call]:
    """Bottleneck candidates that need system data. ``input_conf`` > 0 means the GPU is bursty."""
    out: list[Call] = []
    if f is None:
        return out
    bump = 0.05 if input_conf else 0.0
    if f.mem_mean >= RAM_HIGH or swapping(f):
        ev = [f"RAM {f.mem_now:.0f}% used"]
        if f.swap_now is not None:
            ev.append(f"Swap {f.swap_now:.0f}% used" + (f", rising {f.swap_slope:.1f}%/min" if swapping(f) else ""))
        if input_conf:
            ev.append("GPU utilization is bursty, consistent with stalls on paging")
        conf = max(0.5 + 0.4 * clamp01((f.mem_mean - RAM_HIGH) / 8), 0.65 if swapping(f) else 0.0)
        out.append(
            Call(
                "ram",
                min(0.9, conf + bump),
                "RAM pressure: system swapping" if swapping(f) else "RAM nearly full",
                tuple(ev),
                (
                    "Lower DataLoader num_workers or prefetch_factor (each worker holds its own copies)",
                    "Use memory-mapped or streaming datasets instead of loading everything",
                    "Close other memory-heavy applications",
                ),
            )
        )
    if input_conf and disk_heavy(f):
        out.append(
            Call(
                "disk",
                input_conf + 0.05,
                "Disk-bound: GPU waits on reads",
                (
                    f"Disk reads average {rate(f.disk_r_mean)} (hour peak {rate(f.disk_r_peak)})",
                    "GPU utilization dips between busy plateaus at the same time",
                ),
                (
                    "Cache the dataset in RAM or move it to a faster (NVMe) drive",
                    "Pack many small files into shards (WebDataset, LMDB, memory-mapped arrays)",
                    "Raise DataLoader num_workers so reads overlap with compute",
                ),
            )
        )
    if input_conf and net_heavy(f):
        out.append(
            Call(
                "network",
                input_conf + 0.04,
                "Network-bound: GPU waits on data",
                (
                    f"Network receive averages {rate(f.net_rx_mean)} (hour peak {rate(f.net_rx_peak)})",
                    "GPU utilization dips between busy plateaus at the same time",
                ),
                (
                    "Stage the dataset on local disk before training",
                    "For multi-node runs, check interconnect bandwidth and overlap all-reduce with compute",
                    "Stream compressed shards and decode on worker processes",
                ),
            )
        )
    if g is not None and g.run_share >= 0.2:
        if pegged(f) and g.util_mean < 60:
            out.append(
                Call(
                    "cpu",
                    input_conf + 0.03 if input_conf else 0.45,
                    "CPU-bound: one core pegged",
                    (
                        f"Busiest core averages {f.core_max_mean:.0f}% while CPU overall is {f.cpu_mean:.0f}%",
                        f"GPU mean {g.util_mean:.0f}%: work arrives from a single thread",
                    ),
                    (
                        "Use DataLoader num_workers > 0 so loading leaves the main process",
                        "Vectorize per-sample Python work or move it to the GPU",
                        "Profile the main process (py-spy top) to find the hot loop",
                    ),
                )
            )
        elif f.cpu_mean >= 85 and g.util_mean < 50:
            out.append(
                Call(
                    "cpu",
                    0.5 + 0.3 * clamp01((f.cpu_mean - 85) / 15) + 0.1 * clamp01((50 - g.util_mean) / 50),
                    "CPU-bound: GPU waits on CPU",
                    (f"CPU {f.cpu_mean:.0f}% vs GPU {g.util_mean:.0f}% over the window",),
                    (
                        "Move preprocessing to the GPU (torchvision.transforms.v2, DALI) or precompute it",
                        "Check for Python loops between kernel launches",
                        "Close other CPU-heavy processes",
                    ),
                )
            )
    elif g is None and f.cpu_mean >= 85:
        out.append(
            Call(
                "cpu",
                0.55 + 0.3 * clamp01((f.cpu_mean - 85) / 15),
                "CPU saturated",
                (f"CPU {f.cpu_mean:.0f}% over the last {f.span_s / 60:.0f} min",),
                ("Check the heaviest processes; consider spreading work or lowering priority",),
            )
        )
    return out


def system_findings(f: Optional[SysFeat], input_conf: float, bottleneck: str) -> list[Insight]:
    """Whole-system observations not already stated by the verdict."""
    out: list[Insight] = []
    if f is None:
        return out
    if f.mem_now >= RAM_HIGH and bottleneck != "ram":
        out.append(
            Insight(
                "serious" if f.mem_now >= 95 else "warning",
                f"RAM {f.mem_now:.0f}% used",
                "Little headroom left: the OS will start swapping, and a training job may stall or be killed for "
                "running out of memory.",
            )
        )
    if swapping(f) and bottleneck != "ram":
        out.append(
            Insight(
                "warning",
                "Swap use rising",
                f"Swap grew {f.swap_slope:.1f}% per minute and is now {f.swap_now:.0f}% used. Paging to disk is "
                "orders of magnitude slower than RAM; reduce worker count or dataset caching.",
            )
        )
    if pegged(f) and bottleneck != "cpu":
        out.append(
            Insight(
                "warning" if input_conf else "info",
                "One CPU core pegged at ~100%",
                f"The busiest core averages {f.core_max_mean:.0f}% while overall CPU is {f.cpu_mean:.0f}%. A "
                "single-threaded step (Python GIL, data loading with num_workers=0, a tokenizer) may be pacing "
                "the job.",
            )
        )
    if input_conf and disk_heavy(f) and bottleneck != "disk":
        out.append(
            Insight(
                "warning",
                "Heavy disk reads while the GPU waits",
                f"Reads average {rate(f.disk_r_mean)} (hour peak {rate(f.disk_r_peak)}) during bursty GPU use. "
                "Storage may be pacing the input pipeline.",
            )
        )
    if f.cpu_temp is not None and f.cpu_temp >= 90:
        out.append(
            Insight(
                "warning",
                f"CPU running hot: {f.cpu_temp:.0f} °C",
                "Sustained CPU heat lowers boost clocks, which slows data loading and preprocessing.",
            )
        )
    return out


def system_health(f: Optional[SysFeat], input_conf: float) -> list[tuple[str, str, str, str]]:
    """(name, level, value, detail) rows for CPU, CPU temp, RAM, disk and network."""
    if f is None:
        return []
    out: list[tuple[str, str, str, str]] = []
    cval = f"{f.cpu_now:.0f}%"
    if f.cpu_freq:
        cval += f" @ {f.cpu_freq / 1000:.1f} GHz"
    if f.cpu_mean >= 95:
        out.append(("CPU", "serious", cval, "Saturated"))
    elif f.cpu_mean >= 85:
        out.append(("CPU", "warning", cval, "Nearly saturated"))
    elif pegged(f):
        out.append(("CPU", "warning", cval, f"One core pegged; average {f.cpu_mean:.0f}%"))
    else:
        out.append(("CPU", "good", cval, "Idle" if f.cpu_mean < 10 else "Healthy"))
    if f.cpu_temp is not None:
        t = f.cpu_temp
        level = "critical" if t >= 95 else "warning" if t >= 85 else "good"
        out.append(("CPU temp", level, f"{t:.0f} °C", "Hot" if t >= 85 else "Normal"))
    rval = f"{f.mem_now:.0f}%" + (f" · swap {f.swap_now:.0f}%" if f.swap_now is not None else "")
    if f.mem_now >= 95:
        out.append(("RAM", "critical", rval, "Nearly full: out-of-memory risk"))
    elif swapping(f):
        out.append(("RAM", "serious", rval, "Swapping: training may stall"))
    elif f.mem_now >= RAM_HIGH:
        out.append(("RAM", "warning", rval, "Little headroom left"))
    else:
        out.append(("RAM", "good", rval, "Healthy"))
    if f.disk_r_now is None and f.disk_w_now is None:
        out.append(("Disk", "info", "n/a", "No disk counters"))
    else:
        dval = f"{rate(f.disk_r_now)} read · {rate(f.disk_w_now)} write"
        if input_conf and disk_heavy(f):
            out.append(("Disk", "warning", dval, "Reads may be pacing the GPU"))
        else:
            busy = (f.disk_r_mean or 0) + (f.disk_w_mean or 0) >= 1e6
            out.append(("Disk", "good", dval, "Active" if busy else "Quiet"))
    if f.net_rx_now is None and f.net_tx_now is None:
        out.append(("Network", "info", "n/a", "No network counters"))
    else:
        nval = f"{rate(f.net_rx_now)} down · {rate(f.net_tx_now)} up"
        if input_conf and net_heavy(f):
            out.append(("Network", "warning", nval, "Downloads may be pacing the GPU"))
        else:
            busy = (f.net_rx_mean or 0) + (f.net_tx_mean or 0) >= 1e6
            out.append(("Network", "good", nval, "Active" if busy else "Quiet"))
    return out
