"""Findings, forecasts and health rows for :mod:`gpu_pulse.intel` (GPU side).

Private module. Each finding needs data it can actually measure; wording stays honest about what
``utilization.gpu`` means (time a kernel ran) versus GPM "true usage" (SM / Tensor / DRAM activity).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from ._intel_gpu import (
    MIN_RAW,
    POWER_REASONS,
    THERMAL_REASONS,
    GpuFeat,
    GRow,
    fit,
    launch_bound,
    pct,
)
from .history import IDLE_UTIL
from .models import GpuSample, Insight


def vram_forecast(rows: Sequence[GRow], now: float) -> tuple[Optional[str], Optional[float]]:
    """("VRAM full in ~N min at current rate", eta minutes) from a least-squares fit, else (None, None).

    Needs >=20 samples over >=60 s, a rise of >=0.5 %/min and >=2 points in total, a fair linear
    fit (r² >= 0.6), and a projected fill within 60 minutes.
    """
    pts = [(s.t, v) for s, g in rows if s.t >= now - 180 for v in (g.vram_pct,) if v is not None]
    if len(pts) < MIN_RAW or pts[-1][0] - pts[0][0] < 60:
        return None, None
    xs = [t - pts[0][0] for t, _ in pts]
    slope, r2 = fit(xs, [v for _, v in pts])
    per_min = slope * 60
    last = pts[-1][1]
    if per_min < 0.5 or per_min * xs[-1] / 60 < 2 or r2 < 0.6 or last >= 99.5:
        return None, None
    eta = (100 - last) / per_min
    if eta > 60:
        return None, None
    label = "<1" if eta < 1 else f"{eta:.0f}"
    return f"VRAM full in ~{label} min at current rate", eta


def gpu_findings(rows: Sequence[GRow], now: float, temp_warn: float) -> list[Insight]:
    """Anomalies: sudden activity collapse, low clocks without a reason, fast temperature climb."""
    out: list[Insight] = []
    if len(rows) < MIN_RAW:
        return out
    # 1. Utilization collapse with memory still allocated.
    k = len(rows)
    while k > 0 and (rows[k - 1][1].util or 0) < 5:
        k -= 1
    if k < len(rows):
        quiet = now - rows[k][0].t
        before = [g for s, g in rows[:k] if s.t >= rows[k][0].t - 180]
        if 30 <= quiet <= 300 and len(before) >= MIN_RAW:  # later on, the verdict itself says idle
            prior = sum(g.util or 0 for g in before) / len(before)
            pv = [v for v in (g.vram_pct for g in before) if v is not None]
            vnow = rows[-1][1].vram_pct
            if prior >= 50 and pv and vnow is not None and vnow >= 10 and vnow >= 0.9 * (sum(pv) / len(pv)):
                out.append(
                    Insight(
                        "warning",
                        "GPU activity stopped suddenly",
                        f"Utilization fell from ~{prior:.0f}% to under 5% {quiet:.0f} s ago while {vnow:.0f}% VRAM "
                        "stays allocated. Could be a hang or deadlock (e.g. a stuck DataLoader worker), or a normal "
                        "pause such as evaluation or checkpointing.",
                    )
                )
    recent = [g for s, g in rows if s.t >= now - 120]
    # 2. Clocks well below max while busy, with no reason reported.
    busy = [g for g in recent if (g.util or 0) >= 80 and g.clock_mhz is not None and g.clock_max_mhz]
    if len(busy) >= 10 and not any(g.throttle for g in busy):
        ratio = sum(g.clock_mhz / g.clock_max_mhz for g in busy if g.clock_mhz and g.clock_max_mhz) / len(busy)
        if ratio < 0.6:
            mhz = sum(g.clock_mhz or 0 for g in busy) / len(busy)
            out.append(
                Insight(
                    "info",
                    "Clocks low while busy",
                    f"Graphics clock averaged {mhz:.0f} MHz ({pct(ratio)} of {busy[-1].clock_max_mhz:.0f} MHz max) at "
                    ">=80% utilization with no throttle reason reported. Possible causes: a power-saving mode "
                    "(OS power plan, driver power setting), locked clocks, or kernels too light to trigger boost.",
                )
            )
    # 3. Temperature climbing fast.
    tp = [(s.t, g.temp_c) for s, g in rows if s.t >= now - 120 and g.temp_c is not None]
    if len(tp) >= MIN_RAW and tp[-1][0] - tp[0][0] >= 60:
        slope, r2 = fit([t - tp[0][0] for t, _ in tp], [c for _, c in tp])
        per_min = slope * 60
        cur = tp[-1][1]
        if per_min >= 3 and r2 >= 0.5 and 55 <= cur < temp_warn:
            eta = (temp_warn - cur) / per_min
            out.append(
                Insight(
                    "warning" if eta <= 5 else "info",
                    f"Temperature climbing {per_min:+.1f} °C/min",
                    f"{tp[0][1]:.0f} → {cur:.0f} °C over the last {tp[-1][0] - tp[0][0]:.0f} s. At this rate it "
                    f"reaches {temp_warn:.0f} °C in ~{eta:.0f} min.",
                )
            )
    return out


PCIE_LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}  # per lane, per direction


def pcie_gbs(gen: Optional[int], width: Optional[int]) -> Optional[float]:
    """Theoretical one-direction PCIe bandwidth (GB/s) for a link, None if unknown."""
    lane = PCIE_LANE_GBS.get(gen) if gen else None
    return lane * width if lane and width else None


def gpm_findings(
    f: Optional[GpuFeat], bottleneck: str, input_conf: float, max_gen: Optional[int], max_width: Optional[int]
) -> list[Insight]:
    """True-usage (GPM) and PCIe findings; each needs data it can actually measure."""
    out: list[Insight] = []
    if f is None:
        return out
    if launch_bound(f) and bottleneck != "launch":
        out.append(
            Insight(
                "warning",
                f"Utilization reads {f.run_mean:.0f}% but SMs only {f.sm_mean or 0:.0f}% active",
                "utilization.gpu counts time any kernel runs; SM activity shows how much of the chip works. Few SMs "
                "have work: small grids, a waiting/spinning kernel (e.g. NCCL) or latency-bound code. Try a larger "
                "batch and profile the step to find the low-occupancy kernels.",
            )
        )
    if f.fp32_mean is not None and f.fp32_mean >= 30 and (f.tensor_mean or 0) < 5:
        out.append(
            Insight(
                "info",
                "Tensor Cores idle during FP32 work",
                f"FP32 pipe {f.fp32_mean:.0f}% active vs Tensor Cores {f.tensor_mean or 0:.0f}%. For deep learning, "
                "mixed precision (torch.autocast with bf16/fp16) or TF32 (torch.backends.cuda.matmul.allow_tf32 = "
                "True) moves matmuls onto the Tensor Cores.",
            )
        )
    if f.sm_mean is not None and f.sm_mean >= 50 and f.occupancy_mean is not None and f.occupancy_mean < 20:
        out.append(
            Insight(
                "info",
                f"Low SM occupancy ({f.occupancy_mean:.0f}%)",
                "SMs are active but few warps are resident, so memory latency is poorly hidden. Common causes are "
                "heavy register/shared-memory use or small grids. Not always a problem: tuned GEMMs often run at "
                "low occupancy.",
            )
        )
    cur = pcie_gbs(f.pcie_gen, f.pcie_width)
    best = pcie_gbs(max_gen, max_width)
    if f.run_mean >= 50 and cur and best and cur < best:
        out.append(
            Insight(
                "info",
                f"PCIe link Gen{f.pcie_gen} x{f.pcie_width} (GPU supports Gen{max_gen} x{max_width})",
                f"Host↔GPU transfers are capped near {cur:.1f} GB/s per direction (vs {best:.1f} GB/s). The slot, "
                "chipset or CPU lanes, or the card's own wiring, may set this. It matters only for transfer-heavy "
                "work such as streaming large batches from host memory.",
            )
        )
    if input_conf and cur and f.pcie_rx_mean is not None and f.pcie_rx_mean >= 0.3 * cur * 1e9:
        out.append(
            Insight(
                "warning",
                "Host→GPU transfers heavy while the GPU waits",
                f"{f.pcie_rx_mean / 1e9:.1f} GB/s average host→GPU ({pct(f.pcie_rx_mean / (cur * 1e9))} of the link) "
                "while utilization dips. Copies may be pacing the job: use pinned memory with non_blocking=True, "
                "keep data on the GPU, or send compact dtypes (uint8) and convert on the GPU.",
            )
        )
    return out


def gpu_health(
    g: Optional[GpuSample], f: Optional[GpuFeat], bottleneck: str, forecast: Optional[str], temp_warn: float
) -> list[tuple[str, str, str, str]]:
    """(name, level, value, detail) rows for GPU and VRAM (PCIe link notes are findings)."""
    if g is None:
        return [("GPU", "info", "n/a", "No supported GPU detected")]
    util = g.util
    value = "n/a" if util is None else f"util {util:.0f}%"
    if g.sm_active is not None:  # true usage first when GPM is available
        value = f"SM {g.sm_active:.0f}% · " + value
    if g.temp_c is not None:
        value += f" · {g.temp_c:.0f} °C"
    thr = set(g.throttle)
    if g.temp_c is not None and g.temp_c >= temp_warn:
        row = ("critical", f"Hot: at or above {temp_warn:.0f} °C")
    elif thr & THERMAL_REASONS or bottleneck == "thermal":
        row = ("serious", "Thermal throttling: clocks reduced")
    elif util is not None and util < IDLE_UTIL and (f is None or f.quiet_s >= 10):
        row = ("info", "Idle")
    elif bottleneck == "launch":
        row = ("warning", "Kernels running, but SMs mostly idle")
    elif bottleneck == "memory":
        row = ("warning", "Waiting on memory bandwidth")
    elif bottleneck in ("input", "disk", "network", "cpu", "ram"):
        row = ("warning", "Bursty: often waiting for work")
    elif thr & POWER_REASONS or bottleneck == "power":
        row = ("info", "At power cap (normal at full load)")
    elif f is not None and f.util_mean < IDLE_UTIL or (f is None and (util or 0) < IDLE_UTIL):
        row = ("info", "Idle")
    elif f is not None and f.util_mean >= 70:
        row = ("good", "Busy")
    else:
        row = ("good", "Light load")
    out = [("GPU", row[0], value, row[1])]
    v = g.vram_pct
    if v is not None:
        vval = f"{v:.0f}%"
        name = "GPU memory" if g.unified_memory else "VRAM"  # Apple Silicon: a share of unified memory
        if v >= 95:
            out.append((name, "serious", vval, "Nearly full: out-of-memory risk"))
        elif forecast:
            out.append((name, "warning", vval, forecast))
        elif bottleneck == "vram_held":
            out.append((name, "warning", vval, "Held while the GPU idles"))
        else:
            out.append((name, "good", vval, "Healthy · unified" if g.unified_memory else "Healthy"))
    return out
