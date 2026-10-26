"""Synthetic readings for ``gpu-pulse --demo``: shows every animation and diagnosis on an idle machine.

The simulated GPU cycles through labelled phases (18 min) so each analysis rule fires in turn:

    steady    0-200 s   busy, unconstrained; VRAM creeping up (VRAM forecast)
    input   200-400 s   busy plateaus with short dips every 4.5 s, heavy host->GPU copies (input-bound)
    disk    400-600 s   same dips while disk reads run near 400 MB/s (disk-bound)
    launch  600-800 s   utilization ~95% but SMs ~15% active (launch-bound, needs GPM)
    power  800-1000 s   ~100% busy at the 180 W cap with the "power cap" throttle reason (power)
    idle  1000-1080 s   nothing running, VRAM released

With ``--demo-gpus N`` (N > 1) the simulated GPUs share that phase clock, like one
data-parallel job, and each extra GPU adds a pattern the cross-GPU analysis should catch:

    GPU 1   straggler: short bursts while the others are busy, and much less VRAM
    GPU 2   runs hot and reports "thermal" throttling while its siblings don't
    GPU 3   sits on a narrower PCIe link (x4 vs x8) and holds the most VRAM
    GPU 4+  well-behaved siblings with small offsets

With ``--demo-vendor apple`` one simulated Apple GPU follows the same phases (see ``demo_apple``).

Values are plausible, not measured. Nothing generated here is ever written to real history.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import replace
from typing import Any

from .collectors.system import SystemExtras
from .models import GpuDetails, GpuProcess, GpuSample

DEMO_UUID = "GPU-demo-0000"
DEMO_NAME = "DEMO GPU (simulated)"
STRAGGLER, HOT, NARROW = 1, 2, 3  # roles of the extra simulated GPUs (by index)


def demo_uuid(i: int) -> str:
    return DEMO_UUID if i == 0 else f"GPU-demo-{i:04d}"


def demo_name(i: int, n: int) -> str:
    return DEMO_NAME if n == 1 else f"DEMO GPU {i} (simulated)"


CYCLE_S = 1080.0
_PHASES = (
    ("steady", 200.0),
    ("input", 400.0),
    ("disk", 600.0),
    ("launch", 800.0),
    ("power", 1000.0),
    ("idle", 1080.0),
)
_rnd = random.Random(7)
_T0 = time.time()  # cycle starts when the demo starts, so it opens busy


def demo_phase(t: float) -> tuple[str, float]:
    """(phase name, seconds into that phase) at time t."""
    x = (t - _T0) % CYCLE_S
    start = 0.0
    for name, end in _PHASES:
        if x < end:
            return name, x - start
        start = end
    return "idle", 0.0


def _dip(t: float) -> bool:
    return (t - _T0) % 4.5 < 1.4  # ~1.4 s data-loading gap every 4.5 s


def demo_util(t: float) -> float:
    """Utilization (%) at time t (seconds)."""
    phase, _ = demo_phase(t)
    j = _rnd.uniform(-2, 2)
    if phase == "idle":
        return 1 + _rnd.random() * 3
    if phase in ("input", "disk"):
        return 4 + _rnd.random() * 10 if _dip(t) else 88 + j + 3 * math.sin(t / 7)
    if phase == "power":
        return min(100.0, 98.5 + j / 2)
    return 95 + j  # steady, launch


def demo_gpu(t: float) -> GpuSample:
    phase, into = demo_phase(t)
    u = demo_util(t)
    busy = u > 10
    if phase == "steady":
        vram = 50 + 22 * into / 200  # creeping allocation
    elif phase == "idle":
        vram = 8.0
    else:
        vram = 72 + 0.3 * math.sin(t / 9)
    if phase == "launch":  # kernels back to back, but tiny: SMs mostly idle
        sm, tensor, dram, occ, fp32 = 12 + _rnd.random() * 6, 2.0, 10.0, 10.0, 8.0
    elif busy:
        sm = u * (0.85 if phase == "power" else 0.72)
        tensor = u * (0.7 if phase == "power" else 0.55)
        dram, occ, fp32 = u * 0.5, 42.0, u * 0.22
    else:
        sm, tensor, dram, occ, fp32 = u * 0.3, 0.0, 1.0, 2.0, 0.2
    capped = phase == "power"
    temp = 62 + 8 * min(1.0, into / 200) if phase == "steady" else 76 if capped else 50 if phase == "idle" else 66
    return GpuSample(
        index=0,
        uuid=DEMO_UUID,
        name=DEMO_NAME,
        vendor="nvidia",
        util=round(u),
        mem_util=round(u * 0.55),
        mem_used_mib=round(16_311 * vram / 100, 1),
        mem_total_mib=16_311,
        temp_c=round(temp + _rnd.uniform(-0.5, 0.5)),
        power_w=round(min(180.0, 176 + _rnd.random() * 4) if capped else 22 + u * 1.35, 1),
        power_limit_w=180.0,
        clock_mhz=round(2450 + _rnd.random() * 100) if capped else (2750 if busy else 600),
        clock_max_mhz=3090.0,
        mem_clock_mhz=14_001.0,
        fan_pct=round(35 + u * 0.4),
        pstate="P2" if busy else "P8",
        pcie_gen=4,
        pcie_width=8,
        throttle=("power cap",) if capped else (),
        enc_util=0.0,
        dec_util=0.0,
        pcie_tx_bps=0.3e9 if busy else 2e6,
        pcie_rx_bps=(7e9 if phase == "input" else 1.0e9) if busy else 4e6,
        energy_j=(t - _T0) * 140.0,
        sm_active=round(sm, 1),
        sm_occupancy=occ,
        tensor_active=round(tensor, 1),
        dram_bw_util=round(dram, 1),
        fp32_active=round(fp32, 1),
        fp16_active=round(tensor * 0.6, 1),
        fp64_active=0.0,
        int_active=round(u * 0.1, 1),
    )


def demo_gpus(t: float, n: int = 1) -> tuple[GpuSample, ...]:
    """``n`` simulated GPUs at time t; GPU 0 is :func:`demo_gpu`, the others follow its phase."""
    base = demo_gpu(t)
    if n <= 1:
        return (base,)
    return tuple(_sibling(base, t, i, n) for i in range(n))


def _sibling(base: GpuSample, t: float, i: int, n: int) -> GpuSample:
    ident: dict[str, Any] = {"index": i, "uuid": demo_uuid(i), "name": demo_name(i, n)}
    if i == 0:
        return replace(base, **ident)
    phase, _ = demo_phase(t)
    total = base.mem_total_mib or 16_311.0
    u0 = base.util or 0.0
    busy = u0 >= 50
    if i == STRAGGLER:  # waits on the others: a short burst every 3 s, otherwise nearly idle
        u = (55 + _rnd.random() * 20 if (t - _T0) % 3.0 < 0.7 else 3 + _rnd.random() * 5) if busy else u0
        vram = 34.0 if phase != "idle" else 8.0
        return replace(
            base,
            **ident,
            util=round(u),
            mem_util=round(u * 0.4),
            mem_used_mib=round(total * vram / 100, 1),
            temp_c=round((base.temp_c or 50) - 9),
            power_w=round(24 + u * 0.9, 1),
            clock_mhz=2750.0 if u > 10 else 900.0,
            pstate="P2" if u > 10 else "P5",
            throttle=(),
            sm_active=round(u * 0.6, 1),
            tensor_active=round(u * 0.4, 1),
            dram_bw_util=round(u * 0.3, 1),
            fp32_active=round(u * 0.15, 1),
            fp16_active=round(u * 0.25, 1),
            int_active=round(u * 0.05, 1),
            pcie_rx_bps=3e8 if u > 10 else 5e6,
        )
    j = _rnd.uniform(-1.5, 1.5)
    u = max(0.0, min(100.0, u0 + j))
    vram = (base.vram_pct or 0.0) + (3.0 * math.sin(i)) if phase != "idle" else 8.0
    extra: dict[str, Any] = {}
    if i == HOT and busy:  # poor airflow: hot and clock-limited by temperature
        extra = {
            "temp_c": float(86 + round(_rnd.uniform(-0.5, 0.5))),
            "clock_mhz": 2310.0 + round(_rnd.random() * 60),
            "throttle": tuple(dict.fromkeys(base.throttle + ("thermal",))),
            "fan_pct": 100.0,
        }
    if i == NARROW:  # sits in a chipset slot: x4 link, and rank-0-style extra buffers
        extra = {"pcie_width": 4}
        if phase != "idle":
            vram = 92.0 + 0.3 * math.sin(t / 7)
    fields: dict[str, Any] = {
        **ident,
        "util": round(u),
        "mem_used_mib": round(total * max(0.0, min(100.0, vram)) / 100, 1),
        "temp_c": round((base.temp_c or 50) + (i % 3) - 1),
        "power_w": round(min(base.power_limit_w or 180.0, (base.power_w or 0.0) + j), 1),
        "sm_active": round(max(0.0, (base.sm_active or 0.0) + j), 1),
    }
    fields.update(extra)
    return replace(base, **fields)


def demo_processes(gpus: tuple[GpuSample, ...]) -> list[GpuProcess]:
    """One simulated training process (rank) per GPU that holds memory."""
    return [
        GpuProcess(
            pid=4200 + g.index, name="python.exe", used_mib=max(0.0, (g.mem_used_mib or 0) - 420), gpu_uuid=g.uuid
        )
        for g in gpus
        if (g.vram_pct or 0) > 20
    ]


def demo_system(t: float) -> SystemExtras:
    """Simulated swap/clock/temperature/disk/network readings that match the GPU phase."""
    phase, _ = demo_phase(t)
    disk = phase == "disk"
    return SystemExtras(
        swap_pct=3.0,
        cpu_freq_mhz=round(4200 + _rnd.uniform(-150, 150)),
        cpu_temp_c=round(68 if phase != "idle" else 45 + _rnd.uniform(-1, 1)),
        disk_read_bps=(380e6 + _rnd.random() * 40e6) if disk else 2e6 + _rnd.random() * 4e6,
        disk_write_bps=1e6 + _rnd.random() * 2e6,
        net_rx_bps=0.2e6 + _rnd.random() * 0.8e6,
        net_tx_bps=0.1e6,
    )


def demo_details_all(n: int = 1) -> list[GpuDetails]:
    """Static facts for each of ``n`` simulated GPUs (same board, own index / uuid / name)."""
    base = demo_details()
    if n <= 1:
        return [base]
    return [replace(base, index=i, uuid=demo_uuid(i), name=demo_name(i, n)) for i in range(n)]


def demo_details() -> GpuDetails:
    """Plausible static facts for the simulated GPU (modelled on a 16 GB Blackwell card)."""
    return GpuDetails(
        index=0,
        uuid=DEMO_UUID,
        name=DEMO_NAME,
        architecture="Blackwell",
        compute_capability="12.0",
        cuda_cores=4608,
        sm_count=36,
        tensor_core_gen="5th gen",
        memory_bus_bits=128,
        memory_bandwidth_gbs=448.0,
        fp32_tflops=28.5,
        max_sm_clock_mhz=3090.0,
        max_mem_clock_mhz=14_001.0,
        vram_total_mib=16_311.0,
        bar1_total_mib=16_384.0,
        pcie_max_gen=5,
        pcie_max_width=16,
        power_default_w=180.0,
        power_min_w=150.0,
        power_max_w=180.0,
        num_fans=2,
        vbios="simulated",
        driver="simulated",
        cuda_driver="simulated",
        compute_mode="default",
        display_active=False,
        gpm_supported=True,
    )


def _sibling_bucket(g: dict[str, Any], i: int) -> dict[str, Any]:
    """Per-minute bucket of extra simulated GPU ``i``, derived from GPU 0's (same roles as live)."""
    out = dict(g)
    if i == STRAGGLER:
        util = round(g["util"] * 0.3, 1)
        out.update(util=util, utilMax=min(100.0, util + 40), vram=34.0, sm=round(util * 0.6, 1))
        out.update(busy=0, idle=60 if util < 10 else 40, power=round(24 + util * 0.9, 2))
    elif i == HOT:
        out["tempMax"] = round(g["tempMax"] + 14, 1) if g["util"] > 10 else g["tempMax"]
    elif i == NARROW and g["util"] > 10:
        out["vram"] = 92.0
    return out


def demo_history(hours: int = 25, n: int = 1) -> dict[str, Any]:
    """A day of per-minute buckets in the history file format, with an overnight gap."""
    rnd = random.Random(7)
    now_min = int(time.time() // 60 * 60_000)
    minutes = []
    for i in range(hours * 60, 0, -1):
        t = now_min - i * 60_000
        if 2 <= time.localtime(t / 1000).tm_hour < 8:
            continue
        util = (
            3 + rnd.random() * 4
            if (t // 60_000) % 240 < 40
            else max(0, min(100, 78 + 14 * math.sin(t / 2.8e6) - rnd.random() * 15))
        )
        busy = int(60 * (util / 100) ** 1.5)
        idle = 60 - busy if util < 10 else 3
        g0 = {
            "n": 60,
            "util": round(util, 1),
            "utilMax": min(100.0, util + 12),
            "vram": 70.0 if util > 10 else 34.0,
            "power": round(20 + util * 1.4, 2),
            "tempMax": round(40 + util * 0.35, 1),
            "idle": idle,
            "busy": busy,
            "heldIdle": 0,
            "sm": round(util * 0.7, 1),
            "tensor": round(util * 0.45 if util > 10 else 0.0, 1),
        }
        gpus = {DEMO_UUID: g0}
        for i in range(1, n):
            gpus[demo_uuid(i)] = _sibling_bucket(g0, i)
        minutes.append({"t": t, "n": 60, "cpu": round(30 + rnd.random() * 20, 2), "mem": 48.0, "gpus": gpus})
    return {"version": 1, "minutes": minutes}
