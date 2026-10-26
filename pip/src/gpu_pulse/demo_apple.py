"""Simulated Apple GPU for ``gpu-pulse --demo --demo-vendor apple``.

One GPU follows the NVIDIA demo's phases (see :mod:`gpu_pulse.demo`). Its readings are built as
``ioreg`` / ``system_profiler`` output and go through the real macOS parsers, so the UI shows what a
Mac would: utilization, renderer/tiler and unified memory, but no temperature, power or clocks.
"""

from __future__ import annotations

import json
import math
import plistlib
import time
from typing import Any, Optional

from .collectors.apple import apple_details, gpu_uuid, parse_ioreg, parse_profiler_json, system_memory_mib
from .demo import DEMO_UUID, demo_history, demo_phase, demo_util
from .models import GpuDetails, GpuSample

DEMO_APPLE_MODEL = "Apple M1 Pro (simulated)"
DEMO_APPLE_UUID = gpu_uuid("apple", 0, DEMO_APPLE_MODEL)
_unified_mib: Optional[float] = None


def _unified() -> Optional[float]:
    """This machine's RAM, as the Apple collector uses it (so the GPU memory and RAM rings agree)."""
    global _unified_mib
    if _unified_mib is None:
        _unified_mib = system_memory_mib() or 16_384.0
    return _unified_mib


def demo_apple_ioreg(t: float) -> bytes:
    """``ioreg -r -d 1 -w 0 -c IOAccelerator -a`` output for the simulated Apple GPU at time t."""
    phase, into = demo_phase(t)
    u = demo_util(t)
    if phase == "steady":
        gib = 3.5 + 4.5 * into / 200  # creeping allocation
    else:
        gib = 1.6 if phase == "idle" else 8.0 + 0.05 * math.sin(t / 9)
    in_use = int(gib * 2**30)
    stats = {
        "Device Utilization %": round(u),
        "Renderer Utilization %": round(u * 0.92),
        "Tiler Utilization %": round(u * 0.35),
        "In use system memory": in_use,
        "Alloc system memory": int(in_use * 1.6),
        "recoveryCount": 0,
    }
    entry = {
        "IOClass": "AGXAcceleratorG13X",
        "CFBundleIdentifier": "com.apple.AGXG13X",
        "model": DEMO_APPLE_MODEL,
        "gpu-core-count": 16,
        "PerformanceStatistics": stats,
    }
    return plistlib.dumps([entry])


def demo_apple_profiler() -> bytes:
    """``system_profiler SPDisplaysDataType -json`` output for the simulated Apple GPU."""
    gpu = {
        "_name": DEMO_APPLE_MODEL,
        "sppci_model": DEMO_APPLE_MODEL,
        "sppci_cores": "16",
        "spdisplays_mtlgpufamilysupport": "spdisplays_metal3",
        "spdisplays_vendor": "sppci_vendor_Apple",
    }
    return json.dumps({"SPDisplaysDataType": [gpu]}).encode()


def demo_apple_gpus(t: float) -> tuple[GpuSample, ...]:
    """The simulated Apple GPU at time t, parsed exactly like a real ``ioreg`` read."""
    return tuple(g for g, _ in parse_ioreg(demo_apple_ioreg(t), _unified()))


def demo_apple_details() -> list[GpuDetails]:
    rows = parse_ioreg(demo_apple_ioreg(time.time()), _unified())
    return apple_details([g for g, _ in rows], [i for _, i in rows], parse_profiler_json(demo_apple_profiler()))


def demo_apple_history(hours: int = 25) -> dict[str, Any]:
    """The NVIDIA demo day re-keyed for the Apple GPU: no power/temperature/GPM, unified-memory share.

    Power and temperature are 0, as the collector records missing readings in the history format.
    """
    base = demo_history(hours, 1)
    for m in base["minutes"]:
        g = dict(m["gpus"].pop(DEMO_UUID))
        g.pop("sm", None)
        g.pop("tensor", None)
        g.update(power=0.0, tempMax=0.0, vram=14.0 if g["util"] > 10 else 5.0)
        m["gpus"][DEMO_APPLE_UUID] = g
    return base
