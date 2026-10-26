"""Pure parsers for macOS GPU data: ``ioreg`` IOAccelerator plists and ``system_profiler`` JSON.

No I/O and no exceptions: malformed input gives fewer or emptier records. Key names are matched
through alias tables (first present wins) because they come from public tools and forum posts, not
from a capture verified here; ``scripts/mac_probe.py`` prints what a real Mac reports. Display
entries from system_profiler (``spdisplays_ndrvs``, which carry monitor serial numbers) are never kept.
"""

from __future__ import annotations

import json
import logging
import math
import plistlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from ..models import GpuDetails, GpuSample, GpuVendor

logger = logging.getLogger(__name__)

MIB = 1048576.0
# PerformanceStatistics aliases, first present wins (unverified names: see the module docstring).
UTIL_KEYS = ("Device Utilization %", "GPU Activity(%)")
RENDERER_KEYS = ("Renderer Utilization %",)
TILER_KEYS = ("Tiler Utilization %",)
SYSTEM_MEM_KEYS = ("In use system memory", "Alloc system memory")  # bytes: Apple Silicon, Intel iGPU
VRAM_USED_KEYS = ("vramUsedBytes",)  # bytes: AMD GPUs in Intel Macs
VRAM_FREE_KEYS = ("vramFreeBytes",)
VRAM_TOTAL_MB_KEYS = ("VRAM,totalMB",)  # MiB, in PerformanceStatistics or on the accelerator itself
TEMP_KEYS = ("Temperature(C)",)  # AMD GPUs in Intel Macs only; Apple Silicon has none here
FAN_KEYS = ("Fan Speed(%)",)
CLOCK_KEYS = ("Core Clock(MHz)",)
MEM_CLOCK_KEYS = ("Memory Clock(MHz)",)
CORE_KEYS = ("gpu-core-count",)
_GENERIC = {"apple": "Apple GPU", "amd": "AMD GPU", "intel": "Intel GPU", "nvidia": "NVIDIA GPU"}


@dataclass(frozen=True)
class IoregInfo:
    """Static facts of one accelerator in ``ioreg`` output (alongside its live sample)."""

    ioclass: Optional[str]
    cores: Optional[int]
    named: bool  # the name came from ioreg's "model" key, not a generic fallback


@dataclass(frozen=True)
class ProfilerGpu:
    """One GPU from ``system_profiler SPDisplaysDataType -json`` (display entries dropped)."""

    model: Optional[str]
    cores: Optional[int]
    metal: Optional[str]
    vram_mib: Optional[float]


# ---- value parsing (never raises) ----------------------------------------------------------
def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (str, bytes)):
        try:
            f = float(v.strip())
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _first(d: Optional[dict[str, Any]], keys: Sequence[str]) -> Optional[float]:
    if not d:
        return None
    for k in keys:
        v = _num(d.get(k))
        if v is not None:
            return v
    return None


def _pct(v: Optional[float]) -> Optional[float]:
    return None if v is None or v < 0 or v > 105 else min(100.0, v)


def _bytes_mib(v: Optional[float]) -> Optional[float]:
    return None if v is None or v < 0 else v / MIB


def _in(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    return v if v is not None and lo <= v <= hi else None


def _text(v: Any) -> Optional[str]:
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    if not isinstance(v, str):
        return None
    s = v.replace("\x00", "").strip()
    return s or None


def _int(v: Any) -> Optional[int]:
    n = _num(v)
    return int(n) if n is not None and n >= 0 else None


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def gpu_uuid(vendor: str, index: int, label: str) -> str:
    """Stable id from what ioreg reports (vendor, position, model or class), e.g. ``apple-0-apple-m1-pro``."""
    return f"{vendor}-{index}-{slug(label) or 'gpu'}"


def _vendor(entry: dict[str, Any], model: Optional[str]) -> GpuVendor:
    ident = " ".join(_text(entry.get(k)) or "" for k in ("IOClass", "IOName", "IOMatchCategory")).lower()
    bundle = (_text(entry.get("CFBundleIdentifier")) or "").lower()
    if "agx" in ident or "agx" in bundle or "paravirt" in ident or (model or "").startswith("Apple "):
        return "apple"
    words = f"{ident} {bundle} {(model or '').lower()}"
    if "amd" in words or "radeon" in words:
        return "amd"
    if "nvda" in words or "nvidia" in words or "geforce" in words:
        return "nvidia"
    if "intel" in words:
        return "intel"
    return "apple"  # an unrecognised accelerator on macOS: most likely the Mac's own GPU


# ---- ioreg ---------------------------------------------------------------------------------
def _entries(data: bytes) -> list[dict[str, Any]]:
    if not data or not data.strip():
        return []
    try:
        root = plistlib.loads(data)
    except Exception as e:  # noqa: BLE001 - InvalidFileException, ExpatError, ValueError…: treat as no data
        logger.debug("ioreg output is not a plist: %s", e)
        return []
    if isinstance(root, dict):
        root = [root]
    return [e for e in root if isinstance(e, dict)] if isinstance(root, list) else []


def _memory(
    entry: dict[str, Any], ps: dict[str, Any], vendor: str, unified_total_mib: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """(used MiB, total MiB): dedicated VRAM when reported, else system memory in use by the GPU."""
    total_mb = _first(ps, VRAM_TOTAL_MB_KEYS) or _first(entry, VRAM_TOTAL_MB_KEYS)
    used = _bytes_mib(_first(ps, VRAM_USED_KEYS))
    if used is not None:
        free = _bytes_mib(_first(ps, VRAM_FREE_KEYS))
        total = total_mb if total_mb and total_mb > 0 else (used + free if free is not None else None)
        return used, total
    used = _bytes_mib(_first(ps, SYSTEM_MEM_KEYS))
    if vendor == "apple":
        return used, unified_total_mib if unified_total_mib and unified_total_mib > 0 else None
    return used, total_mb if total_mb and total_mb > 0 else None


def parse_ioreg(data: bytes, unified_total_mib: Optional[float] = None) -> list[tuple[GpuSample, IoregInfo]]:
    """Every accelerator that reports PerformanceStatistics, as (sample, static info). Never raises.

    ``unified_total_mib`` is the system's memory (Apple Silicon GPUs share it): the total shown for them.
    """
    out: list[tuple[GpuSample, IoregInfo]] = []
    for entry in _entries(data):
        ps = entry.get("PerformanceStatistics")
        if not isinstance(ps, dict):
            continue  # e.g. a VM's display adapter: nothing to monitor
        model = _text(entry.get("model"))
        vendor = _vendor(entry, model)
        renderer, tiler = _pct(_first(ps, RENDERER_KEYS)), _pct(_first(ps, TILER_KEYS))
        util = _pct(_first(ps, UTIL_KEYS))
        if util is None and (renderer is not None or tiler is not None):
            util = max(v for v in (renderer, tiler) if v is not None)
        used, total = _memory(entry, ps, vendor, unified_total_mib)
        if util is None and used is None:
            continue  # nothing we understand in its statistics
        i = len(out)
        ioclass = _text(entry.get("IOClass"))
        name = model or _GENERIC[vendor]
        sample = GpuSample(
            index=i,
            uuid=gpu_uuid(vendor, i, model or ioclass or vendor),
            name=name,
            vendor=vendor,
            util=util,
            mem_used_mib=used,
            mem_total_mib=total,
            temp_c=_in(_first(ps, TEMP_KEYS), -20, 150),
            clock_mhz=_in(_first(ps, CLOCK_KEYS), 1, 10_000),
            mem_clock_mhz=_in(_first(ps, MEM_CLOCK_KEYS), 1, 50_000),
            fan_pct=_pct(_first(ps, FAN_KEYS)),
            renderer_util=renderer,
            tiler_util=tiler,
        )
        conf = entry.get("GPUConfigurationVariable")
        cores = _int(_first(entry, CORE_KEYS)) or _int(_first(conf if isinstance(conf, dict) else None, ("num_cores",)))
        out.append((sample, IoregInfo(ioclass=ioclass, cores=cores, named=model is not None)))
    return out


def parse_ioreg_plist(data: bytes, unified_total_mib: Optional[float] = None) -> list[GpuSample]:
    """Live samples from ``ioreg -r -d 1 -w 0 -c IOAccelerator -a`` output; ``[]`` when there is none."""
    return [s for s, _ in parse_ioreg(data, unified_total_mib)]


# ---- system_profiler -----------------------------------------------------------------------
def metal_label(v: Any) -> Optional[str]:
    """``"spdisplays_metal3"`` -> ``"Metal 3"``; older family names are kept readable."""
    s = (_text(v) or "").lower()
    if not s:
        return None
    m = re.search(r"metal[ _]?(\d+)", s)
    if m:
        return f"Metal {m.group(1)}"
    m = re.search(r"mtlgpufamilymac(\d+)", s)
    if m:
        return f"Metal (Mac family {m.group(1)})"
    if s.endswith("supported"):
        return "Metal"
    return s.replace("spdisplays_", "").replace("_", " ") or None


def _size_mib(v: Any) -> Optional[float]:
    m = re.match(r"\s*([\d.]+)\s*(GB|MB)", _text(v) or "", re.IGNORECASE)
    if not m:
        return None
    try:
        n = float(m.group(1))
    except ValueError:
        return None
    return n * 1024 if m.group(2).upper() == "GB" else n


def parse_profiler_json(data: bytes) -> list[ProfilerGpu]:
    """GPUs from ``system_profiler SPDisplaysDataType -json``; display entries and serials are dropped."""
    try:
        root = json.loads(data.decode("utf-8", "replace")) if data else None
    except ValueError:
        return []
    items = root.get("SPDisplaysDataType") if isinstance(root, dict) else None
    out: list[ProfilerGpu] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        metal = next((it[k] for k in ("spdisplays_mtlgpufamilysupport", "spdisplays_metal") if k in it), None)
        out.append(
            ProfilerGpu(
                model=_text(it.get("sppci_model")) or _text(it.get("_name")),
                cores=_int(it.get("sppci_cores")),
                metal=metal_label(metal),
                vram_mib=_size_mib(it.get("spdisplays_vram") or it.get("_spdisplays_vram")),
            )
        )
    return out


def _match(g: GpuSample, pos: int, n: int, prof: Sequence[ProfilerGpu]) -> Optional[ProfilerGpu]:
    same = [p for p in prof if p.model and p.model.lower() == g.name.lower()]
    if same:
        return same[0]
    return prof[pos] if len(prof) == n else None


def apple_details(
    gpus: Sequence[GpuSample], info: Sequence[IoregInfo], prof: Sequence[ProfilerGpu]
) -> list[GpuDetails]:
    """Static facts per GPU, merging ioreg (identity, core count) with system_profiler (model, Metal)."""
    out: list[GpuDetails] = []
    for pos, g in enumerate(gpus):
        inf = info[pos] if pos < len(info) else IoregInfo(None, None, True)
        p = _match(g, pos, len(gpus), prof)
        gen = re.match(r"AGXAccelerator(\w+)", inf.ioclass or "")
        unified = g.unified_memory
        name = g.name if inf.named or not (p and p.model) else p.model
        out.append(
            GpuDetails(
                index=g.index,
                uuid=g.uuid,
                name=name,
                architecture=("Apple GPU" + (f" ({gen.group(1)})" if gen else "")) if unified else None,
                gpu_cores=(p.cores if p else None) or inf.cores,
                metal_family=p.metal if p else None,
                unified_memory=unified,
                vram_total_mib=g.mem_total_mib if unified or not (p and p.vram_mib) else p.vram_mib,
            )
        )
    return out
