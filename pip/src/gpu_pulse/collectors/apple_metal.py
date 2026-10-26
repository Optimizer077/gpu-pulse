"""Apple Silicon / Metal GPU counters from IORegistry, without sudo or extra dependencies."""

from __future__ import annotations

import math
import plistlib
import subprocess
import sys
from typing import Any, Optional
from xml.parsers.expat import ExpatError

from ..models import GpuSample


def parse_apple_gpus(data: bytes) -> tuple[GpuSample, ...]:
    """Decode accelerator properties; missing/invalid counters stay unavailable."""
    entries = plistlib.loads(data)
    if not isinstance(entries, list):
        raise ValueError("Expected an IORegistry array")
    out: list[GpuSample] = []
    for entry in entries:
        if not isinstance(entry, dict) or not str(entry.get("IOClass", "")).startswith("AGX"):
            continue
        stats = entry.get("PerformanceStatistics", {})
        if not isinstance(stats, dict):
            stats = {}

        def number(key: str, maximum: float = math.inf, stats: dict[str, Any] = stats) -> Optional[float]:
            value = stats.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value) if math.isfinite(value) and 0 <= value <= maximum else None

        util = number("Device Utilization %", 100)
        if util is None:
            util = number("Renderer Utilization %", 100)
        allocated = number("Alloc system memory")
        name = entry.get("model")
        out.append(GpuSample(
            index=len(out), uuid=f"apple-{len(out)}", name=name if isinstance(name, str) else "Apple GPU (Metal)",
            vendor="apple", util=util, mem_used_mib=allocated / 1048576 if allocated is not None else None,
            # System RAM is shared, not a dedicated VRAM budget. Percentage-based alerts stay disabled.
            mem_total_mib=None,
        ))
    return tuple(out)


class AppleMetalCollector:
    @staticmethod
    def sample() -> tuple[GpuSample, ...]:
        data = subprocess.check_output(
            ["/usr/sbin/ioreg", "-a", "-r", "-c", "IOAccelerator", "-d", "1"],
            timeout=4, stderr=subprocess.DEVNULL,
        )
        return parse_apple_gpus(data)

    def detect(self) -> bool:
        if sys.platform != "darwin":
            return False
        try:
            return bool(self.sample())
        except (OSError, subprocess.SubprocessError, ValueError, ExpatError):
            return False
