"""GPUs on macOS (Apple Silicon first) through the IORegistry: no admin rights, no extra dependency.

Live readings: ``ioreg -r -d 1 -w 0 -c IOAccelerator -a``, one XML plist dict per accelerator. Apple
Silicon GPUs (IOClass ``AGXAccelerator…``) are expected to carry ``model`` ("Apple M1 Pro"),
``gpu-core-count`` and PerformanceStatistics ("Device / Renderer / Tiler Utilization %", "In use
system memory" in bytes). Intel Macs list each GPU (Intel iGPU, AMD dGPU) separately; AMD ones add
``vramUsedBytes`` / ``vramFreeBytes``. Parsing lives in :mod:`.apple_parse` (pure, fixture-tested).

Static facts: ``system_profiler SPDisplaysDataType -json``, read once (model, cores, Metal support).

Not read: temperature, power and clocks on Apple Silicon (``powermetrics`` needs root), per-process
GPU use. Those fields stay ``None`` and the UI shows n/a.

Cost: one ``ioreg`` process per sample, so the monitor samples Apple GPUs at most every
``MIN_INTERVAL_S`` seconds until that cost is measured on real hardware (``scripts/mac_probe.py``).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from typing import Callable, Optional

from ..models import GpuDetails, GpuSample
from .apple_parse import (
    MIB,
    IoregInfo,
    ProfilerGpu,
    apple_details,
    gpu_uuid,
    metal_label,
    parse_ioreg,
    parse_ioreg_plist,
    parse_profiler_json,
)

__all__ = [
    "AppleCollector",
    "IoregInfo",
    "ProfilerGpu",
    "apple_details",
    "gpu_uuid",
    "metal_label",
    "parse_ioreg",
    "parse_ioreg_plist",
    "parse_profiler_json",
    "run_command",
    "system_memory_mib",
]

logger = logging.getLogger(__name__)

IOREG_ARGS = ("-r", "-d", "1", "-w", "0", "-c", "IOAccelerator", "-a")
PROFILER_ARGS = ("SPDisplaysDataType", "-json")
IOREG_TIMEOUT_S = 2.0
PROFILER_TIMEOUT_S = 20.0  # system_profiler takes 1-5 s; it runs once, off the UI thread
MIN_INTERVAL_S = 1.0  # provisional floor (one process spawn per sample) until measured on a Mac
MAX_MISSES = 5  # failed reads in a row shown as "no reading" before the GPUs are dropped

Runner = Callable[[Sequence[str], float], Optional[bytes]]


def _tool(name: str) -> str:
    for folder in ("/usr/sbin", "/usr/bin"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return shutil.which(name) or name


def run_command(cmd: Sequence[str], timeout: float) -> Optional[bytes]:
    """stdout of ``cmd`` (no shell), or None when it is missing, fails or times out."""
    try:
        r = subprocess.run(
            list(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as e:  # TimeoutExpired is a SubprocessError
        logger.debug("%s failed: %s", cmd[0], e)
        return None
    if r.returncode != 0:
        logger.debug("%s exited with %s", cmd[0], r.returncode)
        return None
    return r.stdout


def system_memory_mib() -> Optional[float]:
    try:
        import psutil

        return psutil.virtual_memory().total / MIB
    except (ImportError, OSError, RuntimeError, AttributeError):
        return None


class AppleCollector:
    """macOS GPUs via ``ioreg`` (live) and ``system_profiler`` (static, once).

    ``runner`` and ``platform`` are injectable so the whole path is testable without a Mac.
    """

    def __init__(
        self, runner: Optional[Runner] = None, platform: Optional[str] = None, unified_total_mib: Optional[float] = None
    ) -> None:
        self._run = runner or run_command
        self._platform = platform or sys.platform
        self._unified = unified_total_mib
        self._ids: list[GpuSample] = []  # GPUs of the last good read (identity for failed reads)
        self._info: list[IoregInfo] = []
        self._details: Optional[list[GpuDetails]] = None
        self._names: dict[str, str] = {}  # uuid -> model name from system_profiler (ioreg had none)
        self._misses = 0

    def _unified_mib(self) -> Optional[float]:
        if self._unified is None:
            self._unified = system_memory_mib() or 0.0
        return self._unified or None

    def _read(self) -> list[tuple[GpuSample, IoregInfo]]:
        data = self._run((_tool("ioreg"), *IOREG_ARGS), IOREG_TIMEOUT_S)
        return parse_ioreg(data, self._unified_mib()) if data else []

    def detect(self) -> bool:
        if self._platform != "darwin":
            return False
        rows = self._read()
        if rows:
            self._ids, self._info = [g for g, _ in rows], [i for _, i in rows]
        return bool(rows)

    def sample(self) -> tuple[GpuSample, ...]:
        rows = self._read()
        names = self._names
        if not rows:  # spawn failed / timed out: keep the GPUs on screen without a reading
            self._misses += 1
            if self._misses > MAX_MISSES:
                return ()
            return tuple(GpuSample(g.index, g.uuid, names.get(g.uuid, g.name), g.vendor) for g in self._ids)
        self._misses = 0
        gpus = [g for g, _ in rows]
        if [g.uuid for g in gpus] != [g.uuid for g in self._ids]:  # a GPU appeared or went (eGPU)
            self._ids, self._info, self._details = gpus, [i for _, i in rows], None
        return tuple(replace(g, name=names[g.uuid]) if g.uuid in names else g for g in gpus)

    def details(self) -> list[GpuDetails]:
        """Static facts per GPU; system_profiler runs once (1-5 s) and is cached."""
        if self._details is None:
            data = self._run((_tool("system_profiler"), *PROFILER_ARGS), PROFILER_TIMEOUT_S)
            details = apple_details(self._ids, self._info, parse_profiler_json(data) if data else [])
            for g, d in zip(self._ids, details):
                if d.name != g.name:
                    self._names[g.uuid] = d.name
            self._details = details
        return self._details
