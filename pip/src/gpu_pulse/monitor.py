"""Ties collectors and history together: one ``tick()`` = one sample recorded."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Optional

from platformdirs import user_data_dir

from .collectors import AmdLinuxCollector, AppleMetalCollector, NvidiaCollector, SystemCollector, collect_specs
from .collectors.system import SystemExtras
from .fleet import Fleet, analyze_fleet
from .history import HistoryFile, HistoryStore
from .insights import build_insights, grade
from .intel import Diagnosis, diagnose
from .models import GpuDetails, GpuProcess, GpuSample, Insight, Sample, Specs, WindowStats

logger = logging.getLogger(__name__)

WINDOWS: tuple[tuple[str, int], ...] = (("5 min", 5), ("30 min", 30), ("1 hour", 60), ("24 hours", 1440))
SAVE_EVERY_S = 60.0


@dataclass(frozen=True)
class MonitorConfig:
    interval_s: float = 1.0
    history_days: int = 7
    temp_warn: float = 80.0
    history_path: Optional[Path] = None  # None -> platform user data dir
    persist: bool = True  # write history back to disk
    demo: bool = False  # simulated GPU + seeded history; never touches the real history file
    demo_gpus: int = 1  # number of simulated GPUs in demo mode


@dataclass(frozen=True)
class Analysis:
    windows: tuple[WindowStats, ...]
    grade: Optional[tuple[str, str]]
    insights: tuple[Insight, ...]


def default_history_path() -> Path:
    return Path(user_data_dir("gpu-pulse", appauthor=False)) / "history.json"


class Monitor:
    def __init__(self, cfg: MonitorConfig) -> None:
        self.cfg = cfg
        self.nvidia = NvidiaCollector()
        self.amd = AmdLinuxCollector()
        self.apple = AppleMetalCollector()
        self.system = SystemCollector()
        self.vendor: Literal["nvidia", "amd", "apple", "none"] = "none"
        if self.nvidia.detect():
            self.vendor = "nvidia"
        elif self.amd.detect():
            self.vendor = "amd"
        elif self.apple.detect():
            self.vendor = "apple"
        self.history = HistoryStore(cfg.history_days)
        self.file = HistoryFile(cfg.history_path or default_history_path())
        if cfg.demo:
            from .demo import demo_history

            self.history.load(demo_history(n=cfg.demo_gpus))
        else:
            # Always read existing history; `persist` only controls writing (status/export are read-only).
            self.history.load(self.file.read())
        self.latest: Optional[Sample] = None
        self.processes: list[GpuProcess] = []
        self._specs: Optional[Specs] = None
        self._details: Optional[list[GpuDetails]] = None
        self._last_save = time.monotonic()
        self._last_procs = 0.0

    def _read_gpus(self) -> tuple[GpuSample, ...]:
        if self.cfg.demo:
            from .demo import demo_gpus

            return demo_gpus(time.time(), self.cfg.demo_gpus)
        try:
            if self.vendor == "nvidia":
                return self.nvidia.sample()
            if self.vendor == "apple":
                return self.apple.sample()
            return self.amd.sample() if self.vendor == "amd" else ()
        except Exception as e:  # noqa: BLE001 - a failed read must not stop recording
            logger.warning("GPU read failed: %s", e)
            return ()

    def _read_extras(self, t: float) -> SystemExtras:
        if self.cfg.demo:
            from .demo import demo_system

            return demo_system(t)
        try:
            return self.system.extras()
        except Exception as e:  # noqa: BLE001 - extras are optional; recording must go on
            logger.warning("System extras read failed: %s", e)
            return SystemExtras()

    def tick(self) -> Sample:
        gpus = self._read_gpus()
        cpu, per_core = self.system.cpu()
        used, total = self.system.mem()
        t = time.time()
        x = self._read_extras(t)
        s = Sample(
            t=t,
            gpus=gpus,
            cpu_util=cpu,
            cpu_per_core=per_core,
            mem_used=used,
            mem_total=total,
            swap_pct=x.swap_pct,
            cpu_freq_mhz=x.cpu_freq_mhz,
            cpu_temp_c=x.cpu_temp_c,
            disk_read_bps=x.disk_read_bps,
            disk_write_bps=x.disk_write_bps,
            net_rx_bps=x.net_rx_bps,
            net_tx_bps=x.net_tx_bps,
        )
        self.latest = s
        self.history.add(s)
        if s.t - self._last_procs >= 5:
            self._last_procs = s.t
            if self.cfg.demo:
                from .demo import demo_processes

                self.processes = demo_processes(gpus)
            elif self.vendor == "nvidia":
                self.processes = self.nvidia.processes()
        if self.cfg.persist and not self.cfg.demo and time.monotonic() - self._last_save >= SAVE_EVERY_S:
            self.save()
        return s

    def save(self) -> bool:
        """Persist history; False if the write was skipped or failed (logged)."""
        self._last_save = time.monotonic()
        try:
            return self.file.save(self.history)
        except OSError as e:
            logger.warning("Could not save history: %s", e)
            return False

    def specs(self) -> Specs:
        if self._specs is None:
            names = tuple(g.name for g in (self.latest.gpus if self.latest else ()))
            driver, cuda = self.nvidia.driver_info()
            specs = collect_specs(names, driver, cuda)
            if self.cfg.demo:  # demo screenshots get shared: don't show the real machine's name
                specs = replace(specs, hostname="gpu-pulse-demo")
            self._specs = specs
        return self._specs

    def gpu_details(self) -> list[GpuDetails]:
        """Static facts per GPU (architecture, cores, bus, limits); cached after the first call."""
        if self._details is None:
            if self.cfg.demo:
                from .demo import demo_details_all

                self._details = demo_details_all(self.cfg.demo_gpus)
            elif self.vendor == "nvidia":
                try:
                    self._details = self.nvidia.details()
                except Exception as e:  # noqa: BLE001 - details are informational
                    logger.warning("GPU details read failed: %s", e)
                    self._details = []
            else:
                self._details = []
        return self._details

    def diagnose(self, gpu_uuid: Optional[str]) -> Diagnosis:
        """``intel.diagnose`` with this monitor's temperature threshold and the GPU's static details."""
        det = next((d for d in self.gpu_details() if d.uuid == gpu_uuid), None)
        return diagnose(self.history, gpu_uuid, temp_warn=self.cfg.temp_warn, details=det)

    def fleet(self) -> Fleet:
        """Cross-GPU comparison (stragglers, uneven VRAM, lone throttling, PCIe links); empty for one GPU."""
        # The process list (NVML or demo) tells GPUs running one job apart from unrelated ones.
        procs = self.processes if self.cfg.demo or self.vendor == "nvidia" else None
        return analyze_fleet(self.history, self.gpu_details(), processes=procs)

    def analysis(self, gpu_uuid: Optional[str]) -> Analysis:
        windows = tuple(self.history.stats(label, m, gpu_uuid) for label, m in WINDOWS)
        gpu = next((g for g in (self.latest.gpus if self.latest else ()) if g.uuid == gpu_uuid), None)
        return Analysis(
            windows=windows,
            # Same 5-minute floor as the insights, so one idle minute doesn't read as a verdict.
            grade=grade(windows[2].util) if windows[2].covered_minutes >= 5 else None,
            insights=tuple(build_insights(windows[2], windows[1], gpu, self.cfg.temp_warn)),
        )

    def close(self) -> None:
        if self.cfg.persist and not self.cfg.demo:
            self.save()
        self.nvidia.close()
