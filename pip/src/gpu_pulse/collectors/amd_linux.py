"""Best-effort AMD GPU reader for the Linux amdgpu driver via sysfs (no rocm-smi needed)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from ..models import GpuSample

DRM = Path("/sys/class/drm")


def _num(p: Path) -> Optional[float]:
    try:
        return float(p.read_text().strip())
    except (OSError, ValueError):
        return None


def _hwmon(dev: Path) -> Optional[Path]:
    try:
        return next((dev / "hwmon").iterdir())
    except (OSError, StopIteration):
        return None


class AmdLinuxCollector:
    def __init__(self, root: Path = DRM) -> None:
        self.root = root
        self.cards: list[Path] = []

    def detect(self) -> bool:
        if not sys.platform.startswith("linux") and self.root == DRM:
            return False
        try:
            self.cards = sorted(
                d / "device"
                for d in self.root.iterdir()
                if d.name.startswith("card") and d.name[4:].isdigit() and (d / "device" / "gpu_busy_percent").exists()
            )
        except OSError:
            self.cards = []
        return bool(self.cards)

    def sample(self) -> tuple[GpuSample, ...]:
        out = []
        for i, dev in enumerate(self.cards):
            hw = _hwmon(dev)
            card = dev.parent.name
            used, total = _num(dev / "mem_info_vram_used"), _num(dev / "mem_info_vram_total")
            power = (_num(hw / "power1_average") or _num(hw / "power1_input")) if hw else None
            cap = _num(hw / "power1_cap") if hw else None
            temp = _num(hw / "temp1_input") if hw else None
            out.append(
                GpuSample(
                    index=i,
                    uuid=f"amd-{card}",
                    name=f"AMD GPU ({card})",
                    vendor="amd",
                    util=_num(dev / "gpu_busy_percent"),
                    mem_util=_num(dev / "mem_busy_percent"),
                    mem_used_mib=used / 1048576 if used is not None else None,
                    mem_total_mib=total / 1048576 if total is not None else None,
                    temp_c=temp / 1000 if temp is not None else None,  # millidegrees
                    power_w=power / 1e6 if power is not None else None,  # microwatts
                    power_limit_w=cap / 1e6 if cap is not None else None,
                )
            )
        return tuple(out)
