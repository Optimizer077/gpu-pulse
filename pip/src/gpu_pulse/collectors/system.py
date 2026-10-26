"""CPU, RAM and static machine specs via psutil and the standard library."""

from __future__ import annotations

import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import psutil

from ..models import Specs

_CPU_SENSORS = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "cpu-thermal", "acpitz")
_CPU_LABELS = ("package id 0", "tctl", "tdie")


def _is_loopback(nic: str) -> bool:
    n = nic.lower()
    return n == "lo" or n.startswith("lo:") or "loopback" in n


@dataclass(frozen=True)
class SystemExtras:
    """Whole-system readings for one sample; ``None`` = unavailable (or no previous counter yet)."""

    swap_pct: Optional[float] = None
    cpu_freq_mhz: Optional[float] = None
    cpu_temp_c: Optional[float] = None
    disk_read_bps: Optional[float] = None
    disk_write_bps: Optional[float] = None
    net_rx_bps: Optional[float] = None
    net_tx_bps: Optional[float] = None


class SystemCollector:
    def __init__(self) -> None:
        psutil.cpu_percent(percpu=True)  # prime the delta counters
        # Previous cumulative counters: (monotonic time, read, write) and (time, rx, tx).
        self._disk: Optional[tuple[float, int, int]] = None
        self._net_nics: dict[str, tuple[int, int]] = {}  # per-NIC (rx, tx) byte counters at _net_t
        self._net_t: Optional[float] = None
        self._net_rate: tuple[Optional[float], Optional[float]] = (None, None)
        self._sensors = getattr(psutil, "sensors_temperatures", None)  # Linux/FreeBSD only

    def cpu(self) -> tuple[float, tuple[float, ...]]:
        per_core = tuple(psutil.cpu_percent(percpu=True))
        return (sum(per_core) / len(per_core) if per_core else 0.0), per_core

    def mem(self) -> tuple[int, int]:
        vm = psutil.virtual_memory()
        return vm.total - vm.available, vm.total

    def extras(self, now: Optional[float] = None) -> SystemExtras:
        """Swap, CPU clock/temperature and disk/network rates. Never raises.

        Rates are deltas of cumulative counters between calls (bytes/s); the first call, a counter
        reset/wrap, or a non-positive time step yields ``None``.
        """
        t = time.monotonic() if now is None else now
        disk_r, disk_w = self._disk_rates(t)
        net_rx, net_tx = self._net_rates(t)
        return SystemExtras(
            swap_pct=self._swap(),
            cpu_freq_mhz=self._freq(),
            cpu_temp_c=self._cpu_temp(),
            disk_read_bps=disk_r,
            disk_write_bps=disk_w,
            net_rx_bps=net_rx,
            net_tx_bps=net_tx,
        )

    @staticmethod
    def _swap() -> Optional[float]:
        try:
            sw = psutil.swap_memory()
            return float(sw.percent) if sw.total else None
        except (OSError, RuntimeError, AttributeError):
            return None

    @staticmethod
    def _freq() -> Optional[float]:
        try:
            f = psutil.cpu_freq()  # on Windows this is often the static base clock
            return float(f.current) if f and f.current else None
        except (OSError, RuntimeError, AttributeError, NotImplementedError):
            return None

    def _cpu_temp(self) -> Optional[float]:
        if self._sensors is None:
            return None
        try:
            temps = self._sensors()
        except (OSError, RuntimeError, AttributeError):
            return None
        for chip in _CPU_SENSORS:
            entries = temps.get(chip) or []
            for e in entries:
                if (e.label or "").lower() in _CPU_LABELS and e.current:
                    return float(e.current)
            vals = [e.current for e in entries if e.current]
            if vals:
                return float(max(vals))
        return None

    @staticmethod
    def _delta(
        prev: Optional[tuple[float, int, int]], cur: tuple[float, int, int]
    ) -> tuple[Optional[float], Optional[float]]:
        if prev is None:
            return None, None
        dt = cur[0] - prev[0]
        a, b = cur[1] - prev[1], cur[2] - prev[2]
        if dt <= 0 or a < 0 or b < 0:  # clock step or counter reset/wrap
            return None, None
        return a / dt, b / dt

    def _disk_rates(self, t: float) -> tuple[Optional[float], Optional[float]]:
        try:
            io = psutil.disk_io_counters(perdisk=False)
        except (OSError, RuntimeError, ValueError):
            io = None
        if io is None:  # no disks visible (some containers)
            self._disk = None
            return None, None
        cur = (t, int(io.read_bytes), int(io.write_bytes))
        out = self._delta(self._disk, cur)
        self._disk = cur
        return out

    NET_EVERY_S = 2.0  # net_io_counters(pernic=True) costs ~11 ms on Windows; reuse the rate in between

    def _net_rates(self, t: float) -> tuple[Optional[float], Optional[float]]:
        """Receive/send bytes/s summed over non-loopback NICs, from per-NIC deltas.

        Per-NIC deltas mean an adapter that appears with existing counters (VPN connect,
        re-enumeration) or resets its counters contributes nothing instead of a fake spike.
        """
        last_t = self._net_t
        if last_t is not None and 0 <= t - last_t < self.NET_EVERY_S:
            return self._net_rate
        try:
            nics = psutil.net_io_counters(pernic=True)
        except (OSError, RuntimeError, ValueError):
            nics = {}
        cur = {n: (int(c.bytes_recv), int(c.bytes_sent)) for n, c in (nics or {}).items() if not _is_loopback(n)}
        prev = self._net_nics
        rate: tuple[Optional[float], Optional[float]] = (None, None)
        dt = t - last_t if last_t is not None else 0.0
        if cur and prev and dt > 0:
            rx = tx = 0
            for name, (r, s) in cur.items():
                p = prev.get(name)
                if p is not None and r >= p[0] and s >= p[1]:
                    rx += r - p[0]
                    tx += s - p[1]
            rate = (rx / dt, tx / dt)
        self._net_nics, self._net_t, self._net_rate = cur, t, rate
        return rate


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if platform.system() == "Darwin":  # platform.processor() is just "arm" / "i386" on macOS
        out = _sysctl("machdep.cpu.brand_string")
        if out:
            return out
    return platform.processor() or "Unknown CPU"


def _sysctl(name: str) -> Optional[str]:
    """``sysctl -n <name>`` on macOS (no shell, 2 s timeout); None when unavailable."""
    try:
        r = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name], stdin=subprocess.DEVNULL, capture_output=True, timeout=2, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = r.stdout.decode("utf-8", "replace").strip() if r.returncode == 0 else ""
    return text or None


def _os_name() -> str:
    if platform.system() == "Linux":
        try:
            info = platform.freedesktop_os_release()
            return info.get("PRETTY_NAME", "Linux")
        except (OSError, AttributeError):
            return f"Linux {platform.release()}"
    if platform.system() == "Windows":
        ver = platform.version()
        major = "11" if int(ver.split(".")[-1] or 0) >= 22000 else platform.release()
        return f"Windows {major} ({ver})"
    if platform.system() == "Darwin":
        mac = platform.mac_ver()[0]
        if mac:
            return f"macOS {mac}"
    return f"{platform.system()} {platform.release()}"


def _disks() -> tuple[str, ...]:
    seen, out = set(), []
    for p in psutil.disk_partitions(all=False):
        if p.fstype in ("", "squashfs", "tmpfs", "overlay") or p.device in seen:
            continue
        seen.add(p.device)
        try:
            u = psutil.disk_usage(p.mountpoint)
        except OSError:
            continue
        out.append(f"{p.mountpoint}  {u.used / 1e9:.0f} / {u.total / 1e9:.0f} GB ({p.fstype})")
    return tuple(out[:6])


def collect_specs(gpu_names: tuple[str, ...], driver: Optional[str], cuda: Optional[str]) -> Specs:
    try:
        freq = psutil.cpu_freq()  # not implemented / raising on some Apple Silicon psutil builds
    except (OSError, RuntimeError, AttributeError, NotImplementedError):
        freq = None
    max_ghz = round(freq.max / 1000, 2) if freq and freq.max else None
    return Specs(
        hostname=socket.gethostname(),
        os=_os_name(),
        arch=platform.machine(),
        uptime_s=time.time() - psutil.boot_time(),
        cpu_model=_cpu_model(),
        cpu_cores=psutil.cpu_count(logical=False) or 0,
        cpu_threads=psutil.cpu_count(logical=True) or 0,
        cpu_max_ghz=max_ghz,
        mem_total=psutil.virtual_memory().total,
        gpus=gpu_names,
        driver=driver,
        cuda=cuda,
        disks=_disks(),
    )
