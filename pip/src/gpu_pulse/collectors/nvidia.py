"""NVIDIA GPUs through NVML (nvidia-ml-py). Works on Windows and Linux."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from ..models import GpuDetails, GpuProcess, GpuSample

logger = logging.getLogger(__name__)

try:
    import pynvml as nv
except ImportError:  # nvidia-ml-py missing
    nv = None

# Clock event ("throttle") reason bits. Values are part of the NVML ABI; the pynvml constant is
# preferred when present. The GPU-idle bit (0x1) is deliberately absent: idle is not a problem.
_REASON_BITS: tuple[tuple[str, int, str], ...] = (
    ("SwPowerCap", 0x4, "power cap"),
    ("SwThermalSlowdown", 0x20, "thermal"),
    ("HwSlowdown", 0x8, "hw slowdown"),
    ("HwThermalSlowdown", 0x40, "hw thermal"),
    ("HwPowerBrakeSlowdown", 0x80, "hw power brake"),
    ("SyncBoost", 0x10, "sync boost"),
    ("ApplicationsClocksSetting", 0x2, "app clocks"),
    ("DisplayClockSetting", 0x100, "display clocks"),
)
# Newer nvidia-ml-py names first; older drivers/bindings only have the "Throttle" variant.
_REASON_QUERIES = ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons")

_ARCH_NAMES = {
    "KEPLER": "Kepler",
    "MAXWELL": "Maxwell",
    "PASCAL": "Pascal",
    "VOLTA": "Volta",
    "TURING": "Turing",
    "AMPERE": "Ampere",
    "ADA": "Ada Lovelace",
    "HOPPER": "Hopper",
    "BLACKWELL": "Blackwell",
    "RUBIN": "Rubin",
}
_TENSOR_GEN = {
    "Volta": "1st gen",
    "Turing": "2nd gen",
    "Ampere": "3rd gen",
    "Ada Lovelace": "4th gen",
    "Hopper": "4th gen",
    "Blackwell": "5th gen",
}
# FP32 cores per SM, used to derive the SM count (NVML exposes cores, not SMs). Only the
# documented 128-core consumer SM layouts: Ampere GA10x (8.6), Ada (8.9), Blackwell GB20x (12.0).
_CORES_PER_SM = {(8, 6): 128, (8, 9): 128, (12, 0): 128}
_COMPUTE_MODES = {0: "default", 1: "exclusive thread", 2: "prohibited", 3: "exclusive process"}
PCIE_POLL_S = 2.0  # nvmlDeviceGetPcieThroughput blocks ~20-30 ms, so it runs off the sample path
PCIE_MAX_DUTY = 0.25  # with many GPUs one poll round grows: keep the poller busy at most 25% of the time
SLOW_FIELDS_S = 2.0  # fan speed and the energy counter (the slowest per-sample calls) refresh this often
# GPU Performance Monitoring metrics ("true usage"), NVML suffix -> GpuSample field.
_GPM_FIELDS: tuple[tuple[str, str], ...] = (
    ("SM_UTIL", "sm_active"),
    ("SM_OCCUPANCY", "sm_occupancy"),
    ("ANY_TENSOR_UTIL", "tensor_active"),
    ("DRAM_BW_UTIL", "dram_bw_util"),
    ("FP32_UTIL", "fp32_active"),
    ("FP16_UTIL", "fp16_active"),
    ("FP64_UTIL", "fp64_active"),
    ("INTEGER_UTIL", "int_active"),
)


def _ok(name: str, *args: Any) -> bool:
    """Run an NVML call for its side effect; False if missing or failed."""
    fn = getattr(nv, name, None) if nv is not None else None
    if fn is None:
        return False
    try:
        fn(*args)
        return True
    except Exception:  # noqa: BLE001 - NVMLError / ctypes errors mean "unavailable"
        return False


class _Gpm:
    """Two GPM sample buffers used ping-pong: each read reports metrics over the last interval."""

    def __init__(self, handle: Any, samples: tuple[Any, Any]) -> None:
        self.handle = handle
        self.samples = samples
        self.idx = 0
        self.have_prev = False
        self.ids: list[tuple[int, str]] = []
        for suffix, attr in _GPM_FIELDS:
            mid = getattr(nv, "NVML_GPM_METRIC_" + suffix, None)
            if isinstance(mid, int):
                self.ids.append((mid, attr))

    @classmethod
    def open(cls, handle: Any) -> Optional[_Gpm]:
        support = _qn("nvmlGpmQueryDeviceSupport", handle)
        if support is None or getattr(support, "isSupportedDevice", 0) != 1:
            return None
        a = _qn("nvmlGpmSampleAlloc")
        b = _qn("nvmlGpmSampleAlloc") if a is not None else None
        if a is None or b is None:
            if a is not None:
                _ok("nvmlGpmSampleFree", a)
            return None
        return cls(handle, (a, b))

    def read(self) -> dict[str, float]:
        cur = self.samples[self.idx]
        if not _ok("nvmlGpmSampleGet", self.handle, cur):
            self.have_prev = False
            return {}
        out: dict[str, float] = {}
        if self.have_prev and self.ids:
            try:
                mg = nv.c_nvmlGpmMetricsGet_t()
                mg.version = nv.NVML_GPM_METRICS_GET_VERSION
                mg.numMetrics = len(self.ids)
                mg.sample1 = self.samples[1 - self.idx]
                mg.sample2 = cur
                for i, (mid, _) in enumerate(self.ids):
                    mg.metrics[i].metricId = mid
                nv.nvmlGpmMetricsGet(mg)
                for i, (_, attr) in enumerate(self.ids):
                    m = mg.metrics[i]
                    v = float(m.value)
                    if int(m.nvmlReturn) == 0 and v == v and abs(v) != float("inf"):
                        out[attr] = min(100.0, max(0.0, v))
            except Exception as e:  # noqa: BLE001 - GPM failure only drops the true-usage fields
                logger.debug("GPM metrics query failed: %s", e)
                out = {}
        self.have_prev = True
        self.idx ^= 1
        return out

    def close(self) -> None:
        for s in self.samples:
            _ok("nvmlGpmSampleFree", s)


def _q(fn: Callable[..., Any], *args: Any) -> Any:
    """Call an NVML query; NVMLError (unsupported/unavailable) becomes ``None``."""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - every NVMLError subclass means "unavailable"
        return None


def _qn(name: str, *args: Any) -> Any:
    """Like ``_q`` for a function that may be missing from older nvidia-ml-py versions."""
    fn = getattr(nv, name, None) if nv is not None else None
    return None if fn is None else _q(fn, *args)


def _f(v: Any, scale: float = 1.0) -> Optional[float]:
    return None if v is None else float(v) / scale


def _i(v: Any) -> Optional[int]:
    return None if v is None else int(v)


def _text(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _opt_text(v: Any) -> Optional[str]:
    s = _text(v).strip() if v is not None else ""
    return s or None


def _flag(v: Any) -> Optional[bool]:
    """NVML enable-state (0/1, or a (current, pending) pair) -> bool."""
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    return None if v is None else bool(v)


def decode_throttle(mask: Optional[int]) -> tuple[str, ...]:
    """NVML clock-event bitmask -> short reasons, e.g. ``("power cap",)``; idle is ignored."""
    if not mask:
        return ()
    out = []
    for suffix, fallback, label in _REASON_BITS:
        bit = fallback
        for prefix in ("nvmlClocksEventReason", "nvmlClocksThrottleReason"):
            value = getattr(nv, prefix + suffix, None) if nv is not None else None
            if isinstance(value, int):
                bit = value
                break
        if mask & bit:
            out.append(label)
    return tuple(out)


def _throttle(h: Any) -> tuple[str, ...]:
    for name in _REASON_QUERIES:
        mask = _qn(name, h)
        if mask is not None:
            return decode_throttle(int(mask))
    return ()


def _arch_name(code: Any) -> Optional[str]:
    if code is None or nv is None:
        return None
    for key, label in _ARCH_NAMES.items():
        value = getattr(nv, "NVML_DEVICE_ARCH_" + key, None)
        if value is not None and value == code:
            return label
    return None


class NvidiaCollector:
    def __init__(self, pcie_poll_s: float = PCIE_POLL_S) -> None:
        self.available = False
        self._inited = False
        self._handles: list[Any] = []
        self._uuids: list[str] = []
        self._names: list[str] = []
        self._max_clock: list[Optional[float]] = []
        self._details: Optional[list[GpuDetails]] = None
        # Latest (tx, rx) bytes/s per GPU from the background poller.
        self._pcie: list[tuple[Optional[float], Optional[float]]] = []
        self._pcie_poll_s = pcie_poll_s
        self._pcie_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._gpm: list[Optional[_Gpm]] = []
        # GPU position -> (monotonic read time, fan %, energy J, wall-clock read time)
        self._slow_cache: dict[int, tuple[float, Optional[float], Optional[float], float]] = {}

    def detect(self) -> bool:
        if nv is None:
            return False
        try:
            nv.nvmlInit()
        except Exception as e:  # noqa: BLE001
            logger.debug("NVML unavailable: %s", e)
            return False
        self._inited = True
        count = _q(nv.nvmlDeviceGetCount) or 0
        # A lost GPU (e.g. Xid 79) or one hidden from a container must not take the others down.
        handles = [_q(nv.nvmlDeviceGetHandleByIndex, i) for i in range(count)]
        skipped = [i for i, h in enumerate(handles) if h is None]
        if skipped:
            logger.warning("Skipping unreadable NVIDIA GPU index(es): %s", skipped)
        self._handles = [h for h in handles if h is not None]
        # Static per-device facts, read once.
        self._uuids = [_text(_q(nv.nvmlDeviceGetUUID, h) or f"nvidia-{i}") for i, h in enumerate(self._handles)]
        self._names = [_text(_q(nv.nvmlDeviceGetName, h) or "NVIDIA GPU") for h in self._handles]
        self._max_clock = [_f(_q(nv.nvmlDeviceGetMaxClockInfo, h, nv.NVML_CLOCK_GRAPHICS)) for h in self._handles]
        self._pcie = [(None, None)] * len(self._handles)
        self._gpm = [_Gpm.open(h) for h in self._handles]
        self.available = bool(self._handles)
        return self.available

    # ---- PCIe throughput (slow call, polled in the background) ----
    def poll_pcie_once(self) -> None:
        """Refresh the cached PCIe throughput of every GPU (NVML reports KB/s)."""
        tx_kind = getattr(nv, "NVML_PCIE_UTIL_TX_BYTES", 0)
        rx_kind = getattr(nv, "NVML_PCIE_UTIL_RX_BYTES", 1)
        for i, h in enumerate(self._handles):
            if self._stop.is_set():  # close() requested: don't start another ~60 ms NVML call
                return
            tx = _qn("nvmlDeviceGetPcieThroughput", h, tx_kind)
            rx = _qn("nvmlDeviceGetPcieThroughput", h, rx_kind)
            self._pcie[i] = (_f(tx, 1 / 1024), _f(rx, 1 / 1024))

    def _pcie_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.poll_pcie_once()
            except Exception as e:  # noqa: BLE001 - a polling hiccup must not kill the thread silently
                logger.debug("PCIe throughput poll failed: %s", e)
            busy = time.monotonic() - t0  # ~60 ms per GPU: 8 GPUs take ~0.5 s per round
            self._stop.wait(max(self._pcie_poll_s, busy * (1 - PCIE_MAX_DUTY) / PCIE_MAX_DUTY))

    def _ensure_pcie_poller(self) -> None:
        if self._pcie_thread is None and self._handles and self._pcie_poll_s > 0:
            self._pcie_thread = threading.Thread(target=self._pcie_loop, name="gpu-pulse-pcie", daemon=True)
            self._pcie_thread.start()

    # ---- live samples ----
    def _read(self, i: int, h: Any) -> GpuSample:
        util = _q(nv.nvmlDeviceGetUtilizationRates, h)
        mem = _q(nv.nvmlDeviceGetMemoryInfo, h)
        pstate = _q(nv.nvmlDeviceGetPerformanceState, h)
        gen = _q(nv.nvmlDeviceGetCurrPcieLinkGeneration, h)
        width = _q(nv.nvmlDeviceGetCurrPcieLinkWidth, h)
        enc = _qn("nvmlDeviceGetEncoderUtilization", h)  # (util %, sampling period µs)
        dec = _qn("nvmlDeviceGetDecoderUtilization", h)
        tx, rx = self._pcie[i] if i < len(self._pcie) else (None, None)
        gpm = self._gpm[i] if i < len(self._gpm) else None
        true_usage = gpm.read() if gpm is not None else {}
        fan, energy, energy_t = self._slow_fields(i, h)
        return GpuSample(
            index=i,
            uuid=self._uuids[i],
            name=self._names[i],
            vendor="nvidia",
            util=_f(util.gpu) if util else None,
            mem_util=_f(util.memory) if util else None,
            mem_used_mib=_f(mem.used, 1048576) if mem else None,
            mem_total_mib=_f(mem.total, 1048576) if mem else None,
            temp_c=_f(_q(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU)),
            power_w=_f(_q(nv.nvmlDeviceGetPowerUsage, h), 1000),  # milliwatts
            power_limit_w=_f(_q(nv.nvmlDeviceGetEnforcedPowerLimit, h), 1000),
            clock_mhz=_f(_q(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_GRAPHICS)),
            clock_max_mhz=self._max_clock[i],
            mem_clock_mhz=_f(_q(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_MEM)),
            fan_pct=fan,
            pstate=f"P{pstate}" if pstate is not None else None,
            pcie_gen=int(gen) if gen is not None else None,
            pcie_width=int(width) if width is not None else None,
            throttle=_throttle(h),
            enc_util=_f(enc[0]) if enc else None,
            dec_util=_f(dec[0]) if dec else None,
            pcie_tx_bps=tx,
            pcie_rx_bps=rx,
            energy_j=energy,
            energy_t=energy_t if energy is not None else None,
            sm_active=true_usage.get("sm_active"),
            sm_occupancy=true_usage.get("sm_occupancy"),
            tensor_active=true_usage.get("tensor_active"),
            dram_bw_util=true_usage.get("dram_bw_util"),
            fp32_active=true_usage.get("fp32_active"),
            fp16_active=true_usage.get("fp16_active"),
            fp64_active=true_usage.get("fp64_active"),
            int_active=true_usage.get("int_active"),
        )

    def _slow_fields(self, i: int, h: Any) -> tuple[Optional[float], Optional[float], float]:
        """(fan %, energy J, wall time the energy was read): the two slowest per-GPU reads (~0.3 ms
        together), refreshed every 2 s. The read time travels with the cached energy so session
        energy isn't computed against the (later) sample time."""
        now = time.monotonic()
        cached = self._slow_cache.get(i)
        if cached is None or now - cached[0] >= SLOW_FIELDS_S:
            fan = _f(_q(nv.nvmlDeviceGetFanSpeed, h))
            energy = _f(_qn("nvmlDeviceGetTotalEnergyConsumption", h), 1000)  # millijoules
            cached = self._slow_cache[i] = (now, fan, energy, time.time())
        return cached[1], cached[2], cached[3]

    def sample(self) -> tuple[GpuSample, ...]:
        self._ensure_pcie_poller()
        return tuple(self._read(i, h) for i, h in enumerate(self._handles))

    # ---- static details ----
    def details(self) -> list[GpuDetails]:
        """Static per-GPU facts (architecture, cores, bus, limits); read once and cached."""
        if self._details is None:
            driver, cuda = self.driver_info()
            self._details = [self._detail(i, h, driver, cuda) for i, h in enumerate(self._handles)]
        return self._details

    def _detail(self, i: int, h: Any, driver: Optional[str], cuda: Optional[str]) -> GpuDetails:
        arch = _arch_name(_qn("nvmlDeviceGetArchitecture", h))
        cc = _qn("nvmlDeviceGetCudaComputeCapability", h)
        cc_t = (int(cc[0]), int(cc[1])) if isinstance(cc, (list, tuple)) and len(cc) == 2 else None
        cores = _i(_qn("nvmlDeviceGetNumGpuCores", h))
        per_sm = _CORES_PER_SM.get(cc_t) if cc_t else None
        sm_clock = _f(_qn("nvmlDeviceGetMaxClockInfo", h, getattr(nv, "NVML_CLOCK_SM", 1)))
        mem_clock = _f(_qn("nvmlDeviceGetMaxClockInfo", h, getattr(nv, "NVML_CLOCK_MEM", 2)))
        bus = _i(_qn("nvmlDeviceGetMemoryBusWidth", h))
        mem = _qn("nvmlDeviceGetMemoryInfo", h)
        bar1 = _qn("nvmlDeviceGetBAR1MemoryInfo", h)
        limits = _qn("nvmlDeviceGetPowerManagementLimitConstraints", h)
        mode = _qn("nvmlDeviceGetComputeMode", h)
        # Prefer the GPU's own capability; fall back to the device+host maximum.
        gen = _qn("nvmlDeviceGetGpuMaxPcieLinkGeneration", h)
        if gen is None:
            gen = _qn("nvmlDeviceGetMaxPcieLinkGeneration", h)
        lo, hi = (limits[0], limits[1]) if isinstance(limits, (list, tuple)) and len(limits) == 2 else (None, None)
        return GpuDetails(
            index=i,
            uuid=self._uuids[i],
            name=self._names[i],
            architecture=arch,
            compute_capability=f"{cc_t[0]}.{cc_t[1]}" if cc_t else None,
            cuda_cores=cores,
            sm_count=cores // per_sm if cores and per_sm else None,  # derived from cores / documented cores-per-SM
            tensor_core_gen=_TENSOR_GEN.get(arch) if arch else None,
            memory_bus_bits=bus,
            # Theoretical peaks: DDR (×2) at the max memory clock NVML reports; 2 FLOPs (FMA) per core per clock.
            memory_bandwidth_gbs=round(mem_clock * 2 * bus / 8 / 1000, 1) if mem_clock and bus else None,
            fp32_tflops=round(cores * 2 * sm_clock / 1e6, 1) if cores and sm_clock else None,
            max_sm_clock_mhz=sm_clock,
            max_mem_clock_mhz=mem_clock,
            vram_total_mib=_f(mem.total, 1048576) if mem else None,
            bar1_total_mib=_f(getattr(bar1, "bar1Total", None), 1048576) if bar1 else None,
            pcie_max_gen=_i(gen),
            pcie_max_width=_i(_qn("nvmlDeviceGetMaxPcieLinkWidth", h)),
            power_default_w=_f(_qn("nvmlDeviceGetPowerManagementDefaultLimit", h), 1000),
            power_min_w=_f(lo, 1000),
            power_max_w=_f(hi, 1000),
            num_fans=_i(_qn("nvmlDeviceGetNumFans", h)),
            vbios=_opt_text(_qn("nvmlDeviceGetVbiosVersion", h)),
            driver=driver,
            cuda_driver=cuda,
            persistence_mode=_flag(_qn("nvmlDeviceGetPersistenceMode", h)),
            compute_mode=_COMPUTE_MODES.get(int(mode)) if mode is not None else None,
            ecc_enabled=_flag(_qn("nvmlDeviceGetCurrentEccMode", h)),
            mig_enabled=_flag(_qn("nvmlDeviceGetMigMode", h)),
            display_active=_flag(_qn("nvmlDeviceGetDisplayActive", h)),
            board_part=_opt_text(_qn("nvmlDeviceGetBoardPartNumber", h)),
            gpm_supported=(self._gpm[i] is not None) if i < len(self._gpm) else None,
            # Serial number is deliberately not read (privacy).
        )

    def processes(self) -> list[GpuProcess]:
        out: list[GpuProcess] = []
        for i, h in enumerate(self._handles):
            for p in _q(nv.nvmlDeviceGetComputeRunningProcesses, h) or []:
                used = getattr(p, "usedGpuMemory", None)
                out.append(
                    GpuProcess(
                        pid=p.pid,
                        name=_process_name(p.pid),
                        # NVML reports None (or a huge sentinel) when Windows WDDM hides per-process memory.
                        used_mib=used / 1048576 if used is not None and used < 2**60 else None,
                        gpu_uuid=self._uuids[i],
                    )
                )
        return out

    def driver_info(self) -> tuple[Optional[str], Optional[str]]:
        if not self.available:
            return None, None
        driver = _q(nv.nvmlSystemGetDriverVersion)
        cuda = _q(nv.nvmlSystemGetCudaDriverVersion)
        cuda_s = f"{cuda // 1000}.{(cuda % 1000) // 10}" if cuda else None
        return (_text(driver) if driver else None), cuda_s

    def close(self) -> None:
        self._stop.set()
        poller_alive = False
        if self._pcie_thread is not None:
            self._pcie_thread.join(timeout=2.0)
            poller_alive = self._pcie_thread.is_alive()
            self._pcie_thread = None
        for gpm in self._gpm:
            if gpm is not None:
                gpm.close()
        self._gpm = []
        if self._inited and nv is not None:
            if poller_alive:
                # NVML must not be shut down under a thread that is mid-call; the process is
                # exiting anyway, so leaving NVML initialised is the safe choice.
                logger.warning("PCIe poller still busy at exit; skipping nvmlShutdown")
            else:
                _q(nv.nvmlShutdown)
            self._inited = False


def _process_name(pid: int) -> str:
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:  # noqa: BLE001 - access denied / exited
        return "protected process" if os.name == "nt" else f"pid {pid}"
