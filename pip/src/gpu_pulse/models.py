"""Data records shared by collectors, history and renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

InsightLevel = Literal["good", "warning", "serious", "critical", "info"]


@dataclass(frozen=True)
class GpuSample:
    """One GPU reading. Missing values are ``None``."""

    index: int
    uuid: str
    name: str
    vendor: Literal["nvidia", "amd"]
    util: Optional[float] = None  # %
    mem_util: Optional[float] = None  # % memory-controller busy
    mem_used_mib: Optional[float] = None
    mem_total_mib: Optional[float] = None
    temp_c: Optional[float] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    clock_mhz: Optional[float] = None
    clock_max_mhz: Optional[float] = None
    mem_clock_mhz: Optional[float] = None
    fan_pct: Optional[float] = None
    pstate: Optional[str] = None
    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None
    # Active clock-throttle reasons from NVML, human-readable (e.g. "power cap", "thermal").
    throttle: tuple[str, ...] = ()
    # Live extras (None when the GPU/driver doesn't report them).
    enc_util: Optional[float] = None  # video encoder %
    dec_util: Optional[float] = None  # video decoder %
    pcie_tx_bps: Optional[float] = None  # bytes/s GPU -> host
    pcie_rx_bps: Optional[float] = None  # bytes/s host -> GPU
    energy_j: Optional[float] = None  # total energy since driver load, joules
    # "True usage" from NVML GPU Performance Monitoring (GPM), % over the last sample interval.
    # None when GPM is unsupported. Unlike `util`, these measure how much of the GPU is busy.
    sm_active: Optional[float] = None  # SM activity
    sm_occupancy: Optional[float] = None  # resident warps vs maximum
    tensor_active: Optional[float] = None  # any Tensor Core pipe
    dram_bw_util: Optional[float] = None  # DRAM bandwidth used vs peak
    fp32_active: Optional[float] = None
    fp16_active: Optional[float] = None
    fp64_active: Optional[float] = None
    int_active: Optional[float] = None
    # Wall-clock time (epoch s) the energy counter was read; it is cached for a few seconds, so it can be
    # older than the sample. None = read with the sample (or no counter).
    energy_t: Optional[float] = None

    @property
    def vram_pct(self) -> Optional[float]:
        if self.mem_used_mib is None or not self.mem_total_mib:
            return None
        return self.mem_used_mib / self.mem_total_mib * 100


@dataclass(frozen=True)
class GpuDetails:
    """Static facts about one GPU, read once. Derived figures are theoretical peaks."""

    index: int
    uuid: str
    name: str
    architecture: Optional[str] = None  # e.g. "Blackwell"
    compute_capability: Optional[str] = None  # e.g. "12.0"
    cuda_cores: Optional[int] = None
    sm_count: Optional[int] = None
    tensor_core_gen: Optional[str] = None  # e.g. "5th gen"
    memory_bus_bits: Optional[int] = None
    memory_bandwidth_gbs: Optional[float] = None  # derived: max mem clock × 2 × bus/8
    fp32_tflops: Optional[float] = None  # derived: cores × 2 × max SM clock
    max_sm_clock_mhz: Optional[float] = None
    max_mem_clock_mhz: Optional[float] = None
    vram_total_mib: Optional[float] = None
    bar1_total_mib: Optional[float] = None
    pcie_max_gen: Optional[int] = None
    pcie_max_width: Optional[int] = None
    power_default_w: Optional[float] = None
    power_min_w: Optional[float] = None
    power_max_w: Optional[float] = None
    num_fans: Optional[int] = None
    vbios: Optional[str] = None
    driver: Optional[str] = None
    cuda_driver: Optional[str] = None
    persistence_mode: Optional[bool] = None
    compute_mode: Optional[str] = None  # "default", "exclusive process", "prohibited"
    ecc_enabled: Optional[bool] = None
    mig_enabled: Optional[bool] = None
    display_active: Optional[bool] = None
    board_part: Optional[str] = None
    gpm_supported: Optional[bool] = None  # True-usage metrics (SM/Tensor/DRAM activity) available


@dataclass(frozen=True)
class Sample:
    t: float  # epoch seconds
    gpus: tuple[GpuSample, ...]
    cpu_util: float
    cpu_per_core: tuple[float, ...]
    mem_used: int
    mem_total: int
    # Whole-system extras (None = not available on this OS / first sample has no rate yet).
    swap_pct: Optional[float] = None
    cpu_freq_mhz: Optional[float] = None
    cpu_temp_c: Optional[float] = None
    disk_read_bps: Optional[float] = None  # bytes/s, all disks
    disk_write_bps: Optional[float] = None
    net_rx_bps: Optional[float] = None  # bytes/s, all non-loopback interfaces
    net_tx_bps: Optional[float] = None

    @property
    def mem_pct(self) -> float:
        return self.mem_used / self.mem_total * 100 if self.mem_total else 0.0


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    name: str
    used_mib: Optional[float]
    gpu_uuid: str


@dataclass(frozen=True)
class WindowStats:
    label: str
    minutes: int
    covered_minutes: int
    util: Optional[float] = None
    util_peak: Optional[float] = None
    vram: Optional[float] = None
    power: Optional[float] = None
    temp_peak: Optional[float] = None
    idle_share: Optional[float] = None
    busy_share: Optional[float] = None
    held_idle_share: Optional[float] = None
    cpu: Optional[float] = None
    mem: Optional[float] = None
    sm: Optional[float] = None  # mean SM activity % (GPM "true usage"); None when never recorded
    tensor: Optional[float] = None  # mean Tensor Core activity %
    sm_coverage: Optional[float] = None  # share (0..1) of GPU samples in the window that carried SM data


@dataclass(frozen=True)
class Insight:
    level: InsightLevel
    title: str
    detail: str


@dataclass(frozen=True)
class Specs:
    hostname: str
    os: str
    arch: str
    uptime_s: float
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    cpu_max_ghz: Optional[float]
    mem_total: int
    gpus: tuple[str, ...] = ()
    driver: Optional[str] = None
    cuda: Optional[str] = None
    disks: tuple[str, ...] = field(default_factory=tuple)
