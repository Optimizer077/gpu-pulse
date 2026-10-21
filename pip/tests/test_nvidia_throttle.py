"""NVIDIA collector with a fake pynvml: throttle reasons, live extras, static details and GPM."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

from gpu_pulse.collectors import nvidia
from gpu_pulse.collectors.nvidia import NvidiaCollector, decode_throttle


class NVMLError(Exception):
    pass


def _unsupported(*_a: Any) -> Any:
    raise NVMLError("Not Supported")


class FakeGpmMetricsGet:
    def __init__(self) -> None:
        self.version = 0
        self.numMetrics = 0
        self.sample1: Any = None
        self.sample2: Any = None
        self.metrics = [SimpleNamespace(metricId=0, nvmlReturn=0, value=0.0) for _ in range(16)]


class FakeNV:
    """Minimal pynvml stand-in: anything not defined behaves like an unsupported NVML call."""

    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_GRAPHICS = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_PCIE_UTIL_TX_BYTES = 0
    NVML_PCIE_UTIL_RX_BYTES = 1
    NVML_DEVICE_ARCH_AMPERE = 7
    NVML_DEVICE_ARCH_ADA = 8
    NVML_DEVICE_ARCH_BLACKWELL = 10
    NVML_GPM_METRICS_GET_VERSION = 1
    NVML_GPM_METRIC_SM_UTIL = 1
    NVML_GPM_METRIC_SM_OCCUPANCY = 2
    NVML_GPM_METRIC_ANY_TENSOR_UTIL = 3
    NVML_GPM_METRIC_DRAM_BW_UTIL = 10
    NVML_GPM_METRIC_FP32_UTIL = 12
    NVML_GPM_METRIC_FP16_UTIL = 13
    NVML_GPM_METRIC_FP64_UTIL = 11
    NVML_GPM_METRIC_INTEGER_UTIL = 14
    nvmlClocksEventReasonSwPowerCap = 0x4
    nvmlClocksEventReasonGpuIdle = 0x1
    c_nvmlGpmMetricsGet_t = FakeGpmMetricsGet

    def __init__(self, missing: tuple[str, ...] = (), **funcs: Callable[..., Any]) -> None:
        self._missing = set(missing)
        self.calls: list[str] = []
        self.freed: list[Any] = []
        self.gpm_pairs: list[tuple[Any, Any]] = []
        self._funcs = funcs

    def __getattr__(self, name: str) -> Any:
        if name in self._missing or not name.startswith("nvml"):
            raise AttributeError(name)
        fn = self._funcs.get(name, _unsupported)

        def wrapper(*a: Any) -> Any:
            self.calls.append(name)
            return fn(*a)

        return wrapper


def base_funcs(**extra: Callable[..., Any]) -> dict[str, Callable[..., Any]]:
    clocks = {0: 3090, 1: 3090, 2: 14001}
    f: dict[str, Callable[..., Any]] = dict(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda i: f"h{i}",
        nvmlDeviceGetUUID=lambda h: "GPU-fake",
        nvmlDeviceGetName=lambda h: b"Fake RTX",
        nvmlDeviceGetMaxClockInfo=lambda h, kind: clocks[kind],
        nvmlDeviceGetUtilizationRates=lambda h: SimpleNamespace(gpu=80, memory=30),
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(used=4 * 2**30, total=16 * 2**30),
    )
    f.update(extra)
    return f


def make(monkeypatch: pytest.MonkeyPatch, missing: tuple[str, ...] = (), **funcs: Callable[..., Any]) -> tuple:
    fake = FakeNV(missing, **base_funcs(**funcs))
    monkeypatch.setattr(nvidia, "nv", fake)
    col = NvidiaCollector(pcie_poll_s=0)  # no background thread in tests
    assert col.detect()
    return col, fake


# ---- throttle reasons ----
def test_decode_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia, "nv", None)  # ABI fallback values
    assert decode_throttle(0x4 | 0x1) == ("power cap",)
    assert decode_throttle(0x1) == ()  # idle is not a problem
    assert decode_throttle(0) == () and decode_throttle(None) == ()
    every = decode_throttle(0x1FF)
    assert every == (
        "power cap",
        "thermal",
        "hw slowdown",
        "hw thermal",
        "hw power brake",
        "sync boost",
        "app clocks",
        "display clocks",
    )


def test_event_reasons_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    col, fake = make(
        monkeypatch,
        nvmlDeviceGetCurrentClocksEventReasons=lambda h: 0x4 | 0x20,
        nvmlDeviceGetCurrentClocksThrottleReasons=lambda h: 0x8,
    )
    assert col.sample()[0].throttle == ("power cap", "thermal")
    assert "nvmlDeviceGetCurrentClocksThrottleReasons" not in fake.calls


@pytest.mark.parametrize("missing", [True, False])
def test_falls_back_to_throttle_reasons(monkeypatch: pytest.MonkeyPatch, missing: bool) -> None:
    kw: dict[str, Any] = {"nvmlDeviceGetCurrentClocksThrottleReasons": lambda h: 0x40}
    names = ("nvmlDeviceGetCurrentClocksEventReasons",) if missing else ()
    col, _ = make(monkeypatch, names, **kw)  # event call either absent or raising NVMLError
    assert col.sample()[0].throttle == ("hw thermal",)


def test_no_reason_support_gives_empty_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    col, _ = make(monkeypatch)
    s = col.sample()[0]
    assert s.throttle == () and s.util == 80.0 and s.temp_c is None


# ---- live extras ----
def test_live_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    col, _ = make(
        monkeypatch,
        nvmlDeviceGetEncoderUtilization=lambda h: [12, 100000],
        nvmlDeviceGetDecoderUtilization=lambda h: [3, 100000],
        nvmlDeviceGetTotalEnergyConsumption=lambda h: 1_500_000,  # mJ
        nvmlDeviceGetPcieThroughput=lambda h, kind: 2048 if kind == 0 else 4096,  # KB/s
    )
    before = col.sample()[0]
    assert before.pcie_rx_bps is None  # nothing polled yet: never blocks the sample path
    col.poll_pcie_once()
    s = col.sample()[0]
    assert (s.enc_util, s.dec_util) == (12.0, 3.0)
    assert s.energy_j == 1500.0
    assert s.pcie_tx_bps == 2048 * 1024 and s.pcie_rx_bps == 4096 * 1024


def test_pcie_poller_thread_starts_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeNV(**base_funcs(nvmlDeviceGetPcieThroughput=lambda h, kind: 1000))
    monkeypatch.setattr(nvidia, "nv", fake)
    col = NvidiaCollector(pcie_poll_s=0.01)
    col.detect()
    col.sample()
    deadline = time.monotonic() + 2
    while col.sample()[0].pcie_rx_bps is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert col.sample()[0].pcie_rx_bps == 1000 * 1024
    col.close()
    assert col._pcie_thread is None


# ---- static details ----
def detail_funcs(**extra: Callable[..., Any]) -> dict[str, Callable[..., Any]]:
    f: dict[str, Callable[..., Any]] = dict(
        nvmlDeviceGetArchitecture=lambda h: 10,
        nvmlDeviceGetCudaComputeCapability=lambda h: (12, 0),
        nvmlDeviceGetNumGpuCores=lambda h: 4608,
        nvmlDeviceGetMemoryBusWidth=lambda h: 128,
        nvmlDeviceGetBAR1MemoryInfo=lambda h: SimpleNamespace(bar1Total=16 * 2**30),
        nvmlDeviceGetGpuMaxPcieLinkGeneration=lambda h: 5,
        nvmlDeviceGetMaxPcieLinkGeneration=lambda h: 4,
        nvmlDeviceGetMaxPcieLinkWidth=lambda h: 16,
        nvmlDeviceGetPowerManagementLimitConstraints=lambda h: [150000, 180000],
        nvmlDeviceGetPowerManagementDefaultLimit=lambda h: 180000,
        nvmlDeviceGetNumFans=lambda h: 1,
        nvmlDeviceGetVbiosVersion=lambda h: "98.06.1f.00.f0",
        nvmlDeviceGetBoardPartNumber=lambda h: "",
        nvmlDeviceGetComputeMode=lambda h: 0,
        nvmlDeviceGetDisplayActive=lambda h: 0,
        nvmlDeviceGetMigMode=lambda h: [0, 0],
        nvmlSystemGetDriverVersion=lambda: "591.74",
        nvmlSystemGetCudaDriverVersion=lambda: 13010,
    )
    f.update(extra)
    return f


def test_details(monkeypatch: pytest.MonkeyPatch) -> None:
    col, fake = make(monkeypatch, **detail_funcs())
    d = col.details()[0]
    assert (d.architecture, d.compute_capability, d.tensor_core_gen) == ("Blackwell", "12.0", "5th gen")
    assert (d.cuda_cores, d.sm_count, d.memory_bus_bits) == (4608, 36, 128)
    assert d.memory_bandwidth_gbs == 448.0  # 14001 MHz × 2 × 128 bit / 8
    assert d.fp32_tflops == 28.5  # 4608 × 2 × 3090 MHz
    assert (d.pcie_max_gen, d.pcie_max_width) == (5, 16)  # the GPU's own capability
    assert (d.power_min_w, d.power_max_w, d.power_default_w) == (150.0, 180.0, 180.0)
    assert d.vram_total_mib == 16384.0 and d.bar1_total_mib == 16384.0
    assert (d.driver, d.cuda_driver, d.vbios) == ("591.74", "13.1", "98.06.1f.00.f0")
    assert d.board_part is None  # empty string -> None
    assert d.persistence_mode is None and d.ecc_enabled is None  # unsupported
    assert d.mig_enabled is False and d.display_active is False and d.compute_mode == "default"
    assert d.gpm_supported is False
    n = len(fake.calls)
    assert col.details() is col.details() and len(fake.calls) == n  # cached
    assert not any("Serial" in c for c in fake.calls)  # privacy


def test_details_unknown_values_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    col, _ = make(
        monkeypatch,
        missing=("nvmlDeviceGetGpuMaxPcieLinkGeneration",),
        **detail_funcs(
            nvmlDeviceGetArchitecture=lambda h: 99,
            nvmlDeviceGetCudaComputeCapability=lambda h: (8, 0),  # A100: 64 cores/SM, not in the table
        ),
    )
    d = col.details()[0]
    assert d.architecture is None and d.tensor_core_gen is None
    assert d.sm_count is None and d.cuda_cores == 4608
    assert d.pcie_max_gen == 4  # falls back to the device+host maximum


def test_details_on_bare_device(monkeypatch: pytest.MonkeyPatch) -> None:
    col, _ = make(monkeypatch)
    d = col.details()[0]
    assert d.name == "Fake RTX" and d.architecture is None and d.fp32_tflops is None


# ---- GPM ("true usage") ----
class Gpm:
    """Fake GPM backend recording sample buffers and returning per-metric values."""

    def __init__(self, supported: int = 1, bad_metric: Optional[int] = None, fail_get: bool = False) -> None:
        self.supported = supported
        self.bad_metric = bad_metric
        self.fail_get = fail_get
        self.allocated: list[object] = []
        self.pairs: list[tuple[object, object]] = []
        self.freed: list[object] = []

    def funcs(self) -> dict[str, Callable[..., Any]]:
        def alloc() -> object:
            s = object()
            self.allocated.append(s)
            return s

        def get(h: Any, s: object) -> object:
            if self.fail_get:
                raise NVMLError("GPU lost")
            return s

        def metrics(mg: FakeGpmMetricsGet) -> FakeGpmMetricsGet:
            self.pairs.append((mg.sample1, mg.sample2))
            for m in mg.metrics[: mg.numMetrics]:
                m.value = float(m.metricId * 5)
                m.nvmlReturn = 3 if m.metricId == self.bad_metric else 0
            return mg

        return dict(
            nvmlGpmQueryDeviceSupport=lambda h: SimpleNamespace(isSupportedDevice=self.supported),
            nvmlGpmSampleAlloc=alloc,
            nvmlGpmSampleGet=get,
            nvmlGpmMetricsGet=metrics,
            nvmlGpmSampleFree=lambda s: self.freed.append(s),
        )


def test_gpm_ping_pong(monkeypatch: pytest.MonkeyPatch) -> None:
    gpm = Gpm(bad_metric=11)  # FP64 metric reports an error
    col, _ = make(monkeypatch, **gpm.funcs())
    first = col.sample()[0]
    assert first.sm_active is None  # no previous sample yet
    s = col.sample()[0]
    assert s.sm_active == 5.0 and s.sm_occupancy == 10.0 and s.tensor_active == 15.0
    assert s.dram_bw_util == 50.0 and s.fp32_active == 60.0 and s.int_active == 70.0
    assert s.fp64_active is None  # per-metric nvmlReturn != 0
    col.sample()
    a, b = gpm.allocated
    assert gpm.pairs == [(a, b), (b, a)]  # buffers alternate: metrics always cover the last interval
    assert col.details()[0].gpm_supported is True
    col.close()
    assert sorted(map(id, gpm.freed)) == sorted(map(id, gpm.allocated))


def test_gpm_unsupported_device(monkeypatch: pytest.MonkeyPatch) -> None:
    gpm = Gpm(supported=0)
    col, _ = make(monkeypatch, **gpm.funcs())
    col.sample()
    s = col.sample()[0]
    assert s.sm_active is None and gpm.allocated == []
    assert col.details()[0].gpm_supported is False


def test_gpm_errors_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    gpm = Gpm(fail_get=True)
    col, _ = make(monkeypatch, **gpm.funcs())
    col.sample()
    assert col.sample()[0].sm_active is None


def test_gpm_missing_from_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    col, _ = make(monkeypatch, missing=("nvmlGpmQueryDeviceSupport",))
    col.sample()
    assert col.sample()[0].tensor_active is None


# ---- real hardware (skipped without an NVIDIA GPU) ----
def test_real_gpu_smoke() -> None:
    if nvidia.nv is None:
        pytest.skip("nvidia-ml-py not installed")
    col = NvidiaCollector()
    if not col.detect():
        pytest.skip("no NVIDIA GPU")
    try:
        col.sample()
        s = col.sample()[0]
        assert isinstance(s.throttle, tuple)
        d = col.details()[0]
        print(f"\n{d}\nthrottle={s.throttle} sm_active={s.sm_active} tensor={s.tensor_active}")
    finally:
        col.close()
