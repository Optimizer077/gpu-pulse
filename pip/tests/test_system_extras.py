"""Whole-system extras: counter-delta rates, swap/clock/temperature, and Monitor wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gpu_pulse.collectors import system
from gpu_pulse.collectors.system import SystemCollector, SystemExtras
from gpu_pulse.monitor import Monitor, MonitorConfig


class FakePsutil:
    """Mutable counters; attributes are looked up by SystemCollector through the module global."""

    def __init__(self) -> None:
        self.disk = SimpleNamespace(read_bytes=1_000_000, write_bytes=500_000)
        self.nics: dict[str, Any] = {
            "Ethernet": SimpleNamespace(bytes_recv=10_000, bytes_sent=2_000),
            "Loopback Pseudo-Interface 1": SimpleNamespace(bytes_recv=9e12, bytes_sent=9e12),
            "lo": SimpleNamespace(bytes_recv=9e12, bytes_sent=9e12),
        }
        self.swap = SimpleNamespace(total=8 * 2**30, percent=12.5)
        self.freq: Any = SimpleNamespace(current=4200.0)

    def cpu_percent(self, percpu: bool = False) -> list[float]:
        return [10.0, 30.0]

    def disk_io_counters(self, perdisk: bool = False) -> Any:
        return self.disk

    def net_io_counters(self, pernic: bool = False) -> Any:
        return self.nics

    def swap_memory(self) -> Any:
        return self.swap

    def cpu_freq(self) -> Any:
        return self.freq


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakePsutil:
    f = FakePsutil()
    monkeypatch.setattr(system, "psutil", f)
    return f


def test_first_sample_has_no_rates(fake: FakePsutil) -> None:
    x = SystemCollector().extras(now=100.0)
    assert (x.disk_read_bps, x.disk_write_bps, x.net_rx_bps, x.net_tx_bps) == (None, None, None, None)
    assert x.swap_pct == 12.5 and x.cpu_freq_mhz == 4200.0
    assert x.cpu_temp_c is None  # no sensors_temperatures (as on Windows)


def test_rates_from_deltas_exclude_loopback(fake: FakePsutil) -> None:
    c = SystemCollector()
    c.extras(now=100.0)
    fake.disk = SimpleNamespace(read_bytes=1_000_000 + 4_000_000, write_bytes=500_000 + 1_000_000)
    fake.nics["Ethernet"] = SimpleNamespace(bytes_recv=10_000 + 2_000_000, bytes_sent=2_000 + 200_000)
    x = c.extras(now=102.0)
    assert x.disk_read_bps == 2_000_000 and x.disk_write_bps == 500_000
    assert x.net_rx_bps == 1_000_000 and x.net_tx_bps == 100_000


def test_counter_reset_and_clock_step_yield_none(fake: FakePsutil) -> None:
    c = SystemCollector()
    c.extras(now=100.0)
    fake.disk = SimpleNamespace(read_bytes=10, write_bytes=10)  # counter reset
    assert c.extras(now=101.0).disk_read_bps is None
    assert c.extras(now=101.0).disk_read_bps is None  # dt == 0
    fake.disk = SimpleNamespace(read_bytes=1010, write_bytes=10)
    assert c.extras(now=102.0).disk_read_bps == 1000.0  # recovers on the next sample


def test_missing_counters_and_errors(fake: FakePsutil, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("denied")

    monkeypatch.setattr(fake, "disk_io_counters", lambda perdisk=False: None)  # containers
    monkeypatch.setattr(fake, "net_io_counters", boom)
    monkeypatch.setattr(fake, "swap_memory", boom)
    fake.freq = None
    c = SystemCollector()
    c.extras(now=1.0)
    x = c.extras(now=2.0)
    assert x == SystemExtras()  # everything None, nothing raised


def test_no_swap_configured(fake: FakePsutil) -> None:
    fake.swap = SimpleNamespace(total=0, percent=0.0)
    assert SystemCollector().extras(now=1.0).swap_pct is None


def test_cpu_temperature_prefers_package_sensor(fake: FakePsutil) -> None:
    c = SystemCollector()
    entry = SimpleNamespace
    c._sensors = lambda: {
        "acpitz": [entry(label="", current=40.0)],
        "coretemp": [entry(label="Core 0", current=61.0), entry(label="Package id 0", current=66.0)],
    }
    assert c.extras(now=1.0).cpu_temp_c == 66.0
    c._sensors = lambda: {"k10temp": [entry(label="Tccd1", current=58.0), entry(label="Tccd2", current=63.0)]}
    assert c.extras(now=2.0).cpu_temp_c == 63.0
    c._sensors = lambda: {"nvme": [entry(label="Composite", current=45.0)]}
    assert c.extras(now=3.0).cpu_temp_c is None  # not a CPU sensor


def test_real_psutil_never_raises() -> None:
    c = SystemCollector()
    c.extras()
    x = c.extras()
    assert x.disk_read_bps is None or x.disk_read_bps >= 0


def test_monitor_demo_fills_extras_and_details() -> None:
    m = Monitor(MonitorConfig(demo=True, persist=False))
    try:
        m.tick()
        s = m.tick()
        assert s.disk_read_bps is not None and s.net_rx_bps is not None and s.swap_pct is not None
        assert s.gpus[0].sm_active is not None
        det = m.gpu_details()
        assert det[0].name == "DEMO GPU (simulated)" and det[0].architecture == "Blackwell"
        assert m.gpu_details() is det  # cached
        d = m.diagnose(s.gpus[0].uuid)
        assert d.health and d.verdict
    finally:
        m.close()
