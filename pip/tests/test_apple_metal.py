"""Apple Silicon collection, unavailable counters, monitor recovery and the terminal UI."""

from __future__ import annotations

import asyncio
import plistlib
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
from rich.console import Console

from gpu_pulse.app import GpuDashboard
from gpu_pulse.collectors import AmdLinuxCollector, AppleMetalCollector, NvidiaCollector
from gpu_pulse.collectors.apple_metal import parse_apple_gpus
from gpu_pulse.history import HistoryStore
from gpu_pulse.models import Sample
from gpu_pulse.monitor import Monitor, MonitorConfig
from gpu_pulse.status import run_status
from gpu_pulse.widgets import RingGauge


def data(stats: dict | None = None, cls: str = "AGXAcceleratorG14X") -> bytes:
    return plistlib.dumps([{
        "IOClass": cls, "model": "Apple M2 Pro", "IORegistryEntryID": 4294968597,
        "PerformanceStatistics": stats if stats is not None else {
            "Device Utilization %": 42, "Renderer Utilization %": 12, "Alloc system memory": 2**31,
        },
    }])


def sample(t: float) -> Sample:
    return Sample(t=t, gpus=parse_apple_gpus(data()), cpu_util=20, cpu_per_core=(20,), mem_used=1, mem_total=4)


def test_live_fields_and_shared_memory() -> None:
    (g,) = parse_apple_gpus(data())
    assert (g.uuid, g.name, g.vendor, g.util, g.mem_used_mib) == ("apple-0", "Apple M2 Pro", "apple", 42, 2048)
    assert g.vram_pct is g.mem_total_mib is g.temp_c is g.power_w is g.clock_mhz is None


@pytest.mark.parametrize("bad", [-1, 101, float("nan"), float("inf"), True, "99"])
def test_invalid_values_and_renderer_fallback(bad: object) -> None:
    (g,) = parse_apple_gpus(data({"Device Utilization %": bad, "Renderer Utilization %": 33}))
    assert g.util == 33 and g.mem_used_mib is None


def test_zero_missing_and_non_apple_accelerators() -> None:
    assert parse_apple_gpus(data({"Device Utilization %": 0, "Renderer Utilization %": 80}))[0].util == 0
    assert parse_apple_gpus(data({}))[0].util is None
    assert parse_apple_gpus(data(cls="IntelAccelerator")) == ()
    assert parse_apple_gpus(plistlib.dumps([])) == ()
    with pytest.raises(ValueError):
        parse_apple_gpus(plistlib.dumps({}))


def test_detection_platform_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def read(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append((args, kwargs))
        return data()

    monkeypatch.setattr("gpu_pulse.collectors.apple_metal.subprocess.check_output", read)
    monkeypatch.setattr("gpu_pulse.collectors.apple_metal.sys.platform", "linux")
    c = AppleMetalCollector()
    assert not c.detect() and not calls
    monkeypatch.setattr("gpu_pulse.collectors.apple_metal.sys.platform", "darwin")
    assert c.detect() and c.sample()[0].util == 42
    assert calls[0][1]["timeout"] == 4

    def fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise subprocess.TimeoutExpired("ioreg", 4)

    monkeypatch.setattr("gpu_pulse.collectors.apple_metal.subprocess.check_output", fail)
    assert not c.detect()
    with pytest.raises(subprocess.TimeoutExpired):
        c.sample()


def test_history_round_trip_merge_missing_values_and_csv() -> None:
    now = int(time.time() // 60) * 60 + 1
    h = HistoryStore()
    h.add(sample(now))
    copy = HistoryStore()
    copy.load(h.snapshot())
    copy.add(sample(now + 0.1))  # merges an open minute with the restored minute
    stats = copy.stats("5 min", 5, "apple-0", now)
    assert stats.util == 42 and stats.vram is stats.power is stats.temp_peak is stats.held_idle_share is None
    assert copy.to_csv().splitlines()[1].split(",")[5:8] == ["", "", ""]
    assert copy.heatmap("apple-0", now=now, metric="power")[1][-1] is None
    s = sample(now + 0.2)
    copy.add(replace(s, gpus=(replace(s.gpus[0], power_w=12, temp_c=45, mem_total_mib=4096),)))
    stats = copy.stats("5 min", 5, "apple-0", now + 0.2)
    assert (stats.vram, stats.power, stats.temp_peak) == (50, 12, 45)


def monitor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Monitor:
    monkeypatch.setattr(NvidiaCollector, "detect", lambda self: False)
    monkeypatch.setattr(AmdLinuxCollector, "detect", lambda self: False)
    monkeypatch.setattr(AppleMetalCollector, "detect", lambda self: True)
    monkeypatch.setattr(AppleMetalCollector, "sample", staticmethod(lambda: parse_apple_gpus(data())))
    return Monitor(MonitorConfig(persist=False, history_path=tmp_path / "history.json"))


def test_monitor_records_recovers_and_renders_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    m = monitor(monkeypatch, tmp_path)
    try:
        assert m.vendor == "apple" and m.tick().gpus[0].util == 42

        def fail():
            raise OSError("unavailable")

        monkeypatch.setattr(m.apple, "sample", fail)
        assert m.tick().gpus == ()
        monkeypatch.setattr(m.apple, "sample", lambda: parse_apple_gpus(data()))
        console = Console(record=True, width=130, color_system=None)
        monkeypatch.setattr("gpu_pulse.status.time.sleep", lambda s: None)
        run_status(m, console)
        out = console.export_text()
        assert "Apple M2 Pro" in out and "2.0 GB shared" in out and "/0G" not in out
        assert not m.processes
    finally:
        m.close()


def test_dashboard_all_tabs_with_apple_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def run() -> None:
        m = monitor(monkeypatch, tmp_path)
        app = GpuDashboard(m, fps=0)
        try:
            async with app.run_test(size=(130, 40)) as pilot:
                await pilot.pause(0.3)
                assert "Apple M2 Pro" in str(app.query_one("#ring-gpu", RingGauge).border_title)
                assert str(app.query_one("#ring-vram", RingGauge).border_title) == "GPU MEMORY"
                for key, tab in (("2", "gpu"), ("3", "system"), ("4", "history"), ("5", "diagnosis")):
                    await pilot.press(key)
                    await pilot.pause(0.1)
                    assert app.tab == tab
        finally:
            m.close()

    asyncio.run(run())
