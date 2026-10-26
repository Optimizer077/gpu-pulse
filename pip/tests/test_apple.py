"""macOS / Apple GPU support, tested without a Mac: ioreg + system_profiler fixtures, an injected
command runner, the simulated Apple GPU, and null-safety of every rule that needs NVIDIA-only data.

The fixtures under tests/fixtures are reconstructions (see their headers); replace them with real
captures from ``scripts/mac_probe.py --save-fixtures`` when available.
"""

from __future__ import annotations

import asyncio
import math
import os
import plistlib
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import pytest
from rich.console import Console

from gpu_pulse import panels
from gpu_pulse.alerts import AlertConfig, AlertEngine
from gpu_pulse.app import GpuDashboard
from gpu_pulse.cli import _parser, main
from gpu_pulse.collectors import system as system_mod
from gpu_pulse.collectors.apple import (
    IOREG_ARGS,
    IOREG_TIMEOUT_S,
    MAX_MISSES,
    AppleCollector,
    apple_details,
    metal_label,
    parse_ioreg,
    parse_ioreg_plist,
    parse_profiler_json,
    run_command,
)
from gpu_pulse.fleet import analyze_fleet, gpu_flag
from gpu_pulse.fleetview import gpu_row
from gpu_pulse.history import HistoryStore
from gpu_pulse.insights import build_insights
from gpu_pulse.intel import diagnose
from gpu_pulse.models import GpuSample, Sample
from gpu_pulse.monitor import Monitor, MonitorConfig
from gpu_pulse.status import run_status
from gpu_pulse.widgets import RingGauge

FIX = Path(__file__).parent / "fixtures"
UNIFIED = 32_768.0  # MiB: a 32 GB Mac
NVIDIA_ONLY = (
    "temp_c",
    "power_w",
    "power_limit_w",
    "clock_mhz",
    "clock_max_mhz",
    "mem_clock_mhz",
    "fan_pct",
    "pstate",
    "pcie_gen",
    "pcie_width",
    "enc_util",
    "dec_util",
    "pcie_tx_bps",
    "pcie_rx_bps",
    "energy_j",
    "sm_active",
    "sm_occupancy",
    "tensor_active",
    "dram_bw_util",
    "fp32_active",
    "fp16_active",
    "fp64_active",
    "int_active",
)


def fixture(name: str) -> bytes:
    return (FIX / name).read_bytes()


class FakeRunner:
    """Stands in for subprocess: answers ``ioreg`` / ``system_profiler`` from fixtures and records calls."""

    def __init__(self, ioreg: Optional[bytes], profiler: Optional[bytes] = None) -> None:
        self.ioreg, self.profiler = ioreg, profiler
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(self, cmd: Sequence[str], timeout: float) -> Optional[bytes]:
        self.calls.append((tuple(cmd), timeout))
        tool = os.path.basename(cmd[0])
        return self.ioreg if tool == "ioreg" else self.profiler if tool == "system_profiler" else None

    def count(self, tool: str) -> int:
        return sum(os.path.basename(c[0]) == tool for c, _ in self.calls)


def apple_gpu(util: float, used: float = 4_096.0, uuid: str = "apple-0-apple-m1-pro", index: int = 0) -> GpuSample:
    return GpuSample(
        index, uuid, "Apple M1 Pro", "apple", util=util, mem_used_mib=used, mem_total_mib=UNIFIED, renderer_util=util
    )


def history_of(make: Any, seconds: int, start: float, step: float = 1.0) -> HistoryStore:
    h = HistoryStore()
    for k in range(int(seconds / step)):
        t = start + k * step
        h.add(Sample(t=t, gpus=make(k), cpu_util=20, cpu_per_core=(20,), mem_used=8 << 30, mem_total=32 << 30))
    return h


# ---- ioreg parser --------------------------------------------------------------------------
def test_fixtures_are_marked_as_reconstructions() -> None:
    for f in FIX.iterdir():
        if "_mac_capture" not in f.name:  # real captures saved by scripts/mac_probe.py --save-fixtures
            assert b"RECONSTRUCTION" in f.read_bytes(), f"{f.name}: say it is not a real capture"


@pytest.mark.skipif(not (FIX / "ioreg_mac_capture.plist").exists(), reason="no real Mac capture saved yet")
def test_real_mac_capture_parses() -> None:
    gpus = parse_ioreg_plist(fixture("ioreg_mac_capture.plist"), UNIFIED)
    assert gpus and all(g.util is not None and g.mem_used_mib is not None for g in gpus)
    if (FIX / "system_profiler_mac_capture.json").exists():
        assert parse_profiler_json(fixture("system_profiler_mac_capture.json"))


def test_apple_silicon_fixture() -> None:
    (g,) = parse_ioreg_plist(fixture("ioreg_apple_m1pro.plist"), UNIFIED)
    assert (g.vendor, g.name, g.uuid, g.index) == ("apple", "Apple M1 Pro", "apple-0-apple-m1-pro", 0)
    assert (g.util, g.renderer_util, g.tiler_util) == (23.0, 21.0, 9.0)
    assert g.mem_used_mib == 1_182_793_728 / 1048576  # "In use system memory", not the larger "Alloc"
    assert g.mem_total_mib == UNIFIED and g.unified_memory and g.mem_label == "GPU memory (unified)"
    assert all(getattr(g, f) is None for f in NVIDIA_ONLY) and g.throttle == ()


def test_intel_mac_with_amd_fixture() -> None:
    intel, amd = parse_ioreg_plist(fixture("ioreg_intel_amd.plist"), UNIFIED)
    assert (intel.vendor, intel.name, intel.uuid) == ("intel", "Intel GPU", "intel-0-intelaccelerator")
    assert intel.util == 5.0 and intel.mem_used_mib == 512.0 and intel.mem_total_mib is None  # no RAM share
    assert (amd.vendor, amd.name) == ("amd", "AMD Radeon Pro 5500M")  # <data> model, NUL stripped
    assert (amd.util, amd.mem_used_mib, amd.mem_total_mib) == (37.0, 2048.0, 8192.0)
    assert (amd.temp_c, amd.fan_pct, amd.clock_mhz, amd.mem_clock_mhz) == (61.0, 34.0, 1300.0, 1500.0)
    assert amd.power_w is None and not amd.unified_memory and amd.mem_label == "VRAM"


@pytest.mark.parametrize("name", ["ioreg_vm.plist", "ioreg_empty.plist"])
def test_vm_and_empty_output_have_no_gpu(name: str) -> None:
    assert parse_ioreg_plist(fixture(name), UNIFIED) == []


@pytest.mark.parametrize(
    "data", [b"", b"   \n", b"not a plist", b"\x00\x01\x02", b"<plist><array><dict><key>x</key>", b"bplist00junk"]
)
def test_garbage_never_raises(data: bytes) -> None:
    assert parse_ioreg_plist(data) == []


def test_values_are_parsed_defensively() -> None:
    odd = {
        "IOClass": "AGXAcceleratorG14G",
        "PerformanceStatistics": {
            "Device Utilization %": True,  # bool is not a number
            "Renderer Utilization %": "42",  # numeric string
            "Tiler Utilization %": 250,  # out of range
            "In use system memory": -5,  # negative
        },
    }
    nan = {"IOClass": "AGXAcceleratorG15X", "PerformanceStatistics": {"Device Utilization %": math.nan}}
    data = plistlib.dumps(["junk", 7, odd, {"IOClass": "X", "PerformanceStatistics": [1, 2]}, nan])
    (g,) = parse_ioreg_plist(data, UNIFIED)  # the NaN-only entry has nothing usable and is skipped
    assert (g.name, g.uuid, g.vendor) == ("Apple GPU", "apple-0-agxacceleratorg14g", "apple")
    assert (g.util, g.renderer_util, g.tiler_util, g.mem_used_mib) == (42.0, 42.0, None, None)
    single = plistlib.dumps(odd)  # a dict root instead of an array
    assert len(parse_ioreg_plist(single, UNIFIED)) == 1


def test_uuid_is_stable_across_reads() -> None:
    a = parse_ioreg_plist(fixture("ioreg_apple_m1pro.plist"), UNIFIED)[0]
    busy = fixture("ioreg_apple_m1pro.plist").replace(b"<integer>23</integer>", b"<integer>99</integer>")
    b = parse_ioreg_plist(busy, 16_384.0)[0]
    assert a.uuid == b.uuid and b.util == 99.0


# ---- system_profiler -----------------------------------------------------------------------
def test_profiler_fixture_drops_displays_and_serials() -> None:
    (p,) = parse_profiler_json(fixture("system_profiler_apple_m1pro.json"))
    assert (p.model, p.cores, p.metal, p.vram_mib) == ("Apple M1 Pro", 16, "Metal 3", None)
    assert "SERIAL" not in repr(p) and "Color LCD" not in repr(p)
    intel, amd = parse_profiler_json(fixture("system_profiler_intel_amd.json"))
    assert intel.model == "Intel UHD Graphics 630" and intel.vram_mib is None  # shared VRAM is not VRAM
    assert (amd.model, amd.vram_mib, amd.metal) == ("AMD Radeon Pro 5500M", 8192.0, "Metal 3")
    assert parse_profiler_json(b"{broken") == [] and parse_profiler_json(b"[]") == []


def test_metal_labels() -> None:
    assert metal_label("spdisplays_metal3") == "Metal 3"
    assert metal_label("spdisplays_mtlgpufamilymac2") == "Metal (Mac family 2)"
    assert metal_label("spdisplays_supported") == "Metal"
    assert metal_label(None) is None and metal_label(3) is None


def test_details_merge_ioreg_and_profiler() -> None:
    rows = parse_ioreg(fixture("ioreg_apple_m1pro.plist"), UNIFIED)
    prof = parse_profiler_json(fixture("system_profiler_apple_m1pro.json"))
    (d,) = apple_details([g for g, _ in rows], [i for _, i in rows], prof)
    assert (d.architecture, d.gpu_cores, d.metal_family, d.unified_memory) == ("Apple GPU (G13X)", 16, "Metal 3", True)
    assert d.vram_total_mib == UNIFIED and d.compute_capability is None and d.cuda_cores is None
    line = panels.arch_summary(d, rows[0][0]).plain
    assert line == "Apple M1 Pro · 16-core GPU · Metal 3 · unified memory"
    # No profiler output: ioreg's core count still shows.
    (bare,) = apple_details([g for g, _ in rows], [i for _, i in rows], [])
    assert bare.gpu_cores == 16 and bare.metal_family is None


# ---- collector -----------------------------------------------------------------------------
def test_collector_reads_through_the_runner_without_a_shell() -> None:
    run = FakeRunner(fixture("ioreg_apple_m1pro.plist"), fixture("system_profiler_apple_m1pro.json"))
    c = AppleCollector(runner=run, platform="darwin", unified_total_mib=UNIFIED)
    assert c.detect()
    (g,) = c.sample()
    assert g.util == 23.0 and g.mem_total_mib == UNIFIED
    cmd, timeout = run.calls[0]
    assert os.path.basename(cmd[0]) == "ioreg" and cmd[1:] == IOREG_ARGS and timeout == IOREG_TIMEOUT_S
    assert c.details()[0].metal_family == "Metal 3" and c.details() is c.details()
    assert run.count("system_profiler") == 1  # static facts are read once


def test_collector_off_macos_spawns_nothing() -> None:
    run = FakeRunner(fixture("ioreg_apple_m1pro.plist"))
    assert not AppleCollector(runner=run, platform="linux").detect() and run.calls == []
    assert not AppleCollector(runner=FakeRunner(fixture("ioreg_vm.plist")), platform="darwin").detect()


def test_failed_reads_keep_the_gpu_without_a_reading() -> None:
    run = FakeRunner(fixture("ioreg_apple_m1pro.plist"))
    c = AppleCollector(runner=run, platform="darwin", unified_total_mib=UNIFIED)
    assert c.detect()
    run.ioreg = None  # ioreg missing / timed out
    for _ in range(MAX_MISSES):
        (g,) = c.sample()
        assert g.uuid == "apple-0-apple-m1-pro" and g.util is None and g.mem_used_mib is None
    assert c.sample() == ()  # gone for good after repeated failures
    run.ioreg = fixture("ioreg_apple_m1pro.plist")
    assert c.sample()[0].util == 23.0


def test_generic_names_are_upgraded_from_system_profiler() -> None:
    run = FakeRunner(fixture("ioreg_intel_amd.plist"), fixture("system_profiler_intel_amd.json"))
    c = AppleCollector(runner=run, platform="darwin", unified_total_mib=UNIFIED)
    assert c.detect()
    intel, amd = c.details()
    assert intel.name == "Intel UHD Graphics 630" and amd.vram_total_mib == 8192.0
    s = c.sample()
    assert s[0].name == "Intel UHD Graphics 630" and s[0].uuid == "intel-0-intelaccelerator"  # id unchanged


def test_run_command_real_subprocess() -> None:
    py = sys.executable
    assert (run_command([py, "-c", "print('ok')"], 30) or b"").strip() == b"ok"
    assert run_command([py, "-c", "import sys; sys.exit(3)"], 30) is None
    assert run_command([py, "-c", "import time; time.sleep(30)"], 0.5) is None  # timeout
    assert run_command(["gpu-pulse-no-such-tool-xyz"], 5) is None


# ---- monitor -------------------------------------------------------------------------------
def test_monitor_picks_apple_first_and_raises_the_interval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = FakeRunner(fixture("ioreg_apple_m1pro.plist"), fixture("system_profiler_apple_m1pro.json"))
    monkeypatch.setattr(
        "gpu_pulse.monitor.AppleCollector",
        lambda: AppleCollector(runner=run, platform="darwin", unified_total_mib=UNIFIED),
    )
    nvml: list[int] = []
    monkeypatch.setattr("gpu_pulse.monitor.NvidiaCollector.detect", lambda self: nvml.append(1) or False)
    mon = Monitor(MonitorConfig(interval_s=0.5, history_path=tmp_path / "h.json", persist=False))
    assert mon.vendor == "apple" and mon.cfg.interval_s == 1.0 and nvml == []
    s = mon.tick()
    assert s.gpus[0].uuid == "apple-0-apple-m1-pro" and s.gpus[0].temp_c is None
    assert mon.gpu_details()[0].gpu_cores == 16
    assert mon.processes == [] and not mon.has_processes and mon.fleet().findings == ()
    mon.close()


def test_apple_demo_goes_through_the_parser() -> None:
    mon = Monitor(MonitorConfig(demo=True, demo_vendor="apple", interval_s=0.5, persist=False))
    assert mon.vendor == "apple" and mon.cfg.interval_s == 1.0  # same floor as a real Mac
    (g,) = mon.tick().gpus
    assert g.vendor == "apple" and g.renderer_util is not None and g.mem_total_mib
    assert all(getattr(g, f) is None for f in NVIDIA_ONLY)
    (d,) = mon.gpu_details()
    assert d.unified_memory and d.metal_family == "Metal 3" and d.uuid == g.uuid
    assert mon.history.stats("1 hour", 60, g.uuid).util is not None  # seeded history matches the GPU id
    assert mon.processes == []
    mon.close()


def test_nvidia_demo_keeps_its_interval() -> None:
    mon = Monitor(MonitorConfig(demo=True, interval_s=0.5, persist=False))
    assert mon.cfg.interval_s == 0.5 and mon.has_processes
    mon.close()


# ---- rules stay quiet without NVIDIA-only data ---------------------------------------------
def test_alerts_never_report_heat_for_apple() -> None:
    eng = AlertEngine(AlertConfig(interval_s=1.0))
    fired: list[str] = []
    t0 = 1_700_000_000.0
    for k in range(40 * 60):  # 20 min busy, then 20 min idle while holding 61% of unified memory
        g = apple_gpu(95.0 if k < 1200 else 2.0, used=20_000.0)
        fired += [a.rule for a in eng.update(Sample(t0 + k, (g,), 10.0, (10.0,), 1, 2))]
    assert "hot" not in fired and "vram_full" not in fired
    assert set(fired) == {"job_finished", "idle_held"}


def test_diagnosis_uses_utilization_and_no_nvidia_rules() -> None:
    start = time.time() - 600
    h = history_of(lambda k: (apple_gpu(95.0),), 600, start)
    d = diagnose(h, "apple-0-apple-m1-pro", now=start + 600)
    assert d.bottleneck not in ("power", "thermal", "launch", "memory") and d.score_basis == "utilization"
    assert d.throttle == () and d.score is not None
    assert "Power and thermal limits not visible" in d.evidence
    text = " ".join(f.title for f in d.findings)
    assert not any(w in text for w in ("PCIe", "Clocks", "Temperature", "Tensor", "SM"))
    assert [r.name for r in d.health][:2] == ["GPU", "GPU memory"]
    hour = h.stats("1 hour", 60, "apple-0-apple-m1-pro", start + 600)
    titles = " ".join(i.title for i in build_insights(hour, hour, apple_gpu(95.0), 80.0))
    assert "°C" not in titles and "power" not in titles.lower()
    bucket = h.snapshot()["minutes"][-1]["gpus"]["apple-0-apple-m1-pro"]
    assert bucket["power"] == 0 and bucket["tempMax"] == 0 and "sm" not in bucket


def test_fleet_rules_need_no_sensors() -> None:
    start = time.time() - 120

    def two(k: int) -> tuple[GpuSample, ...]:
        return (apple_gpu(95.0, 12_000.0), apple_gpu(20.0 if k % 3 else 90.0, 12_000.0, "apple-1-x", 1))

    fleet = analyze_fleet(history_of(two, 120, start, 0.5), now=start + 120)
    assert not any(w in f.title for f in fleet.findings for w in ("throttl", "PCIe", "power-capped", "thermal"))
    assert gpu_flag(apple_gpu(95.0), None).level == "good"
    assert "GPU mem" in gpu_row(apple_gpu(50.0), False, None, 150, 80.0).plain


# ---- UI ------------------------------------------------------------------------------------
def _text(widget: Any) -> str:
    """A widget's rendered text with whitespace collapsed (wrapped lines are padded to the panel width)."""
    lines = widget.render_lines(widget.region.reset_offset)
    return " ".join(" ".join("".join(seg.text for seg in line) for line in lines).replace("│", " ").split())


async def _drive(size: tuple[int, int]) -> dict[str, Any]:
    mon = Monitor(MonitorConfig(demo=True, demo_vendor="apple", persist=False, interval_s=0.5))
    app = GpuDashboard(mon, fps=12)
    out: dict[str, Any] = {}
    async with app.run_test(size=size) as pilot:
        await pilot.pause(1.5)
        for rid in ("temp", "power"):
            ring = app.query_one(f"#ring-{rid}", RingGauge)
            out[rid] = (ring.target, ring.caption)
        out["mem_title"] = str(app.query_one("#ring-vram", RingGauge).border_title)
        arch = app.query_one("#ov-arch")
        out["arch"] = _text(arch) if arch.display else None
        await pilot.press("2")
        await pilot.pause(0.4)
        out["procs"] = _text(app.query_one("#procs"))
        out["live"] = _text(app.query_one("#gpu-live"))
        out["card"] = _text(app.query_one("#arch"))
        for key in ("3", "4", "5"):
            await pilot.press(key)
            await pilot.pause(0.3)
        out["tab"] = app.tab
        out["diag"] = _text(app.query_one("#diag-full"))
        out["health"] = _text(app.query_one("#health-full"))
    mon.close()
    return out


@pytest.mark.parametrize("size", [(80, 24), (120, 36), (160, 62)])
def test_dashboard_shows_an_apple_gpu(size: tuple[int, int]) -> None:
    r = asyncio.run(_drive(size))
    assert r["temp"] == (None, "n/a") and r["power"] == (None, "n/a")
    assert r["mem_title"].startswith("GPU MEMORY")
    if r["arch"] is not None:
        assert "16-core GPU · Metal 3 · unified memory" in r["arch"]
    assert "not available on macOS" in r["procs"]
    assert "Renderer" in r["live"] and "Tiler" in r["live"]
    assert "Metal 3" in r["card"] and "Unified yes" in r["card"]
    assert r["tab"] == "diagnosis" and "not reported by this GPU" in r["diag"]
    assert "GPU memory" in r["health"] and "VRAM" not in r["health"]


def test_status_names_unified_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gpu_pulse.status.time.sleep", lambda s: None)
    mon = Monitor(MonitorConfig(demo=True, demo_vendor="apple", persist=False))
    console = Console(record=True, width=120, color_system=None)
    run_status(mon, console)
    out = console.export_text()
    mon.close()
    assert "Apple M1 Pro (simulated)" in out and "GPU memory (unified)" in out and "GPU mem" in out
    sensors = next(line for line in out.splitlines() if "Temp" in line)
    assert "Power" in sensors and sensors.count("n/a") == 2 and "°C" not in sensors
    assert "VRAM" not in out


# ---- CLI and machine specs -----------------------------------------------------------------
def test_demo_vendor_flag() -> None:
    assert _parser().parse_args(["--demo", "--demo-vendor", "apple", "status"]).demo_vendor == "apple"
    assert _parser().parse_args(["status"]).demo_vendor == "nvidia"
    for argv in (["--demo-vendor", "apple", "status"], ["--demo", "--demo-vendor", "apple", "--demo-gpus", "2"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2


def test_macos_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(system_mod.platform, "mac_ver", lambda: ("14.5", ("", "", ""), "arm64"))
    monkeypatch.setattr(system_mod, "_sysctl", lambda name: "Apple M1 Pro" if "brand" in name else None)
    assert system_mod._os_name() == "macOS 14.5" and system_mod._cpu_model() == "Apple M1 Pro"

    def no_freq() -> None:
        raise FileNotFoundError("no cpu frequency on this machine")

    monkeypatch.setattr(system_mod.psutil, "cpu_freq", no_freq)
    assert system_mod.collect_specs(("Apple M1 Pro",), None, None).cpu_max_ghz is None
