"""Multi-GPU support: simulated GPUs, cross-GPU findings, history per GPU, CLI, status and the UI."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

import pytest
from rich.console import Console

from gpu_pulse import demo
from gpu_pulse.app import GpuDashboard
from gpu_pulse.cli import _parser
from gpu_pulse.fleet import aggregate, analyze_fleet, gpu_flag
from gpu_pulse.fleetview import fleet_height, fleet_panel
from gpu_pulse.history import HistoryStore
from gpu_pulse.models import GpuDetails, GpuSample, Sample
from gpu_pulse.monitor import Monitor, MonitorConfig
from gpu_pulse.status import run_status
from gpu_pulse.widgets import RingGauge

T0 = 1_700_000_000.0


def gpu(i: int, util: float, **kw: Any) -> GpuSample:
    base: dict[str, Any] = dict(
        mem_used_mib=8_000.0,
        mem_total_mib=16_000.0,
        temp_c=65.0,
        power_w=150.0,
        power_limit_w=180.0,
        pcie_gen=4,
        pcie_width=16,
    )
    base.update(kw)
    return GpuSample(i, f"GPU-{i}", f"Test GPU {i}", "nvidia", util=util, **base)


def store(make: Callable[[int], tuple[GpuSample, ...]], seconds: int = 60, step: float = 0.5) -> HistoryStore:
    h = HistoryStore()
    for k in range(int(seconds / step)):
        t = T0 + k * step
        h.add(Sample(t=t, gpus=make(k), cpu_util=20, cpu_per_core=(20,), mem_used=1, mem_total=4))
    return h


def fleet_of(make: Callable[[int], tuple[GpuSample, ...]], details: Any = ()) -> Any:
    return analyze_fleet(store(make), details, now=T0 + 60)


# ---- CLI -----------------------------------------------------------------------------------
def test_demo_gpus_flag_parses() -> None:
    assert _parser().parse_args(["--demo"]).demo_gpus == 1
    assert _parser().parse_args(["--demo", "--demo-gpus", "4", "status"]).demo_gpus == 4


def test_demo_gpus_without_demo_is_an_error() -> None:
    from gpu_pulse.cli import main

    with pytest.raises(SystemExit) as e:
        main(["--demo-gpus", "4", "status"])
    assert e.value.code == 2


@pytest.mark.parametrize("bad", ["0", "17", "two", "-1"])
def test_demo_gpus_flag_rejects_bad_counts(bad: str) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--demo", "--demo-gpus", bad])


# ---- simulated GPUs ------------------------------------------------------------------------
def test_demo_gpus_are_distinct_and_play_their_roles() -> None:
    t = demo._T0 + 30  # "steady" phase: the job is busy
    gpus = demo.demo_gpus(t, 4)
    assert [g.index for g in gpus] == [0, 1, 2, 3]
    assert len({g.uuid for g in gpus}) == 4 and len({g.name for g in gpus}) == 4
    assert gpus[0].uuid == demo.DEMO_UUID
    utils = [demo.demo_gpus(demo._T0 + 30 + k * 0.5, 4)[1].util or 0 for k in range(12)]
    assert sum(utils) / len(utils) < 40 < (gpus[0].util or 0)  # straggler: bursts, mostly waiting
    assert "thermal" in gpus[2].throttle and (gpus[2].temp_c or 0) >= 85
    assert gpus[3].pcie_width == 4 and (gpus[3].vram_pct or 0) > 85 > (gpus[1].vram_pct or 0)
    assert demo.demo_gpus(t, 1) == (demo.demo_gpu(t),) or demo.demo_gpus(t, 1)[0].uuid == demo.DEMO_UUID
    details = demo.demo_details_all(4)
    assert [d.uuid for d in details] == [g.uuid for g in gpus]


def test_demo_monitor_samples_n_gpus_with_history_and_processes() -> None:
    mon = Monitor(MonitorConfig(demo=True, demo_gpus=3, persist=False))
    s = mon.tick()
    assert len(s.gpus) == 3 and len(mon.gpu_details()) == 3
    assert {p.gpu_uuid for p in mon.processes} <= {g.uuid for g in s.gpus}
    for g in s.gpus:  # every simulated GPU has its own day of per-minute history
        assert mon.history.stats("24 hours", 1440, g.uuid).covered_minutes > 600
    mon.close()


# ---- cross-GPU findings --------------------------------------------------------------------
def test_single_gpu_has_no_fleet_findings() -> None:
    f = fleet_of(lambda k: (gpu(0, 90),))
    assert f.findings == () and len(f.stats) == 1


def test_balanced_gpus_have_no_findings() -> None:
    f = fleet_of(lambda k: tuple(gpu(i, 90 + (k + i) % 3) for i in range(4)))
    assert f.findings == ()
    assert all(flag.level == "good" and not flag.tags for flag in f.flags.values())


def test_straggler_is_flagged_with_honest_wording() -> None:
    f = fleet_of(lambda k: (gpu(0, 92), gpu(1, 6), gpu(2, 90)))
    titles = [i.title for i in f.findings]
    assert titles == ["GPU 1 lags its siblings: a straggler if they run one job"]
    ins = f.findings[0]
    # VRAM in use on every GPU, but no process list to confirm one job: a hint, not a warning.
    assert ins.level == "info" and "NCCL" in ins.detail and "Unrelated jobs" in ins.detail
    assert "over the last 60 s" in ins.detail and "can't confirm" in ins.detail
    assert f.flags["GPU-1"].tags == ("straggler",) and f.basis == "util"


def test_straggler_uses_sm_activity_when_every_gpu_has_it() -> None:
    # Utilization looks equal; SM activity shows GPU 2 doing far less real work.
    f = fleet_of(lambda k: (gpu(0, 95, sm_active=70.0), gpu(1, 95, sm_active=72.0), gpu(2, 95, sm_active=12.0)))
    assert f.basis == "sm"
    assert [i.title for i in f.findings] == ["GPU 2 lags its siblings: a straggler if they run one job"]


def test_idle_machine_is_not_an_imbalance() -> None:
    f = fleet_of(lambda k: (gpu(0, 3), gpu(1, 40), gpu(2, 2)))
    assert not any("lags" in i.title for i in f.findings)


def test_uneven_vram() -> None:
    f = fleet_of(lambda k: (gpu(0, 90, mem_used_mib=14_800.0), gpu(1, 90, mem_used_mib=5_000.0)))
    (ins,) = f.findings
    assert ins.title.startswith("VRAM uneven across GPUs: GPU 0 92%") and ins.level == "warning"
    assert "set_device" in ins.detail


def test_lone_thermal_throttle_but_not_when_all_throttle() -> None:
    lone = fleet_of(lambda k: (gpu(0, 90), gpu(1, 90, throttle=("thermal",), temp_c=87.0), gpu(2, 90)))
    assert [i.title for i in lone.findings] == ["GPU 1 thermal-throttles while its siblings don't"]
    assert "87°C" in lone.findings[0].detail and lone.flags["GPU-1"].tags == ("thermal",)
    everyone = fleet_of(lambda k: tuple(gpu(i, 90, throttle=("thermal",)) for i in range(3)))
    assert not any("thermal" in i.title for i in everyone.findings)


def test_lone_power_cap_points_at_a_lower_limit() -> None:
    f = fleet_of(lambda k: (gpu(0, 90), gpu(1, 90, throttle=("power cap",), power_limit_w=150.0)))
    (ins,) = f.findings
    assert ins.title == "GPU 1 power-capped while its siblings aren't"
    assert "150 W vs 180 W" in ins.detail and "-pl 180" in ins.detail


def test_pcie_width_mismatch_and_idle_generation_drop() -> None:
    d = [GpuDetails(i, f"GPU-{i}", "x", pcie_max_gen=4, pcie_max_width=16) for i in range(2)]
    narrow = fleet_of(lambda k: (gpu(0, 90), gpu(1, 90, pcie_width=4)), d)
    (ins,) = narrow.findings
    assert ins.title == "GPU 1 on a narrower PCIe link (Gen4 x4)" and ins.level == "warning"
    assert "supports x16" in ins.detail and narrow.flags["GPU-1"].tags == ("x4 link",)
    # An idle GPU drops its link generation to save power: not a finding.
    idle = fleet_of(lambda k: (gpu(0, 90), gpu(1, 2, pcie_gen=1)))
    assert not any("PCIe" in i.title for i in idle.findings)
    busy_slow = fleet_of(lambda k: (gpu(0, 90), gpu(1, 90, pcie_gen=3)))
    assert [i.title for i in busy_slow.findings] == ["GPU 1 on a slower PCIe link (Gen3 x16)"]


def test_demo_fleet_finds_all_four_patterns() -> None:
    mon = Monitor(MonitorConfig(demo=True, demo_gpus=4, persist=False))
    t = demo._T0 + 30
    for k in range(80):  # 40 s: past the 30 s a GPU needs before it is compared
        s = Sample(t=t + k * 0.5, gpus=demo.demo_gpus(t + k * 0.5, 4), cpu_util=10, cpu_per_core=(10,),
                   mem_used=1, mem_total=4)  # fmt: skip
        mon.history.add(s)
    procs = demo.demo_processes(demo.demo_gpus(t, 4))
    f = analyze_fleet(mon.history, mon.gpu_details(), now=t + 40, processes=procs)
    titles = " | ".join(i.title for i in f.findings)
    for expected in ("GPU 1 lags", "VRAM uneven", "GPU 2 thermal-throttles", "GPU 3 on a narrower PCIe link"):
        assert expected in titles
    mon.close()


def test_aggregate_and_flags() -> None:
    gs = (gpu(0, 80, temp_c=70.0), gpu(1, 40, temp_c=88.0, throttle=("thermal",)), gpu(2, 60, power_w=None))
    agg = aggregate(gs)
    assert agg.count == 3 and agg.util_mean == 60 and agg.hottest == (1, 88.0) and agg.throttled == (1,)
    assert agg.power_w == 300.0 and agg.power_limit_w == 540.0 and round(agg.vram_total_gib, 1) == 46.9
    assert gpu_flag(gs[1], None, 80.0).level == "critical" and gpu_flag(gs[0], None, 80.0).level == "warning"


def test_fleet_panel_rows_and_heights() -> None:
    gs = tuple(gpu(i, 50 + i) for i in range(6))
    assert fleet_height(1, 120) == 0 and fleet_height(4, 120) == 7
    assert fleet_height(6, 180) == 3 + 3  # two GPUs per line on wide terminals
    body, hits = fleet_panel(gs, 2, None, 116)
    lines = [r.plain for r in body.renderables]  # type: ignore[union-attr]
    assert len(lines) == 7 and lines[2].startswith("▶")  # selected GPU marked
    assert [h.pos for h in hits] == list(range(6)) and all(len(x) <= 116 for x in lines)
    many = tuple(gpu(i, 50) for i in range(12))
    _, hits = fleet_panel(many, 11, None, 100, max_lines=4)  # scrolls to keep the selection visible
    assert len(hits) == 4 and hits[-1].pos == 11


# ---- NVML cost per GPU ---------------------------------------------------------------------
def test_nvml_sampling_reads_slow_fields_every_two_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_nvidia_throttle import make

    col, fake = make(
        monkeypatch,
        nvmlDeviceGetCount=lambda: 3,
        nvmlDeviceGetUUID=lambda h: f"GPU-{h}",
        nvmlDeviceGetFanSpeed=lambda h: 40,
        nvmlDeviceGetTotalEnergyConsumption=lambda h: 2_000_000,
    )
    for _ in range(5):
        gs = col.sample()
    assert [g.uuid for g in gs] == ["GPU-h0", "GPU-h1", "GPU-h2"] and [g.index for g in gs] == [0, 1, 2]
    assert all(g.fan_pct == 40.0 and g.energy_j == 2000.0 for g in gs)
    # 5 samples x 3 GPUs, but fan and energy are read once per GPU within the 2 s window.
    assert fake.calls.count("nvmlDeviceGetUtilizationRates") == 15
    assert fake.calls.count("nvmlDeviceGetFanSpeed") == 3
    assert fake.calls.count("nvmlDeviceGetTotalEnergyConsumption") == 3


# ---- history stays per GPU -----------------------------------------------------------------
def test_two_gpus_minutes_stay_separate() -> None:
    now = float(int(time.time() // 60) * 60 - 20)  # 40 s into a recent minute (older ones fall out of retention)
    h = HistoryStore()
    for k in range(240):  # 2 minutes at 0.5 s
        t = now - 120 + k * 0.5
        gs = (gpu(0, 80.0, mem_used_mib=12_000.0), gpu(1, 10.0, mem_used_mib=2_000.0, sm_active=5.0))
        h.add(Sample(t=t, gpus=gs, cpu_util=20, cpu_per_core=(20,), mem_used=1, mem_total=4))
    start, a = h.heatmap("GPU-0", hours=1, now=now)
    _, b = h.heatmap("GPU-1", hours=1, now=now)
    filled_a = [v for v in a if v is not None]
    filled_b = [v for v in b if v is not None]
    assert filled_a and set(filled_a) == {80.0} and set(filled_b) == {10.0}
    assert h.heatmap("GPU-1", hours=1, now=now, metric="sm")[1][-1] == 5.0
    assert h.heatmap("GPU-0", hours=1, now=now, metric="sm")[1][-1] is None  # GPU 0 never reported GPM
    s0, s1 = h.stats("5 min", 5, "GPU-0", now), h.stats("5 min", 5, "GPU-1", now)
    assert (s0.util, s1.util) == (80.0, 10.0) and s0.covered_minutes == s1.covered_minutes == 3
    assert round(s0.vram or 0) == 75 and round(s1.vram or 0) == 12
    assert (s0.busy_share, s1.idle_share) == (1.0, 0.0)
    rows = h.to_csv().strip().splitlines()[1:]
    assert sum(",GPU-0," in r for r in rows) == sum(",GPU-1," in r for r in rows) == 3
    copy = HistoryStore()
    copy.load(h.snapshot())  # round trip keeps both GPUs apart
    assert copy.stats("5 min", 5, "GPU-1", now).util == 10.0


# ---- status --------------------------------------------------------------------------------
def test_status_prints_every_gpu_and_an_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gpu_pulse.status.time.sleep", lambda s: None)
    mon = Monitor(MonitorConfig(demo=True, demo_gpus=3, persist=False))
    console = Console(record=True, width=150, color_system=None)
    run_status(mon, console)
    out = console.export_text()
    mon.close()
    for i in range(3):
        assert f"GPU {i}  DEMO GPU {i}" in out
    assert "All 3 GPUs" in out and "hottest GPU 2" in out


# ---- dashboard -----------------------------------------------------------------------------
def _ring_title(app: GpuDashboard) -> str:
    return str(app.query_one("#ring-gpu", RingGauge).border_title)


async def _drive(size: tuple[int, int], n: int = 4) -> dict[str, Any]:
    mon = Monitor(MonitorConfig(demo=True, demo_gpus=n, persist=False, interval_s=0.5))
    app = GpuDashboard(mon, fps=12)
    out: dict[str, Any] = {}
    async with app.run_test(size=size) as pilot:
        await pilot.pause(1.0)
        fleet = app.query_one("#fleet")
        out["fleet_shown"] = fleet.display
        out["hits"] = len(getattr(fleet, "hits", []))
        out["header"] = app.query_one("#header").render().plain if fleet.display else ""  # type: ignore[union-attr]
        out["title0"] = _ring_title(app)
        await pilot.press("g")
        await pilot.pause(0.3)
        out["after_g"] = (app.gpu_idx, _ring_title(app))
        await pilot.press("G")
        await pilot.pause(0.3)
        out["after_G"] = app.gpu_idx
        await pilot.press("g", "3")  # g then a digit: jump to GPU 3, stay on the Overview
        await pilot.pause(0.3)
        out["after_g3"] = (app.gpu_idx, app.tab)
        if fleet.display:
            await pilot.click("#fleet", offset=(10, 3))  # third GPU line (border row is y=0)
            await pilot.pause(0.3)
            out["after_click"] = app.gpu_idx
        rings = list(app.query(RingGauge))
        out["rings"] = len(rings)
        out["ring_h"] = rings[0].region.height
        out["layout"] = app.layout_plan
        await pilot.press("3")  # a digit on its own still switches tabs
        await pilot.pause(0.3)
        out["tab"] = app.tab
    mon.close()
    return out


@pytest.mark.parametrize("size", [(90, 30), (130, 40), (200, 60)])
def test_dashboard_with_four_gpus(size: tuple[int, int]) -> None:
    r = asyncio.run(_drive(size))
    assert r["fleet_shown"] and r["hits"] == 4
    assert "GPU 0" in r["header"] and "1/4" in r["header"]
    assert r["title0"].startswith("GPU 0 ·")
    assert r["after_g"][0] == 1 and r["after_g"][1].startswith("GPU 1 ·")
    assert r["after_G"] == 0
    assert r["after_g3"] == (3, "overview")
    assert r["after_click"] == 2
    assert r["tab"] == "system"
    assert r["rings"] == 6 and r["ring_h"] == r["layout"].ring_h and r["layout"].fleet_h == 7


def test_single_gpu_layout_has_no_fleet_panel() -> None:
    r = asyncio.run(_drive((130, 40), n=1))
    assert not r["fleet_shown"] and r["layout"].fleet_h == 0
    assert r["after_g"][0] == 0 and r["title0"].startswith("GPU ·")


def test_fleet_findings_feed_the_diagnosis_insights() -> None:
    async def run() -> str:
        mon = Monitor(MonitorConfig(demo=True, demo_gpus=4, persist=False, interval_s=0.5))
        t, now = demo._T0 + 30, time.time()
        for k in range(80):  # pre-fill 40 s of the raw window instead of waiting for it
            gs = demo.demo_gpus(t + k * 0.5, 4)
            mon.history.add(Sample(t=now - 40 + k * 0.5, gpus=gs, cpu_util=10, cpu_per_core=(10,), mem_used=1,
                                   mem_total=4))  # fmt: skip
        app = GpuDashboard(mon, fps=0)
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.press("5")
            await pilot.pause(0.6)
            text = "\n".join(
                "".join(seg.text for seg in line)
                for line in app.query_one("#insights").render_lines(app.query_one("#insights").region.reset_offset)
            )
        mon.close()
        return text

    text = asyncio.run(run())
    assert "lags its siblings" in text and "narrower PCIe link" in text
