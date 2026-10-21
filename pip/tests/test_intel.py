"""Diagnosis engine: bottleneck rules, score, sessions, forecasts, findings, health, speed."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any, Callable, Optional

import pytest

from gpu_pulse import demo, intel
from gpu_pulse.history import HistoryStore
from gpu_pulse.intel import Diagnosis, diagnose
from gpu_pulse.models import GpuDetails, GpuSample, Sample

T0 = 1_792_576_800.0  # minute-aligned
STEP = 0.5


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    intel._HOURLY.clear()
    yield
    intel._HOURLY.clear()


def gs(util: Optional[float], **kw: Any) -> GpuSample:
    base: dict[str, Any] = dict(
        index=0,
        uuid="G",
        name="Test GPU",
        vendor="nvidia",
        util=util,
        mem_used_mib=2_000.0,
        mem_total_mib=10_000.0,
        temp_c=60.0,
        power_w=100.0,
        power_limit_w=200.0,
        clock_mhz=2_700.0,
        clock_max_mhz=3_000.0,
    )
    base.update(kw)
    return GpuSample(**base)


def smp(t: float, g: Optional[GpuSample], cpu: float = 20.0, **kw: Any) -> Sample:
    cores = kw.pop("cores", (cpu,) * 8)
    return Sample(
        t=t,
        gpus=(g,) if g else (),
        cpu_util=cpu,
        cpu_per_core=cores,
        mem_used=kw.pop("mem", 40),
        mem_total=100,
        **kw,
    )


def feed(h: HistoryStore, start: float, seconds: float, make: Callable[[float, int], Sample]) -> float:
    """Add samples every 0.5 s; returns the time of the last one."""
    t = start
    for i in range(int(seconds / STEP)):
        t = start + i * STEP
        h.add(make(t, i))
    return t


def bursty(t: float) -> float:
    """Busy plateaus with a ~1.5 s gap every 5 s (data-loader stalls)."""
    return 5.0 if (t - T0) % 5.0 < 1.5 else 92.0


def run(make: Callable[[float, int], Sample], seconds: float = 300, **kw: Any) -> Diagnosis:
    h = HistoryStore()
    last = feed(h, T0, seconds, make)
    return diagnose(h, "G", now=last, **kw)


# ---- robustness ----
def test_empty_history_never_raises() -> None:
    d = diagnose(HistoryStore(), None, now=T0)
    assert d.verdict == "Collecting data" and d.score is None and d.session is None
    assert d.bottleneck == "unknown"


def test_no_gpu_gives_system_view() -> None:
    d = run(lambda t, i: smp(t, None), seconds=120)
    assert d.verdict == "No GPU; system view only"
    names = [r.name for r in d.health]
    assert names[0] == "GPU" and d.health[0].value == "n/a"
    assert {"CPU", "RAM", "Disk", "Network"} <= set(names)


def test_all_none_fields_never_raise() -> None:
    bare = GpuSample(index=0, uuid="G", name="x", vendor="amd", util=50.0)
    d = run(lambda t, i: smp(t, bare), seconds=120)
    assert isinstance(d, Diagnosis) and d.verdict != "Analysis unavailable"
    d2 = run(lambda t, i: smp(t, GpuSample(index=0, uuid="G", name="x", vendor="amd")), seconds=60)
    assert d2.verdict != "Analysis unavailable"


def test_internal_error_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object) -> Diagnosis:
        raise ValueError("bug")

    monkeypatch.setattr(intel, "_diagnose", boom)
    assert diagnose(HistoryStore(), None).verdict == "Analysis unavailable"


# ---- bottleneck rules ----
def test_idle() -> None:
    d = run(lambda t, i: smp(t, gs(2.0)))
    assert d.bottleneck == "idle" and d.confidence >= 0.6


def test_vram_held_while_idle() -> None:
    d = run(lambda t, i: smp(t, gs(2.0, mem_used_mib=7_000.0)))
    assert d.bottleneck == "vram_held"
    assert any("empty_cache" in a for a in d.actions)


def test_steady_busy_is_unconstrained_with_honest_note() -> None:
    d = run(lambda t, i: smp(t, gs(95.0)))
    assert d.bottleneck == "none" and d.verdict == "GPU kept busy, no limits seen"
    assert any("kernel ran" in e for e in d.evidence)  # utilization caveat when GPM is absent


def test_input_bound_bursty_utilization() -> None:
    d = run(lambda t, i: smp(t, gs(bursty(t))))
    assert d.bottleneck == "input" and d.verdict == "Input-bound: GPU waits on data"
    assert any("num_workers" in a for a in d.actions)
    assert any("dips/min" in e for e in d.evidence)


def test_long_pauses_are_not_input_dips() -> None:
    # 60 s busy / 30 s eval pause: low stretches longer than 15 s are pauses, not loader stalls.
    d = run(lambda t, i: smp(t, gs(95.0 if (t - T0) % 90 < 60 else 2.0)))
    assert d.bottleneck != "input"


def test_cpu_saturated_gpu_waiting() -> None:
    d = run(lambda t, i: smp(t, gs(30.0), cpu=95.0))
    assert d.bottleneck == "cpu" and d.verdict == "CPU-bound: GPU waits on CPU"


def test_single_core_pegged() -> None:
    d = run(lambda t, i: smp(t, gs(40.0), cpu=17.0, cores=(100.0,) + (5.0,) * 7))
    assert d.bottleneck == "cpu" and d.verdict == "CPU-bound: one core pegged"


def test_power_cap_from_throttle_reason() -> None:
    d = run(lambda t, i: smp(t, gs(98.0, power_w=199.0, clock_mhz=2_400.0, throttle=("power cap",))))
    assert d.bottleneck == "power" and d.throttle == ("power cap",)
    assert any("Expected at full load" in a for a in d.actions)


def test_power_cap_inferred_without_reasons() -> None:
    d = run(lambda t, i: smp(t, gs(98.0, power_w=198.0, clock_mhz=2_000.0)))
    assert d.bottleneck == "power" and d.confidence < 0.7


def test_thermal_throttle() -> None:
    d = run(lambda t, i: smp(t, gs(97.0, temp_c=86.0, clock_mhz=2_000.0, throttle=("thermal",))))
    assert d.bottleneck == "thermal"
    assert d.health[0].level == "critical"  # 86 °C >= 80 °C warning threshold


def test_memory_heuristic_is_low_confidence() -> None:
    d = run(lambda t, i: smp(t, gs(70.0, mem_util=85.0)))
    assert d.bottleneck == "memory" and d.confidence <= 0.5


def test_memory_bound_from_gpm_dram() -> None:
    d = run(lambda t, i: smp(t, gs(85.0, mem_util=10.0, sm_active=60.0, dram_bw_util=92.0)))
    assert d.bottleneck == "memory" and d.confidence >= 0.55
    assert any("GPM" in e for e in d.evidence)


def test_gpm_replaces_memory_heuristic() -> None:
    d = run(lambda t, i: smp(t, gs(70.0, mem_util=85.0, sm_active=60.0, dram_bw_util=30.0)))
    assert d.bottleneck != "memory"


def test_launch_bound_from_gpm() -> None:
    d = run(lambda t, i: smp(t, gs(95.0, sm_active=10.0, tensor_active=1.0)))
    # Renamed after review: high util + idle SMs has several causes, so the verdict stays descriptive.
    assert d.bottleneck == "launch" and d.verdict == "Low SM parallelism"
    assert d.confidence <= 0.5
    assert d.health[0].value.startswith("SM 10%")
    assert not any("SMs only" in f.title for f in d.findings)  # not repeated as a finding


def test_ram_pressure_and_swapping() -> None:
    d = run(lambda t, i: smp(t, gs(50.0), mem=96))
    assert d.bottleneck == "ram" and d.verdict == "RAM nearly full"
    d2 = run(lambda t, i: smp(t, gs(50.0), mem=80, swap_pct=5.0 + (t - T0) / 30))
    assert d2.bottleneck == "ram" and d2.verdict == "RAM pressure: system swapping"
    assert next(r for r in d2.health if r.name == "RAM").level == "serious"


def test_disk_bound_beats_generic_input() -> None:
    d = run(lambda t, i: smp(t, gs(bursty(t)), disk_read_bps=300e6, disk_write_bps=1e6))
    assert d.bottleneck == "disk" and d.runner_up == "input"
    assert any(e.startswith("Runner-up: input-bound") for e in d.evidence)
    assert next(r for r in d.health if r.name == "Disk").level == "warning"


def test_network_bound() -> None:
    d = run(lambda t, i: smp(t, gs(bursty(t)), net_rx_bps=80e6, net_tx_bps=1e6))
    assert d.bottleneck == "network"


def test_heavy_disk_without_bursty_gpu_is_not_disk_bound() -> None:
    d = run(lambda t, i: smp(t, gs(95.0), disk_read_bps=300e6))
    assert d.bottleneck == "none"


def test_went_idle_overrides_older_pattern() -> None:
    h = HistoryStore()
    feed(h, T0, 150, lambda t, i: smp(t, gs(bursty(t))))
    last = feed(h, T0 + 150, 60, lambda t, i: smp(t, gs(1.0)))
    d = diagnose(h, "G", now=last)
    assert d.bottleneck == "idle" and d.verdict == "GPU idle now"


def test_minute_stats_fallback_without_raw_samples() -> None:
    h = HistoryStore()
    feed(h, T0, 600, lambda t, i: smp(t, gs(95.0)))
    h.recent.clear()  # e.g. the dashboard just started with history loaded from disk
    d = diagnose(h, "G", now=T0 + 599)
    assert d.bottleneck == "none"
    assert any("per-minute averages" in e for e in d.evidence)


def test_verdicts_fit_the_headline() -> None:
    makers: list[Callable[[float, int], Sample]] = [
        lambda t, i: smp(t, gs(2.0)),
        lambda t, i: smp(t, gs(2.0, mem_used_mib=7_000.0)),
        lambda t, i: smp(t, gs(95.0)),
        lambda t, i: smp(t, gs(bursty(t))),
        lambda t, i: smp(t, gs(30.0), cpu=95.0),
        lambda t, i: smp(t, gs(98.0, throttle=("power cap",))),
        lambda t, i: smp(t, gs(97.0, throttle=("thermal",))),
        lambda t, i: smp(t, gs(85.0, sm_active=60.0, dram_bw_util=92.0)),
        lambda t, i: smp(t, gs(95.0, sm_active=10.0)),
        lambda t, i: smp(t, gs(50.0), mem=96),
        lambda t, i: smp(t, gs(bursty(t)), disk_read_bps=300e6),
        lambda t, i: smp(t, gs(bursty(t)), net_rx_bps=80e6),
        lambda t, i: smp(t, None),
    ]
    seen = set()
    for make in makers:
        intel._HOURLY.clear()
        d = run(make, seconds=120)
        assert 0 < len(d.verdict) <= 40, d.verdict
        assert 0.0 <= d.confidence <= 1.0
        seen.add(d.bottleneck)
    assert len(seen) >= 12


# ---- score ----
def test_score_needs_five_minutes() -> None:
    assert run(lambda t, i: smp(t, gs(95.0)), seconds=200).score is None


def test_score_basis_and_range() -> None:
    busy = run(lambda t, i: smp(t, gs(100.0)), seconds=600)
    assert busy.score is not None and busy.score >= 95 and busy.score_basis == "utilization"
    assert any("Score basis: utilization.gpu" in e for e in busy.evidence)
    idle = run(lambda t, i: smp(t, gs(0.0)), seconds=600)
    assert idle.score == 0
    held = run(lambda t, i: smp(t, gs(0.0, mem_used_mib=8_000.0)), seconds=600)
    assert held.score == 0
    bursty_d = run(lambda t, i: smp(t, gs(bursty(t))), seconds=600)
    assert bursty_d.score is not None and bursty_d.score < busy.score - 20


def test_score_prefers_sm_activity_when_available() -> None:
    d = run(lambda t, i: smp(t, gs(95.0, sm_active=20.0)), seconds=600)
    assert d.score_basis == "sm_active"
    assert any("Score basis: SM activity" in e for e in d.evidence)
    plain = run(lambda t, i: smp(t, gs(95.0)), seconds=600)
    assert d.score is not None and plain.score is not None and d.score < plain.score


def test_throttling_lowers_score() -> None:
    free = run(lambda t, i: smp(t, gs(100.0)), seconds=600)
    hot = run(lambda t, i: smp(t, gs(100.0, throttle=("thermal",))), seconds=600)
    assert free.score is not None and hot.score is not None and hot.score <= free.score - 15


# ---- sessions ----
def test_session_bridges_short_gaps_and_tracks_running() -> None:
    h = HistoryStore()
    feed(h, T0, 600, lambda t, i: smp(t, gs(90.0, mem_used_mib=6_000.0, energy_j=(t - T0) * 150)))
    feed(h, T0 + 600, 60, lambda t, i: smp(t, gs(2.0, energy_j=(t - T0) * 150)))  # 1 idle minute
    last = feed(h, T0 + 660, 300, lambda t, i: smp(t, gs(80.0, mem_used_mib=7_000.0, energy_j=(t - T0) * 150)))
    s = diagnose(h, "G", now=last).session
    assert s is not None and s.end is None
    assert s.start == T0 and 15 <= s.minutes <= 17
    assert s.peak_vram_pct == pytest.approx(70.0)
    assert s.energy_wh == pytest.approx((last - T0) * 150 / 3600, rel=0.01)
    intel._HOURLY.clear()
    end = feed(h, last + STEP, 300, lambda t, i: smp(t, gs(1.0)))
    ended = diagnose(h, "G", now=end).session
    assert ended is not None and ended.end is not None and ended.start == T0


def test_session_energy_falls_back_to_power() -> None:
    d = run(lambda t, i: smp(t, gs(90.0, power_w=120.0)), seconds=600)
    assert d.session is not None and d.session.energy_wh == pytest.approx(120 * 600 / 3600, rel=0.05)


def test_no_session_when_never_active() -> None:
    assert run(lambda t, i: smp(t, gs(3.0))).session is None


# ---- VRAM forecast ----
def test_vram_forecast_rising() -> None:
    d = run(lambda t, i: smp(t, gs(90.0, mem_used_mib=5_000.0 + (t - T0) * 5)), seconds=240)
    # 5 MiB/s of 10 000 MiB = 3 %/min; ends near 62% -> ~13 min to full.
    assert d.vram_forecast is not None and d.vram_forecast.startswith("VRAM full in ~")
    assert d.vram_eta_min is not None and 10 <= d.vram_eta_min <= 15
    assert next(r for r in d.health if r.name == "VRAM").level == "warning"


def test_vram_forecast_flat_or_slow_is_none() -> None:
    assert run(lambda t, i: smp(t, gs(90.0, mem_used_mib=5_000.0))).vram_forecast is None
    slow = run(lambda t, i: smp(t, gs(90.0, mem_used_mib=5_000.0 + (t - T0) * 0.2)), seconds=240)
    assert slow.vram_forecast is None


# ---- findings ----
def titles(d: Diagnosis) -> list[str]:
    return [f.title for f in d.findings]


def test_collapse_with_memory_still_held() -> None:
    h = HistoryStore()
    feed(h, T0, 240, lambda t, i: smp(t, gs(92.0, mem_used_mib=6_000.0)))
    last = feed(h, T0 + 240, 60, lambda t, i: smp(t, gs(0.0, mem_used_mib=6_000.0)))
    d = diagnose(h, "G", now=last)
    assert "GPU activity stopped suddenly" in titles(d)


def test_collapse_ignored_when_memory_released() -> None:
    h = HistoryStore()
    feed(h, T0, 240, lambda t, i: smp(t, gs(92.0, mem_used_mib=6_000.0)))
    last = feed(h, T0 + 240, 60, lambda t, i: smp(t, gs(0.0, mem_used_mib=300.0)))
    assert "GPU activity stopped suddenly" not in titles(diagnose(h, "G", now=last))


def test_low_clocks_without_reason() -> None:
    assert "Clocks low while busy" in titles(run(lambda t, i: smp(t, gs(90.0, clock_mhz=1_000.0))))
    assert "Clocks low while busy" not in titles(
        run(lambda t, i: smp(t, gs(90.0, clock_mhz=1_000.0, throttle=("power cap",))))
    )


def test_temperature_climbing() -> None:
    d = run(lambda t, i: smp(t, gs(90.0, temp_c=55.0 + (t - T0) / 12)), seconds=180)  # +5 °C/min to 70 °C
    assert any(x.startswith("Temperature climbing +5.0") for x in titles(d))


def test_tensor_cores_idle_and_low_occupancy() -> None:
    d = run(lambda t, i: smp(t, gs(92.0, sm_active=70.0, fp32_active=50.0, tensor_active=0.0, sm_occupancy=10.0)))
    assert "Tensor Cores idle during FP32 work" in titles(d)
    assert "Low SM occupancy (10%)" in titles(d)


def test_pcie_link_and_heavy_transfers() -> None:
    det = GpuDetails(index=0, uuid="G", name="x", pcie_max_gen=5, pcie_max_width=16)
    d = run(lambda t, i: smp(t, gs(bursty(t), pcie_gen=3, pcie_width=8, pcie_rx_bps=5e9)), details=det)
    t = titles(d)
    assert "PCIe link Gen3 x8 (GPU supports Gen5 x16)" in t
    assert "Host→GPU transfers heavy while the GPU waits" in t
    idle = run(lambda t, i: smp(t, gs(2.0, pcie_gen=1, pcie_width=8)), details=det)
    assert not any(x.startswith("PCIe link") for x in titles(idle))  # idle downtraining is normal


def test_system_findings() -> None:
    d = run(lambda t, i: smp(t, gs(bursty(t)), cpu=17.0, cores=(100.0,) + (5.0,) * 7, mem=93, cpu_temp_c=92.0))
    t = titles(d)
    assert "RAM 93% used" in t or d.bottleneck == "ram"
    assert "CPU running hot: 92 °C" in t
    assert any("core pegged" in x for x in t) or d.bottleneck == "cpu"


# ---- health ----
def test_health_rows_and_levels() -> None:
    d = run(
        lambda t, i: smp(
            t,
            gs(95.0, mem_used_mib=9_700.0),
            cpu=90.0,
            mem=97,
            swap_pct=1.0,
            cpu_freq_mhz=4_200.0,
            cpu_temp_c=70.0,
            disk_read_bps=2e6,
            disk_write_bps=1e6,
            net_rx_bps=5e5,
            net_tx_bps=1e5,
        )
    )
    rows = {r.name: r for r in d.health}
    assert list(rows) == ["GPU", "VRAM", "CPU", "CPU temp", "RAM", "Disk", "Network"]
    assert rows["VRAM"].level == "serious"
    assert rows["CPU"].level == "warning" and rows["CPU"].value == "90% @ 4.2 GHz"
    assert rows["RAM"].level == "critical" and rows["RAM"].value == "97% · swap 1%"
    assert rows["Disk"].value == "2 MB/s read · 1 MB/s write"
    assert rows["Network"].value == "500 KB/s down · 100 KB/s up"
    assert all(r.level in ("good", "warning", "serious", "critical", "info") for r in d.health)


# ---- demo exercises the engine ----
def test_demo_cycle_hits_every_phase() -> None:
    h = HistoryStore()
    t0 = demo._T0
    expected = {190.0: "none", 390.0: "input", 590.0: "disk", 790.0: "launch", 990.0: "power", 1075.0: "idle"}
    got = {}
    for i in range(int(1080 / STEP)):
        t = t0 + i * STEP
        x = demo.demo_system(t)
        h.add(smp(t, demo.demo_gpu(t), disk_read_bps=x.disk_read_bps, net_rx_bps=x.net_rx_bps))
        if i * STEP in expected:
            intel._HOURLY.clear()
            got[i * STEP] = diagnose(h, demo.DEMO_UUID, now=t, details=demo.demo_details()).bottleneck
    assert got == expected


# ---- performance ----
def test_full_hour_is_fast() -> None:
    h = HistoryStore()
    last = feed(
        h,
        T0,
        3600,
        lambda t, i: smp(t, gs(bursty(t), sm_active=50.0, throttle=("power cap",) if i % 7 else ()), disk_read_bps=1e6),
    )
    assert len(h.recent) == 7200
    intel._HOURLY.clear()
    a = time.perf_counter()
    diagnose(h, "G", now=last)
    cold = (time.perf_counter() - a) * 1000
    warm_runs = []
    for _ in range(20):
        a = time.perf_counter()
        diagnose(h, "G", now=last)
        warm_runs.append((time.perf_counter() - a) * 1000)
    warm = sorted(warm_runs)[10]
    print(f"\ndiagnose() on 7200 samples: cold {cold:.2f} ms, warm median {warm:.2f} ms")
    # The UI calls diagnose() every 2 s, so the warm median is the real budget (~1 ms locally).
    # The one-off cold call only guards against gross regressions: shared CI runners are slow and
    # it measured 54.7 ms once on Python 3.9 here, which made a 50 ms bound flaky.
    assert warm < 20
    assert cold < 500
