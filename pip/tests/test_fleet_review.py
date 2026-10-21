"""Regressions for the fourth code review: cross-GPU findings, data span, PCIe card limits, energy timing."""

from __future__ import annotations

import pytest

from gpu_pulse import demo, intel
from gpu_pulse.fleet import analyze_fleet
from gpu_pulse.fleetview import summary_line
from gpu_pulse.history import HistoryStore
from gpu_pulse.models import GpuDetails, GpuProcess, Sample
from tests.test_intel import gs, smp
from tests.test_multigpu import T0, fleet_of, gpu, store


def _titles(f: object) -> list[str]:
    return [i.title for i in f.findings]  # type: ignore[attr-defined]


# ---- HIGH 1: GPUs that don't share one job --------------------------------------------------
def _busy_and_idle(k: int) -> tuple:
    return (gpu(0, 95, mem_used_mib=14_080.0, throttle=("power cap",)), gpu(1, 0, mem_used_mib=320.0))


@pytest.mark.parametrize("procs", [None, [GpuProcess(1, "python", 13_000.0, "GPU-0")]])
def test_idle_gpu_next_to_busy_one_is_not_a_straggler(procs: object) -> None:
    # The review's reproduction: GPU 0 at 95% and power-capped, GPU 1 idle with ~no VRAM.
    f = analyze_fleet(store(_busy_and_idle), now=T0 + 60, processes=procs)  # type: ignore[arg-type]
    assert _titles(f) == ["GPU 1 idle while GPU 0 works"]
    (ins,) = f.findings
    assert ins.level == "info" and "not taking part" in ins.detail and "If these GPUs should run one job" in ins.detail
    assert all(flag.level == "good" and not flag.tags for flag in f.flags.values())  # no blame on either GPU


def test_gpus_running_different_processes_are_not_compared() -> None:
    procs = [GpuProcess(1, "python", 13_000.0, "GPU-0"), GpuProcess(2, "ollama", 6_000.0, "GPU-1")]
    f = analyze_fleet(
        store(lambda k: (gpu(0, 95, mem_used_mib=14_080.0, throttle=("power cap",)), gpu(1, 5, mem_used_mib=6_400.0))),
        now=T0 + 60,
        processes=procs,
    )
    assert f.findings == ()  # in use, but unrelated jobs: no straggler, no uneven VRAM, no lone power cap


def test_shared_process_straggler_is_at_most_a_warning() -> None:
    procs = [GpuProcess(i, "python", 6_000.0, f"GPU-{i}") for i in range(3)]
    f = analyze_fleet(store(lambda k: (gpu(0, 92), gpu(1, 3), gpu(2, 90))), now=T0 + 60, processes=procs)
    (ins,) = f.findings
    assert ins.level == "warning" and "run python" in ins.detail


def test_lone_throttle_needs_busy_siblings() -> None:
    # GPU 1 holds VRAM (so it could share the job) but is not busy: GPU 0's power cap isn't "alone".
    f = fleet_of(lambda k: (gpu(0, 95, throttle=("power cap",)), gpu(1, 20)))
    assert not any("power-capped" in t for t in _titles(f))
    busy = fleet_of(lambda k: (gpu(0, 95, throttle=("power cap",)), gpu(1, 90)))
    (ins,) = busy.findings
    assert ins.title == "GPU 0 power-capped while its siblings aren't"
    assert "uneven split" not in ins.detail and "If these GPUs run one job" in ins.detail


def test_every_cross_gpu_finding_is_conditional() -> None:
    t = demo._T0 + 30
    h = HistoryStore()
    for k in range(80):
        h.add(Sample(t=t + k * 0.5, gpus=demo.demo_gpus(t + k * 0.5, 4), cpu_util=10, cpu_per_core=(10,),
                     mem_used=1, mem_total=4))  # fmt: skip
    f = analyze_fleet(h, demo.demo_details_all(4), now=t + 40, processes=demo.demo_processes(demo.demo_gpus(t, 4)))
    assert len(f.findings) >= 4
    for ins in f.findings:
        assert "if these gpus" in ins.detail.lower(), ins.title
        assert ins.level in ("info", "warning")  # never serious: the GPUs may run unrelated jobs
    for ins in f.findings:
        if "lags" in ins.title or "VRAM uneven" in ins.title:
            assert "if they run one job" in ins.title


# ---- MEDIUM 2: enough data, and the real span in the wording ---------------------------------
def test_fleet_waits_for_30_seconds_and_states_the_real_span() -> None:
    make = lambda k: (gpu(0, 92), gpu(1, 6), gpu(2, 90))  # noqa: E731
    short = analyze_fleet(store(make, seconds=25), now=T0 + 25)  # 50 samples, but only 25 s
    assert short.findings == () and short.stats == ()
    assert "after 30 s" in summary_line((gpu(0, 92), gpu(1, 6)), short, 120).plain
    mid = analyze_fleet(store(make, seconds=47.5), now=T0 + 48)
    assert "over the last 47 s" in mid.findings[0].detail
    full = analyze_fleet(store(make, seconds=120), now=T0 + 120)
    assert "over the last 2 min" in full.findings[0].detail
    balanced = analyze_fleet(store(lambda k: (gpu(0, 90), gpu(1, 91)), seconds=40), now=T0 + 40)
    assert "in the last 40 s" in summary_line((gpu(0, 90), gpu(1, 91)), balanced, 120).plain


# ---- MEDIUM 3: a card that is x8 / Gen3 by design --------------------------------------------
def _details(*maxes: tuple[int, int]) -> list[GpuDetails]:
    return [GpuDetails(i, f"GPU-{i}", "x", pcie_max_gen=g, pcie_max_width=w) for i, (g, w) in enumerate(maxes)]


def test_pcie_card_limit_is_info_not_a_slot_warning() -> None:
    x8 = fleet_of(lambda k: (gpu(0, 90), gpu(1, 90, pcie_width=8)), _details((4, 16), (4, 8)))
    (ins,) = x8.findings
    assert ins.level == "info" and "card limit (x8 by design)" in ins.title and "not the slot" in ins.detail
    assert x8.flags["GPU-1"].tags == ()
    gen3 = fleet_of(lambda k: (gpu(0, 90), gpu(1, 90, pcie_gen=3)), _details((4, 16), (3, 16)))
    (ins,) = gen3.findings
    assert ins.level == "info" and "card limit (Gen3 by design)" in ins.title
    # The same x8 link on a card that can do x16: the slot (warning).
    slot = fleet_of(lambda k: (gpu(0, 90), gpu(1, 90, pcie_width=8)), _details((4, 16), (4, 16)))
    (ins,) = slot.findings
    assert ins.level == "warning" and "supports x16" in ins.detail and slot.flags["GPU-1"].tags == ("x8 link",)
    # No static details: can't tell card from slot, so never more than info.
    unknown = fleet_of(lambda k: (gpu(0, 90), gpu(1, 90, pcie_width=8)))
    assert [i.level for i in unknown.findings] == ["info"]


# ---- LOW 9: cached energy counter keeps its read time ----------------------------------------
def _cached_energy_rows(seconds: float, every: float = 2.0) -> list:
    """Samples every 0.5 s at 150 W; the counter is read every ``every`` s and carried until the next read."""
    rows = []
    for k in range(int(seconds / 0.5)):
        t = T0 + k * 0.5
        read = T0 + (k * 0.5 // every) * every
        g = gs(90.0, power_w=150.0, energy_j=1_000.0 + (read - T0) * 150, energy_t=read)
        rows.append((smp(t, g), g))
    return rows


def test_session_energy_counts_the_seconds_after_the_last_counter_read() -> None:
    rows = _cached_energy_rows(60)  # last sample at +59.5 s carries the reading taken at +58 s
    assert intel._energy(rows, T0, T0, 0.0) == pytest.approx(59.5 * 150 / 3600, abs=0.01)


def test_short_session_with_one_counter_read_falls_back_to_power() -> None:
    rows = _cached_energy_rows(1.5)  # three samples, one shared counter reading
    assert intel._energy(rows, T0, T0, 0.0) == pytest.approx(1.0 * 150 / 3600, abs=0.01)


def test_nvml_energy_carries_its_read_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpu_pulse.collectors import nvidia
    from tests.test_nvidia_throttle import make

    ticks = [100.0]

    def clock() -> float:  # wall clock that moves 0.5 s per call
        ticks[0] += 0.5
        return ticks[0]

    col, _ = make(monkeypatch, nvmlDeviceGetTotalEnergyConsumption=lambda h: 2_000_000)
    monkeypatch.setattr(nvidia.time, "time", clock)
    a, b = col.sample()[0], col.sample()[0]  # second sample reuses the cached counter
    assert a.energy_j == b.energy_j == 2000.0
    assert a.energy_t is not None and a.energy_t == b.energy_t  # timed at the one read, not per sample
