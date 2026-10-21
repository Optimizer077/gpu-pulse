"""Alert engine: the four rules, hysteresis, cooldown, gap resets and None handling (synthetic samples)."""

from __future__ import annotations

from typing import Optional

from gpu_pulse import demo
from gpu_pulse.alerts import Alert, AlertConfig, AlertEngine, format_line
from gpu_pulse.models import GpuSample, Sample

TOTAL = 16384.0  # MiB = 16 GB


def gpu(
    i: int = 0,
    util: Optional[float] = 50.0,
    temp: Optional[float] = 60.0,
    vram: Optional[float] = 25.0,  # % of TOTAL
    uuid: Optional[str] = None,
) -> GpuSample:
    used = None if vram is None else TOTAL * vram / 100
    return GpuSample(
        i, uuid or f"GPU-{i}", "test", "nvidia", util=util, temp_c=temp, mem_used_mib=used, mem_total_mib=TOTAL
    )


def sample(t: float, *gpus: GpuSample) -> Sample:
    return Sample(t=t, gpus=gpus, cpu_util=0.0, cpu_per_core=(), mem_used=0, mem_total=1)


def feed(eng: AlertEngine, t0: float, t1: float, *gpus: GpuSample, dt: float = 1.0) -> list[Alert]:
    """Same GPU readings every ``dt`` s for t0 <= t < t1; all alerts fired."""
    out: list[Alert] = []
    t = t0
    while t < t1:
        out += eng.update(sample(t, *gpus))
        t += dt
    return out


# ---- hot ----------------------------------------------------------------------------------
def test_hot_fires_after_30s_once() -> None:
    eng = AlertEngine()
    assert feed(eng, 0, 30, gpu(temp=85)) == []  # t = 0 .. 29: held 29 s
    fired = eng.update(sample(30, gpu(temp=85)))
    assert [(a.rule, a.level, a.gpu_index, a.t) for a in fired] == [("hot", "warning", 0, 30)]
    assert fired[0].title == "GPU 0 running hot: 85 °C"
    assert feed(eng, 31, 600, gpu(temp=88)) == []  # stays hot: no refire
    assert eng.active() == [("hot", "GPU-0")]


def test_hot_threshold_is_inclusive_and_follows_temp_warn() -> None:
    eng = AlertEngine(AlertConfig(temp_warn=70))
    assert [a.rule for a in feed(eng, 0, 31, gpu(temp=70))] == ["hot"]
    assert feed(AlertEngine(), 0, 100, gpu(temp=79.9)) == []


def test_hot_hysteresis_rearms_only_below_warn_minus_3() -> None:
    eng = AlertEngine(AlertConfig(cooldown_s=0))
    assert len(feed(eng, 0, 31, gpu(temp=85))) == 1
    feed(eng, 31, 100, gpu(temp=78))  # under 80 but not under 77: still latched
    assert eng.active() == [("hot", "GPU-0")]
    assert feed(eng, 100, 200, gpu(temp=85)) == []  # never re-armed
    feed(eng, 200, 205, gpu(temp=76))  # clears
    assert eng.active() == []
    assert len(feed(eng, 205, 240, gpu(temp=85))) == 1  # re-armed, fires again after 30 s


def test_cooldown_suppresses_refire_and_latches_silently() -> None:
    eng = AlertEngine()
    assert len(feed(eng, 0, 31, gpu(temp=85))) == 1
    feed(eng, 31, 40, gpu(temp=60))
    assert feed(eng, 40, 1000, gpu(temp=85)) == []  # true again inside the 30 min cooldown
    assert eng.active() == [("hot", "GPU-0")]  # latched without notifying ...
    assert feed(eng, 1000, 2500, gpu(temp=85)) == []  # ... and not delivered when the cooldown ends
    feed(eng, 2500, 2510, gpu(temp=60))
    fired = feed(eng, 2510, 2600, gpu(temp=85))
    assert [a.t for a in fired] == [2540]  # cooldown over (last fire at 30): fires normally


def test_cooldown_is_per_rule_and_per_gpu() -> None:
    eng = AlertEngine()
    a = feed(eng, 0, 31, gpu(0, temp=85, vram=97), gpu(1, temp=85))
    assert sorted((x.rule, x.gpu_uuid) for x in a) == [("hot", "GPU-0"), ("hot", "GPU-1"), ("vram_full", "GPU-0")]


# ---- gaps and None --------------------------------------------------------------------------
def test_sample_gap_over_3x_interval_resets_the_timer() -> None:
    eng = AlertEngine(AlertConfig(interval_s=1.0))
    feed(eng, 0, 20, gpu(temp=85))  # 19 s held
    assert feed(eng, 24, 50, gpu(temp=85)) == []  # 5 s gap (> 3 s): restarts at 24, fires only at 54
    assert [a.t for a in feed(eng, 50, 60, gpu(temp=85))] == [54]


def test_gap_of_exactly_3x_interval_keeps_the_timer() -> None:
    eng = AlertEngine(AlertConfig(interval_s=1.0))
    feed(eng, 0, 20, gpu(temp=85))
    assert [a.t for a in feed(eng, 22, 40, gpu(temp=85))] == [30]  # 19 -> 22 is a 3 s gap


def test_backwards_clock_resets_the_timer() -> None:
    eng = AlertEngine()
    feed(eng, 100, 125, gpu(temp=85))
    assert feed(eng, 50, 75, gpu(temp=85)) == []


def test_none_never_satisfies_or_clears() -> None:
    eng = AlertEngine()
    for t in range(0, 120):  # a None every 20 s breaks "continuously" for 30 s
        assert eng.update(sample(t, gpu(temp=None if t % 20 == 0 else 85))) == []
    eng2 = AlertEngine()
    feed(eng2, 0, 31, gpu(temp=85))
    feed(eng2, 31, 100, gpu(temp=None, util=None, vram=None))  # None does not clear either
    assert eng2.active() == [("hot", "GPU-0")]
    assert feed(AlertEngine(), 0, 2000, gpu(util=None, temp=None, vram=None)) == []


def test_missing_total_means_no_vram_rules() -> None:
    g = GpuSample(0, "GPU-0", "x", "nvidia", util=1, mem_used_mib=15000, mem_total_mib=None)
    assert feed(AlertEngine(AlertConfig(idle_held_min=1)), 0, 200, g) == []


# ---- multi-GPU ----------------------------------------------------------------------------
def test_gpus_are_independent_and_keyed_by_uuid() -> None:
    eng = AlertEngine()
    fired = feed(eng, 0, 31, gpu(0, temp=85, uuid="A"), gpu(1, temp=60, uuid="B"))
    assert [(a.gpu_uuid, a.gpu_index) for a in fired] == [("A", 0)]
    # Indices swap (driver re-enumeration): state follows the uuid, so A doesn't refire and B starts fresh.
    assert feed(eng, 31, 61, gpu(1, temp=85, uuid="A"), gpu(0, temp=85, uuid="B")) == []
    fired = eng.update(sample(61, gpu(0, temp=85, uuid="B")))
    assert [(a.gpu_uuid, a.gpu_index, a.t) for a in fired] == [("B", 0, 61)]
    assert eng.active() == [("hot", "B")]  # only GPUs in the latest sample count as active


def test_one_gpu_dropping_out_does_not_reset_another() -> None:
    eng = AlertEngine()
    feed(eng, 0, 20, gpu(0, temp=85), gpu(1, temp=85))
    fired = feed(eng, 20, 31, gpu(0, temp=85))  # GPU 1 vanished; GPU 0 keeps its 20 s
    assert [(a.gpu_index, a.t) for a in fired] == [(0, 30)]


# ---- vram_full ------------------------------------------------------------------------------
def test_vram_full_fires_after_20s_with_sizes() -> None:
    eng = AlertEngine()
    fired = feed(eng, 0, 25, gpu(vram=97))
    assert [(a.rule, a.t) for a in fired] == [("vram_full", 20)]
    assert fired[0].title == "GPU 0 VRAM nearly full (15.5/16 GB)"


def test_vram_full_clears_below_90_only() -> None:
    eng = AlertEngine(AlertConfig(cooldown_s=0))
    feed(eng, 0, 21, gpu(vram=96))
    feed(eng, 21, 60, gpu(vram=92))
    assert feed(eng, 60, 120, gpu(vram=96)) == []
    feed(eng, 120, 125, gpu(vram=85))
    assert [a.t for a in feed(eng, 125, 150, gpu(vram=96))] == [145]


# ---- idle_held ------------------------------------------------------------------------------
def test_idle_held_fires_after_configured_minutes() -> None:
    eng = AlertEngine(AlertConfig(interval_s=5))  # 15 min default; `log` samples every 5 s
    fired = feed(eng, 0, 15 * 60 + 5, gpu(util=3, vram=60), dt=5)
    assert [(a.rule, a.level, a.t) for a in fired] == [("idle_held", "info", 900)]
    assert fired[0].title == "GPU 0 idle for 15 min while holding 9.6 GB of VRAM"


def test_idle_held_needs_both_conditions() -> None:
    cfg = AlertConfig(idle_held_min=1)
    assert feed(AlertEngine(cfg), 0, 300, gpu(util=3, vram=45)) == []  # not enough VRAM held
    assert feed(AlertEngine(cfg), 0, 300, gpu(util=10, vram=80)) == []  # 10% is not idle


def test_idle_held_clears_on_util_15_or_vram_40() -> None:
    eng = AlertEngine(AlertConfig(idle_held_min=1, cooldown_s=0))
    assert len(feed(eng, 0, 61, gpu(util=3, vram=60))) == 1
    feed(eng, 61, 70, gpu(util=12, vram=60))  # neither idle nor cleared
    assert eng.active() == [("idle_held", "GPU-0")]
    feed(eng, 70, 72, gpu(util=15, vram=60))
    assert eng.active() == []
    assert len(feed(eng, 72, 140, gpu(util=3, vram=60))) == 1
    feed(eng, 140, 142, gpu(util=3, vram=39))
    assert eng.active() == []


# ---- job_finished ---------------------------------------------------------------------------
def test_job_finished_after_long_busy_stretch() -> None:
    eng = AlertEngine()  # 10 min default
    assert feed(eng, 0, 660, gpu(util=80)) == []  # busy 0 .. 659 (11 min)
    fired = feed(eng, 660, 800, gpu(util=2))
    assert [(a.rule, a.level, a.t) for a in fired] == [("job_finished", "info", 720)]  # quiet 660 -> 720
    assert fired[0].title == "Job on GPU 0 finished after 11 min (avg 80%)"
    assert feed(eng, 800, 2000, gpu(util=2)) == []  # once per stretch


def test_short_job_is_not_reported() -> None:
    eng = AlertEngine()
    feed(eng, 0, 300, gpu(util=90))
    assert feed(eng, 300, 500, gpu(util=1)) == []


def test_short_dips_are_bridged_and_averaged() -> None:
    eng = AlertEngine(AlertConfig(cooldown_s=0))
    alerts: list[Alert] = []
    for k in range(12):  # 12 × (40 s busy at 90% + 20 s dip at 0%): one 11.7 min stretch
        alerts += feed(eng, k * 60, k * 60 + 40, gpu(util=90))
        alerts += feed(eng, k * 60 + 40, k * 60 + 60, gpu(util=0))
    assert alerts == []  # 20 s quiet never reaches 60 s
    fired = feed(eng, 720, 800, gpu(util=0))
    assert len(fired) == 1
    # 11 full cycles + the last 40 s busy: (12 × 40 × 90) / 700 samples = 61.7%; the trailing dip doesn't count.
    assert fired[0].title == "Job on GPU 0 finished after 12 min (avg 62%)"


def test_lukewarm_tail_without_quiet_drops_the_stretch() -> None:
    eng = AlertEngine()
    feed(eng, 0, 700, gpu(util=80))
    feed(eng, 700, 780, gpu(util=12))  # neither busy nor quiet for > 60 s
    assert feed(eng, 780, 900, gpu(util=2)) == []


def test_long_sample_gap_drops_the_stretch() -> None:
    eng = AlertEngine()
    feed(eng, 0, 700, gpu(util=80))
    assert feed(eng, 800, 900, gpu(util=2)) == []  # 100 s without samples (laptop asleep, logger stopped)


def test_job_threshold_is_configurable() -> None:
    eng = AlertEngine(AlertConfig(job_min=2))
    feed(eng, 0, 125, gpu(util=50))
    assert [a.rule for a in feed(eng, 125, 200, gpu(util=0))] == ["job_finished"]


# ---- demo and formatting --------------------------------------------------------------------
def test_four_gpu_demo_fires_hot_on_gpu_2_within_a_minute() -> None:
    eng = AlertEngine(AlertConfig(interval_s=0.5))
    fired: list[Alert] = []
    for k in range(120):  # the first 60 s of the demo at 0.5 s per sample
        t = demo._T0 + k * 0.5
        fired += eng.update(sample(t, *demo.demo_gpus(t, 4)))
    assert [(a.rule, a.gpu_index) for a in fired] == [("hot", 2)]
    assert 30 <= fired[0].t - demo._T0 <= 31
    assert "running hot: 8" in fired[0].title


def test_format_line() -> None:
    a = Alert("hot", "GPU-0", 0, "warning", "GPU 0 running hot: 85 °C", "detail", 1.7e9)
    line = format_line(a)
    assert line.endswith("  WARNING  GPU 0 running hot: 85 °C")
    assert len(line.split("  ")[0]) == len("2026-01-01 00:00:00")
