"""The diagnosis must not contradict the live reading right after start (review of v0.2 screenshots)."""

from __future__ import annotations

import time

from gpu_pulse.history import HistoryStore
from gpu_pulse.intel import diagnose
from gpu_pulse.models import GpuSample, Sample


def _gpu(util: float, vram_mib: float = 9000) -> GpuSample:
    return GpuSample(
        index=0,
        uuid="G",
        name="Test",
        vendor="nvidia",
        util=util,
        mem_used_mib=vram_mib,
        mem_total_mib=10_000,
        temp_c=60,
        power_w=100,
        power_limit_w=200,
    )


def _sample(t: float, util: float) -> Sample:
    return Sample(t=t, gpus=(_gpu(util),), cpu_util=10, cpu_per_core=(10,), mem_used=1, mem_total=2)


def _mid_minute() -> float:
    """A 'now' 30 s into the current minute, so the test never straddles a minute boundary."""
    return (time.time() // 60) * 60 + 30


def test_idle_minutes_do_not_override_a_busy_live_reading() -> None:
    now = _mid_minute()
    h = HistoryStore()
    m0 = int(now // 60 * 60)
    # Five recorded minutes of an idle GPU holding VRAM (as per-minute buckets) ...
    for k in range(5, 0, -1):
        for i in range(6):
            h.add(_sample(m0 - k * 60 + i * 10, 2.0))
    # ... then the job starts: only a few raw samples so far (too few for the raw-sample rules).
    for i in range(3):
        h.add(_sample(now - 1.5 + i * 0.5, 95.0))
    d = diagnose(h, "G", now=now)
    assert d.bottleneck not in ("idle", "vram_held"), d.verdict


def test_idle_minutes_still_reported_when_live_reading_is_idle() -> None:
    now = _mid_minute()
    h = HistoryStore()
    m0 = int(now // 60 * 60)
    for k in range(5, 0, -1):
        for i in range(6):
            h.add(_sample(m0 - k * 60 + i * 10, 2.0))
    for i in range(3):
        h.add(_sample(now - 1.5 + i * 0.5, 1.0))
    d = diagnose(h, "G", now=now)
    assert d.bottleneck in ("idle", "vram_held")
