"""History aggregation, persistence and insights (mirrors the extension's tests)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from gpu_pulse.history import HistoryFile, HistoryStore
from gpu_pulse.insights import build_insights, grade
from gpu_pulse.models import GpuSample, Sample

T0 = 1_792_576_800.0  # 2026-10-21T10:00:00Z, minute-aligned


def gpu(util: float, used: float = 1000, temp: float = 60) -> GpuSample:
    return GpuSample(
        index=0,
        uuid="G",
        name="Test",
        vendor="nvidia",
        util=util,
        mem_used_mib=used,
        mem_total_mib=10_000,
        temp_c=temp,
        power_w=100,
        power_limit_w=200,
    )


def sample(t: float, util: float, used: float = 1000, cpu: float = 20) -> Sample:
    return Sample(t=t, gpus=(gpu(util, used),), cpu_util=cpu, cpu_per_core=(cpu,), mem_used=50, mem_total=100)


def fill(h: HistoryStore, start: float, minutes: float, util, used: float = 1000) -> None:  # noqa: ANN001
    for i in range(int(minutes * 6)):
        h.add(sample(start + i * 10, util(i), used))


def at(minute: int) -> float:
    return T0 + minute * 60 - 1


def test_weighted_mean_and_shares() -> None:
    h = HistoryStore()
    fill(h, T0, 10, lambda i: 100 if i % 2 else 0)
    s = h.stats("10m", 10, "G", at(10))
    assert s.covered_minutes == 10
    assert s.util == pytest.approx(50)
    assert s.util_peak == 100
    assert s.idle_share == pytest.approx(0.5)
    assert s.busy_share == pytest.approx(0.5)


def test_window_bounds() -> None:
    h = HistoryStore()
    fill(h, T0, 60, lambda i: 0 if i < 180 else 80)
    assert h.stats("30m", 30, "G", at(60)).util == pytest.approx(80)
    assert h.stats("1h", 60, "G", at(60)).util == pytest.approx(40)


def test_held_idle() -> None:
    h = HistoryStore()
    fill(h, T0, 5, lambda i: 2, used=6000)
    assert h.stats("5m", 5, "G", at(5)).held_idle_share == 1


def test_restart_same_minute_merges(tmp_path: Path) -> None:
    f = HistoryFile(tmp_path / "h.json")
    a = HistoryStore()
    fill(a, T0, 1, lambda i: 20)  # 6 samples, minute open
    f.save(a)
    b = HistoryStore()
    b.load(f.read())
    b.add(sample(T0 + 55, 80))
    b.add(sample(T0 + 61, 0))  # closes minute 0
    # Buckets store means rounded to 2 decimals.
    assert b.stats("1m", 1, "G", T0 + 59).util == pytest.approx((20 * 6 + 80) / 7, abs=0.01)


def test_save_merges_other_writer_and_respects_clear(tmp_path: Path) -> None:
    f = HistoryFile(tmp_path / "h.json")
    other = HistoryStore()
    fill(other, T0 - 600, 1, lambda i: 90)
    f.save(other)
    mine = HistoryStore()
    fill(mine, T0, 2, lambda i: 10)
    f.save(mine)  # merges the other writer's minute
    assert mine.stats("1h", 60, "G", at(2)).util_peak == 90
    data = json.loads(f.path.read_text())
    assert len(data["minutes"]) >= 3

    clearer = HistoryStore()
    clearer.clear(int((T0 + 180) * 1000))
    f.save(clearer)
    mine.add(sample(T0 + 200, 50))
    f.save(mine)  # sees the newer clear
    assert mine.stats("1h", 60, "G", at(3)).covered_minutes <= 1


def test_heatmap_and_csv() -> None:
    h = HistoryStore()
    fill(h, T0, 2, lambda i: 50)
    start, cells = h.heatmap("G", 1, now=T0 + 59 * 60)
    assert len(cells) == 60 and cells[0] == 50 and cells[1] == 50 and cells[2] is None
    lines = h.to_csv().strip().splitlines()
    assert lines[1].startswith("2026-10-21T10:00:00Z,G,6,50")


def test_load_ignores_garbage() -> None:
    h = HistoryStore()
    h.load({"version": 1, "minutes": [{"t": 1}, None, {"t": T0 * 1000, "n": 1, "cpu": 1, "mem": 1, "gpus": {}}]})
    assert len(h.minutes) == 1
    h.load("not json")  # ignored


@pytest.mark.parametrize("util,level", [(None, None), (5, "idle"), (35, "low"), (70, "good"), (95, "max")])
def test_grade(util: Optional[float], level: Optional[str]) -> None:
    g = grade(util)
    assert (g[1] if g else None) == level


def test_insights_held_and_idle() -> None:
    h = HistoryStore()
    fill(h, T0, 30, lambda i: 1, used=8000)
    titles = [
        i.title
        for i in build_insights(h.stats("1h", 60, "G", at(30)), h.stats("30m", 30, "G", at(30)), gpu(1, 8000), 80)
    ]
    assert "Memory held while GPU idles" in titles
    assert "GPU mostly idle" in titles


def test_insights_heat_and_full_vram() -> None:
    h = HistoryStore()
    levels = {i.level for i in build_insights(h.stats("1h", 60, "G"), h.stats("30m", 30, "G"), gpu(90, 9800, 85), 80)}
    assert {"critical", "serious"} <= levels
