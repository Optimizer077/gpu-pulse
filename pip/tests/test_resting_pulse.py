"""The pulse strip scrolls slowly (and repaints rarely) while the GPU is idle."""

from __future__ import annotations

from gpu_pulse.widgets import PulseStrip


def _moves(util: float, seconds: float = 2.0, fps: float = 12.0) -> int:
    p = PulseStrip()
    p.refresh = lambda *a, **k: None  # type: ignore[method-assign]  # not mounted in this unit test
    p.util = util
    for _ in range(50):  # let the smoothed level settle at this utilization
        p.wave.advance(util, 1)
    return sum(p.step(1 / fps) for _ in range(int(seconds * fps)))


def test_idle_gpu_repaints_a_few_times_a_second() -> None:
    assert _moves(0.0) <= 2 * 5  # ~4 scroll steps/s at rest


def test_busy_gpu_scrolls_every_frame() -> None:
    assert _moves(90.0) >= 2 * 11


def test_speed_rises_with_utilization() -> None:
    p = PulseStrip()
    speeds = []
    for u in (0.0, 10.0, 20.0, 40.0, 90.0):
        p.util = u
        speeds.append(p.columns_per_s())
    assert speeds == sorted(speeds)
    assert speeds[0] == PulseStrip.RESTING_COLUMNS_PER_S
    assert speeds[-1] == PulseStrip.COLUMNS_PER_S
