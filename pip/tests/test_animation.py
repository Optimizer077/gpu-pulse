"""CPU-cost behaviour of the animated widgets: settling, partial repaints, paused frames."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from rich.text import Text
from textual.app import App, ComposeResult

from gpu_pulse import render as R
from gpu_pulse import strips, widgets
from gpu_pulse.app import GpuDashboard
from gpu_pulse.monitor import Monitor, MonitorConfig
from gpu_pulse.widgets import QuietStatic, RingGauge


class GaugeApp(App[None]):
    CSS = "RingGauge { height: 15; width: 40; }"

    def compose(self) -> ComposeResult:
        yield RingGauge("GPU", R.GPU, R.GPU_GLOW, id="g")


def _settle(gauge: RingGauge, dt: float = 1 / 12, limit: int = 200) -> int:
    for i in range(limit):
        if not gauge.step(dt):
            return i
    raise AssertionError("gauge never settled")


async def _ring_run() -> dict[str, Any]:
    app = GaugeApp()
    out: dict[str, Any] = {}
    async with app.run_test(size=(60, 20)) as pilot:
        gauge = app.query_one(RingGauge)
        gauge.set_target(80.0, "utilization", "")
        out["steps"] = _settle(gauge)
        out["settled"] = gauge.settled
        await pilot.pause(0.2)
        painted = gauge.rows_painted
        out["still"] = [gauge.step(1 / 12) for _ in range(5)]
        await pilot.pause(0.2)
        out["repainted_when_still"] = gauge.rows_painted - painted
        painted = gauge.rows_painted
        gauge.step(1 / 12, shimmer=0.5)  # a highlight halfway along the lit arc
        await pilot.pause(0.2)
        out["shimmer_rows"] = gauge.rows_painted - painted
        out["height"] = gauge.content_size.height
    return out


def test_ring_settles_then_stops_repainting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(widgets, "BOOT_S", 0.0)  # skip the startup sweep
    r = asyncio.run(_ring_run())
    assert r["steps"] < 60 and r["settled"]  # eased in well under 5 s at 12 fps
    assert r["still"] == [False] * 5  # settled: no more frames requested
    assert r["repainted_when_still"] == 0
    assert 0 < r["shimmer_rows"] < r["height"]  # the shimmer repaints only the rows it crosses


def test_row_key_ignores_motion_below_one_dot() -> None:
    plan = strips.ring_plan(30, 15)
    for row in plan:
        assert strips.row_key(row, 0.5, None) == strips.row_key(row, 0.5 + 1e-6, None)
    assert any(strips.row_key(r, 0.2, None) != strips.row_key(r, 0.8, None) for r in plan)
    # A shimmer beyond the lit arc changes nothing.
    assert all(strips.row_key(r, 0.3, None) == strips.row_key(r, 0.3, 0.9) for r in plan)


def test_quiet_static_skips_identical_updates() -> None:
    async def run() -> int:
        class One(App[None]):
            def compose(self) -> ComposeResult:
                yield QuietStatic(id="s")

        app = One()
        async with app.run_test() as pilot:
            s = app.query_one(QuietStatic)
            for _ in range(3):
                s.update(Text("same", R.style(R.INK)))
            s.update(Text("different", R.style(R.INK)))
            await pilot.pause()
            return s.updates

    assert asyncio.run(run()) == 2


async def _frames_run(fps: float) -> dict[str, Any]:
    mon = Monitor(MonitorConfig(demo=True, persist=False, interval_s=0.5))
    app = GpuDashboard(mon, fps=fps)
    out: dict[str, Any] = {}
    async with app.run_test(size=(130, 40)) as pilot:
        await pilot.pause(1.5)
        if fps:
            out["overview_paused"] = app.frames_paused  # the pulse keeps the Overview animating
            await pilot.press("4")  # History: nothing animates
            await pilot.pause(0.6)
            out["history_paused"] = app.frames_paused
            f0 = app.frames
            await pilot.pause(0.6)
            out["frames_on_history"] = app.frames - f0
            await pilot.press("1")
            await pilot.pause(0.6)
            out["resumed"] = not app.frames_paused and app.frames > f0
        gauge = app.query_one("#ring-gpu", RingGauge)
        out["snapped"] = gauge.value == (gauge.target or 0.0)
    mon.close()
    return out


def test_frame_timer_pauses_off_overview() -> None:
    r = asyncio.run(_frames_run(12))
    assert r["overview_paused"] is False
    assert r["history_paused"] is True and r["frames_on_history"] == 0
    assert r["resumed"]


def test_fps_zero_snaps_values_without_frames() -> None:
    r = asyncio.run(_frames_run(0))
    assert r["snapped"]
