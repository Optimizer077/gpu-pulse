"""Regressions for the fourth code review: GPU selection, ALL GPUS width, shimmer cost, help resume, `g` prefix."""

from __future__ import annotations

import asyncio
from typing import Any

from gpu_pulse import render as R
from gpu_pulse.app import SHIMMER_EVERY_S, GpuDashboard
from gpu_pulse.fleetview import fleet_columns
from gpu_pulse.monitor import Monitor, MonitorConfig
from gpu_pulse.widgets import RingGauge


def _monitor(n: int, interval_s: float = 0.5) -> Monitor:
    return Monitor(MonitorConfig(demo=True, demo_gpus=n, persist=False, interval_s=interval_s))


def _ring_title(app: GpuDashboard) -> str:
    return str(app.query_one("#ring-gpu", RingGauge).border_title)


# ---- MEDIUM 4: the selection follows the physical GPU ----------------------------------------
def test_selection_follows_the_gpu_when_another_drops_out() -> None:
    async def run() -> dict[str, Any]:
        mon = _monitor(3, interval_s=30)  # no timed samples: the test drives them
        app = GpuDashboard(mon, fps=0)
        out: dict[str, Any] = {}
        read = mon._read_gpus
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause(0.3)
            app.select_gpu(2)
            await pilot.pause(0.1)
            mon._read_gpus = lambda: tuple(g for g in read() if g.index != 1)  # type: ignore[method-assign]
            app.refresh_data()  # GPU 1 is gone (driver reset): GPU 2 is now at position 1
            await pilot.pause(0.2)
            out["kept"] = (app.gpu_idx, app.gpu_uuid, _ring_title(app))
            mon._read_gpus = lambda: tuple(g for g in read() if g.index == 0)  # type: ignore[method-assign]
            app.refresh_data()  # the selected GPU itself is gone: fall back to what is there
            await pilot.pause(0.2)
            out["fallback"] = (app.gpu_idx, _ring_title(app))
            mon._read_gpus = read  # type: ignore[method-assign]
            app.refresh_data()  # it comes back: shown again, at its old position
            await pilot.pause(0.2)
            out["back"] = (app.gpu_idx, _ring_title(app))
        mon.close()
        return out

    r = asyncio.run(run())
    assert r["kept"][0] == 1 and r["kept"][1] == "GPU-demo-0002" and r["kept"][2].startswith("GPU 2 ·")
    assert r["fallback"][0] == 0 and r["fallback"][1].startswith("GPU ·")  # one GPU left: no index in the title
    assert r["back"][0] == 2 and r["back"][1].startswith("GPU 2 ·")


# ---- MEDIUM 5: columns are planned at the width the panel is drawn at ------------------------
def test_all_gpus_fit_near_the_two_column_threshold() -> None:
    async def run(size: tuple[int, int], scrollbar: bool) -> dict[str, Any]:
        mon = _monitor(8)
        app = GpuDashboard(mon, fps=0)
        async with app.run_test(size=size) as pilot:
            await pilot.pause(0.5)
            if scrollbar:  # the review's case: the Overview's scrollbar takes one column from the panel
                app.query_one("#overview").styles.overflow_y = "scroll"
                await pilot.pause(1.2)
            fp = app.query_one("#fleet")
            lines = fp.render_lines(fp.region.reset_offset)
            out = {
                "hits": len(getattr(fp, "hits", [])),
                "text": "\n".join("".join(seg.text for seg in line) for line in lines),
                "scrollbar": app.query_one("#overview").show_vertical_scrollbar,
                "cols": (fleet_columns(8, app.fleet_width), fleet_columns(8, fp.inner_width)),  # type: ignore[attr-defined]
            }
        mon.close()
        return out

    for size in ((174, 40), (175, 40), (174, 60)):
        for scrollbar in (False, True):
            r = asyncio.run(run(size, scrollbar))
            assert r["scrollbar"] == scrollbar, size
            assert r["hits"] == 8, (size, scrollbar)  # every GPU shown ...
            assert "to scroll" not in r["text"], (size, scrollbar)  # ... without a "g/G to scroll" hint
            assert r["cols"][0] == r["cols"][1], (size, scrollbar)  # planned and drawn columns agree


# ---- MEDIUM 6: shimmer bookkeeping -----------------------------------------------------------
def _gauge() -> RingGauge:
    return RingGauge("CPU", R.CPU, R.CPU_GLOW)


def test_shimmer_needs_a_visible_move_of_two_units() -> None:
    g = _gauge()
    g.set_target(50.0)
    assert g.consume_change()  # first value
    g.set_target(50.4)  # same number on screen
    assert not g._changed and not g.consume_change()
    g.set_target(51.0)  # shown value moved by one: jitter
    assert not g.consume_change()
    g.set_target(52.2)  # two units from the last pass
    assert g.consume_change() and not g.consume_change()
    g.set_target(None)
    assert g.consume_change()


def test_shimmer_pass_consumes_every_gauge_and_only_moved_rings_shimmer() -> None:
    mon = _monitor(1)
    app = GpuDashboard(mon)
    gauges = [_gauge() for _ in range(3)]
    app.gauges = gauges
    for g in gauges:
        g.set_target(40.0)
    app._shimmer_at(app._t0 + SHIMMER_EVERY_S + 0.1)
    assert app._shimmer_mask == [True, True, True] and not any(g._changed for g in gauges)
    gauges[1].set_target(45.0)
    gauges[2].set_target(41.0)  # one unit: not enough
    assert app._shimmer_at(app._t0 + 2 * SHIMMER_EVERY_S + 0.1) is not None
    assert app._shimmer_mask == [False, True, False] and not any(g._changed for g in gauges)
    assert app._shimmer_at(app._t0 + 3 * SHIMMER_EVERY_S + 0.1) is None  # nothing moved: no pass
    mon.close()


# ---- LOW 7: closing help restarts the animation at once --------------------------------------
def test_closing_help_restarts_frames_immediately() -> None:
    async def run() -> tuple[bool, bool]:
        mon = _monitor(1, interval_s=30)  # no sample would wake the frames for 30 s
        app = GpuDashboard(mon, fps=12)
        async with app.run_test(size=(130, 40)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("question_mark")
            await pilot.pause(0.5)
            paused = app.frames_paused
            await pilot.press("escape")
            await pilot.pause(0.2)
            resumed = not app.frames_paused
        mon.close()
        return paused, resumed

    assert asyncio.run(run()) == (True, True)


# ---- LOW 8: the `g` prefix -------------------------------------------------------------------
def test_g_prefix_swallows_unknown_digits_and_ends_on_other_keys() -> None:
    async def run() -> dict[str, Any]:
        mon = _monitor(2)
        app = GpuDashboard(mon, fps=0)
        out: dict[str, Any] = {}
        async with app.run_test(size=(130, 40)) as pilot:
            await pilot.pause(0.3)
            await pilot.press("g", "3")  # no GPU 3 on a 2-GPU box: swallowed, not the System tab
            await pilot.pause(0.2)
            out["g3"] = (app.tab, app.gpu_idx)
            await pilot.press("g", "r", "2")  # another key ends the prefix: 2 opens the GPU tab
            await pilot.pause(0.2)
            out["g_r_2"] = app.tab
            await pilot.press("g", "right", "1")  # so does a tab switch
            await pilot.pause(0.2)
            out["g_right_1"] = app.tab
        mon.close()
        return out

    r = asyncio.run(run())
    assert r["g3"] == ("overview", 1)
    assert r["g_r_2"] == "gpu"
    assert r["g_right_1"] == "overview"


def test_g_prefix_takes_two_digits_on_big_machines() -> None:
    async def run() -> tuple[int, int, str]:
        mon = _monitor(12)
        app = GpuDashboard(mon, fps=0)
        async with app.run_test(size=(180, 60)) as pilot:
            await pilot.pause(0.3)
            await pilot.press("g", "1", "1")  # GPU 11
            await pilot.pause(0.2)
            eleven = app.gpu_idx
            await pilot.press("g", "1")  # GPU 1; the window for a second digit then runs out
            await pilot.pause(1.8)
            one = app.gpu_idx
            await pilot.press("2")
            await pilot.pause(0.2)
            tab = app.tab
        mon.close()
        return eleven, one, tab

    assert asyncio.run(run()) == (11, 1, "gpu")
