"""Dashboard UI: responsive layout rules and headless runs of the Textual app in demo mode."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from gpu_pulse import anim, cards, panels
from gpu_pulse import render as R
from gpu_pulse.app import GpuDashboard
from gpu_pulse.intel import Diagnosis
from gpu_pulse.layout import plan
from gpu_pulse.models import GpuSample, Sample
from gpu_pulse.monitor import Monitor, MonitorConfig
from gpu_pulse.widgets import RingGauge, ring_lines

SIZES = ((80, 24), (90, 28), (120, 36), (130, 40), (200, 60))


# ---- pure layout rules ---------------------------------------------------------------------
def test_plan_grows_rings_with_terminal() -> None:
    small, large = plan(90, 28), plan(200, 60)
    assert large.ring_h > small.ring_h
    assert plan(200, 60).ring_cols == 6


def test_plan_compact_and_meters() -> None:
    tiny = plan(60, 18)
    assert tiny.compact and tiny.columns == 1 and not tiny.show_pulse and not tiny.show_bottom
    assert tiny.meters  # rings too small to draw become one-line meters
    assert not plan(130, 40).compact


@pytest.mark.parametrize("w,h", [(70, 60), (100, 70), (130, 40), (240, 70)])
def test_plan_ring_block_fits_screen(w: int, h: int) -> None:
    p = plan(w, h)
    assert p.ring_cols * p.ring_rows == 6
    assert p.ring_rows * p.ring_h <= h - 2
    if not p.meters:  # round: content width ≈ 2 × content height
        assert (w // p.ring_cols - 2) >= 2 * (p.ring_h - 2) - 1


@pytest.mark.parametrize("h", range(5, 23))
def test_ring_text_never_touches_the_ring(h: int) -> None:
    for w in (2 * h - 1, 2 * h + 1, 2 * h + 6):
        g = anim.ring_geometry(w, h)
        for num, unit, cap, sub in (("100", "%", "utilization", "1h avg 50%"), ("151", "W", "of 180 W cap", "")):
            lines, top = ring_lines(g, num, unit, cap, sub, R.GPU)
            for i, (text, _, _) in enumerate(lines):
                assert g.fits(top + i, len(text), margin=0) or lines == ((f"{num}{unit}", R.INK, True),)


def test_big_digits_when_roomy() -> None:
    g = anim.ring_geometry(40, 20)
    lines, _ = ring_lines(g, "87", "%", "utilization", "1h avg 50%", R.GPU)
    assert len(lines) == 5  # 3 rows of big digits + caption + sub-line


def test_fmt_rate_units() -> None:
    assert R.fmt_rate(None) == "n/a"
    assert R.fmt_rate(512) == "512 B/s"
    assert R.fmt_rate(3_400_000) == "3.4 MB/s"
    assert R.fmt_rate(2.5e9) == "2.5 GB/s"


def test_true_usage_fallback_and_divergence() -> None:
    bare = GpuSample(0, "u", "g", "nvidia", util=95)
    assert "not available" in panels.true_usage(bare, 40, 10).renderables[0].plain  # type: ignore[union-attr]
    stalled = GpuSample(0, "u", "g", "nvidia", util=95, sm_active=12)
    assert panels.divergence(stalled) is not None and panels.divergence(stalled)[0] == "warning"  # type: ignore[index]


def test_health_fallback_covers_system() -> None:
    s = Sample(t=0, gpus=(), cpu_util=50, cpu_per_core=(40, 60), mem_used=8 << 30, mem_total=16 << 30)
    names = [r.name for r in cards.health_rows(Diagnosis(), s, None, 80, None)]
    assert names == ["CPU", "RAM", "Disk", "Network"]


def test_sessions_bridge_short_gaps() -> None:
    cells: list[Any] = [0] * 10 + [80] * 5 + [0, 0] + [70] * 4 + [None] * 20 + [90] * 2
    runs = cards.find_sessions(0, cells)
    assert len(runs) == 1 and runs[0].minutes == 11


# ---- headless app runs ---------------------------------------------------------------------
def _text(widget: Any) -> str:
    return "\n".join("".join(seg.text for seg in line) for line in widget.render_lines(widget.region.reset_offset))


async def _drive(size: tuple[int, int]) -> dict[str, Any]:
    mon = Monitor(MonitorConfig(demo=True, persist=False, interval_s=0.5))
    app = GpuDashboard(mon, fps=20)
    out: dict[str, Any] = {}
    async with app.run_test(size=size) as pilot:
        await pilot.pause(1.2)
        rings = list(app.query(RingGauge))
        out["rings"] = len(rings)
        out["ring_h"] = rings[0].region.height
        out["layout"] = app.layout_plan
        out["overview_health"] = _text(app.query_one("#health")) if app.layout_plan.columns == 3 else ""
        for key in ("2", "3", "4", "5"):
            await pilot.press(key)
            await pilot.pause(0.3)
            out[f"tab{key}"] = app.tab
        out["health_text"] = _text(app.query_one("#health-full"))
        await pilot.press("right")
        await pilot.pause(0.2)
        out["wrapped"] = app.tab
        await pilot.press("question_mark")
        await pilot.pause(0.3)
        out["help"] = type(app.screen).__name__
        await pilot.press("escape")
        await pilot.press("1")
        await pilot.pause(0.3)
        out["back"] = app.tab
    mon.close()
    return out


@pytest.mark.parametrize("size", SIZES)
def test_dashboard_runs_at_size(size: tuple[int, int]) -> None:
    r = asyncio.run(_drive(size))
    assert r["rings"] == 6
    assert [r["tab2"], r["tab3"], r["tab4"], r["tab5"]] == ["gpu", "system", "history", "diagnosis"]
    assert r["wrapped"] == "overview" and r["back"] == "overview"
    assert r["help"] == "HelpScreen"
    assert "GPU" in r["health_text"] and "RAM" in r["health_text"]
    assert r["ring_h"] == r["layout"].ring_h
    if r["layout"].columns == 3:  # wide terminals show the health grid on the Overview
        assert "CPU" in r["overview_health"] and "Network" in r["overview_health"]


def test_ring_height_follows_terminal_size() -> None:
    small = asyncio.run(_drive((90, 28)))["ring_h"]
    large = asyncio.run(_drive((200, 60)))["ring_h"]
    assert large > small


def test_compact_terminal_runs() -> None:
    r = asyncio.run(_drive((60, 20)))
    assert r["layout"].compact and r["rings"] == 6
