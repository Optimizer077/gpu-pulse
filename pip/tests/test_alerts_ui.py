"""Alerts in the dashboard (toasts, header, Diagnosis list, mute, bell) and in the CLI (`log --alerts`)."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import pytest

from gpu_pulse import cli, demo
from gpu_pulse.alerts import AlertConfig
from gpu_pulse.app import GpuDashboard
from gpu_pulse.models import GpuSample, Sample
from gpu_pulse.monitor import Monitor, MonitorConfig


def _text(widget: Any) -> str:
    return "\n".join("".join(seg.text for seg in line) for line in widget.render_lines(widget.region.reset_offset))


def _sample(t: float, gpus: tuple[GpuSample, ...]) -> Sample:
    return Sample(t=t, gpus=gpus, cpu_util=10.0, cpu_per_core=(10.0,), mem_used=1 << 30, mem_total=8 << 30)


def _future_steady_start() -> float:
    """A demo-clock time in the future at the start of the 'steady' phase (all four GPUs busy, GPU 2 hot)."""
    cycles = math.ceil((time.time() - demo._T0) / demo.CYCLE_S) + 1
    return demo._T0 + cycles * demo.CYCLE_S


async def _drive(bell: bool) -> dict[str, Any]:
    # A long sampling interval: only the mount tick is real; the test injects the 4-GPU demo's own
    # readings for its first 35 s, as the dashboard would have sampled them at 0.5 s.
    mon = Monitor(MonitorConfig(demo=True, demo_gpus=4, persist=False, interval_s=30))
    app = GpuDashboard(mon, fps=0, alert_config=AlertConfig(interval_s=0.5), bell=bell)
    rings: list[int] = []
    app.bell = lambda: rings.append(1)  # type: ignore[method-assign]
    out: dict[str, Any] = {}
    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.pause(0.3)
        t0 = _future_steady_start()
        for k in range(71):
            t = t0 + k * 0.5
            app._alerts(_sample(t, demo.demo_gpus(t, 4)))
        app.refresh_data(sample=False)
        await pilot.pause(0.2)
        out["toasts"] = [(n.title, n.severity) for n in app._notifications]
        out["header"] = _text(app.query_one("#header"))
        await pilot.press("5")
        await pilot.pause(0.4)
        out["panel"] = _text(app.query_one("#alerts"))
        await pilot.press("m")
        await pilot.pause(0.2)
        out["header_muted"] = _text(app.query_one("#header"))
        full = GpuSample(7, "GPU-test-7", "x", "nvidia", util=90, mem_used_mib=15900, mem_total_mib=16000)
        for k in range(42):  # VRAM full for 21 s while muted: listed, but no toast or bell
            app._alerts(_sample(t0 + 100 + k * 0.5, (full,)))
        app.draw_alerts()
        await pilot.pause(0.2)
        out["toasts_muted"] = len(app._notifications)
        out["panel_muted"] = _text(app.query_one("#alerts"))
        await pilot.press("m")
        await pilot.pause(0.2)
        out["header_unmuted"] = _text(app.query_one("#header"))
        out["rings"] = len(rings)
    mon.close()
    return out


def test_demo_hot_gpu_raises_toast_header_count_and_list_entry() -> None:
    r = asyncio.run(_drive(bell=True))
    alert_toasts = [t for t in r["toasts"] if "running hot" in t[0]]
    assert alert_toasts == [("GPU 2 running hot: 86 °C", "warning")]
    assert "▲ 1 alert" in r["header"]
    assert "GPU 2 running hot: 86 °C" in r["panel"] and "Watching:" in r["panel"]
    assert "alerts muted" in r["header_muted"]
    assert "VRAM nearly full" in r["panel_muted"] and "GPU 2 running hot" in r["panel_muted"]
    assert r["panel_muted"].index("VRAM nearly full") < r["panel_muted"].index("GPU 2 running hot")  # newest first
    assert r["toasts_muted"] == len(r["toasts"]) + 1  # +1 is the "Alerts muted" notice, not the alert
    assert "alerts muted" not in r["header_unmuted"]
    assert r["rings"] == 1  # one bell for the one unmuted alert


def test_no_alerts_flag_disables_the_engine() -> None:
    async def run() -> tuple[str, list[str]]:
        mon = Monitor(MonitorConfig(demo=True, persist=False, interval_s=0.5))
        app = GpuDashboard(mon, fps=0, alerts=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("5")
            await pilot.pause(0.4)
            await pilot.press("m")
            await pilot.pause(0.1)
            panel = _text(app.query_one("#alerts"))
            notes = [n.message for n in app._notifications]
        mon.close()
        return panel, notes

    panel, notes = asyncio.run(run())
    assert "Alerts are off" in panel
    assert any("--no-alerts" in n for n in notes)


# ---- CLI ----------------------------------------------------------------------------------
def test_alert_options_defaults_and_overrides() -> None:
    p = cli._parser()
    a = p.parse_args([])
    assert (a.alerts, a.bell, a.idle_held_min, a.job_min) == (None, False, 15.0, 10.0)
    assert p.parse_args(["--no-alerts"]).alerts is False
    a = p.parse_args(["--job-min", "7", "--idle-held-min", "3", "log"])  # global flags survive the subcommand
    assert (a.job_min, a.idle_held_min, a.alerts, a.notify) == (7.0, 3.0, None, False)
    a = p.parse_args(["log", "--alerts", "--job-min", "5", "--notify"])
    assert (a.alerts, a.job_min, a.notify, a.idle_held_min) == (True, 5.0, True, 15.0)
    with pytest.raises(SystemExit):
        p.parse_args(["--job-min", "0"])


def _hot(t: float) -> Sample:
    return _sample(t, (GpuSample(0, "GPU-0", "x", "nvidia", util=90, temp_c=88),))


def test_log_alerts_prints_one_line_per_alert(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.alert_printer(cli._parser().parse_args(["log"]), 5.0) is None  # off unless asked
    on_sample = cli.alert_printer(cli._parser().parse_args(["log", "--alerts"]), 5.0)
    assert on_sample is not None
    for k in range(10):  # 5 s samples: hot from 0 s, fires at 30 s, once
        on_sample(_hot(1.7e9 + k * 5))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].endswith("  WARNING  GPU 0 running hot: 88 °C")


def test_notify_uses_notify_send_on_linux(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/notify-send" if name == "notify-send" else None)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    on_sample = cli.alert_printer(cli._parser().parse_args(["log", "--notify"]), 1.0)  # --notify implies alerts
    assert on_sample is not None
    for k in range(31):
        on_sample(_hot(1.7e9 + k))
    assert len(calls) == 1
    assert calls[0][:5] == ["/usr/bin/notify-send", "-a", "GPU Pulse", "-u", "normal"]
    assert calls[0][5] == "GPU 0 running hot: 88 °C"
    assert "running hot" in capsys.readouterr().out


def test_notify_without_a_mechanism_just_prints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "platform", "win32")
    assert cli.desktop_notifier() is None
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.desktop_notifier() is None
