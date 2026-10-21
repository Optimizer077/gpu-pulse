"""Regressions for the third code review (diagnosis rules, I/O peaks, NIC deltas)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from gpu_pulse import _intel_gpu as G
from gpu_pulse._intel_system import p95
from gpu_pulse.collectors import system as S


def _busy_feat(**kw: object) -> G.GpuFeat:
    """A steady, fully busy GPU window; override fields per test."""
    base = G.GpuFeat(
        **{f: 0 for f in G.GpuFeat.__dataclass_fields__ if f not in kw}  # type: ignore[arg-type]
    )
    defaults = dict(
        span_s=180.0,
        run_mean=99.0,
        run_share=1.0,
        run_cv=0.02,
        high_share=1.0,
        util_mean=99.0,
        limit_share=0.0,
        near_cap_share=0.0,
        quiet_s=0.0,
        idle_share=0.0,
        held_share=0.0,
        sm_mean=90.0,
        tensor_mean=40.0,
        dram_mean=None,
    )
    defaults.update(kw)
    return replace(base, **{k: v for k, v in defaults.items() if k in G.GpuFeat.__dataclass_fields__})


def _top(calls: list[G.Call]) -> str:
    return max(calls, key=lambda c: c.confidence).bottleneck if calls else "none?"


def test_memory_bound_is_not_reported_as_unconstrained() -> None:
    calls = G.gpu_calls(_busy_feat(dram_mean=95.0), temp_warn=80)
    assert "none" not in {c.bottleneck for c in calls}
    assert _top(calls) == "memory"


def test_low_sm_rule_stays_out_of_memory_bound_and_is_modest() -> None:
    assert G.launch_bound(_busy_feat(sm_mean=20.0, dram_mean=85.0)) == 0.0
    conf = G.launch_bound(_busy_feat(sm_mean=12.0, dram_mean=10.0))
    assert 0 < conf <= 0.5


def test_p95_ignores_single_burst() -> None:
    vals = [100e6] * 99 + [3e9]
    assert p95(vals) == 100e6
    assert p95([]) is None


def test_net_rates_ignore_new_adapter(monkeypatch) -> None:  # noqa: ANN001
    snaps = iter(
        [
            {"eth0": SimpleNamespace(bytes_recv=1_000, bytes_sent=500)},
            {
                "eth0": SimpleNamespace(bytes_recv=3_000, bytes_sent=900),
                "vpn0": SimpleNamespace(bytes_recv=9_000_000_000, bytes_sent=9_000_000_000),  # appears late
            },
        ]
    )
    monkeypatch.setattr(S.psutil, "net_io_counters", lambda pernic=True: next(snaps))
    c = S.SystemCollector.__new__(S.SystemCollector)
    c._net_nics, c._net_t, c._net_rate = {}, None, (None, None)
    assert c._net_rates(100.0) == (None, None)
    rx, tx = c._net_rates(102.0)
    assert (rx, tx) == (1_000.0, 200.0)  # only eth0's delta over 2 s
    assert c._net_rates(103.0) == (rx, tx)  # within NET_EVERY_S: cached, no new read
