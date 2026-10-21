"""Optional per-minute GPM keys ("sm", "tensor") in history buckets stay backward compatible."""

from __future__ import annotations

from typing import Optional

from gpu_pulse.history import HistoryStore
from gpu_pulse.models import GpuSample, Sample

T0 = 1_792_576_800.0


def sample(t: float, util: float, sm: Optional[float] = None, tensor: Optional[float] = None) -> Sample:
    g = GpuSample(
        index=0,
        uuid="G",
        name="x",
        vendor="nvidia",
        util=util,
        mem_used_mib=1000,
        mem_total_mib=10_000,
        sm_active=sm,
        tensor_active=tensor,
    )
    return Sample(t=t, gpus=(g,), cpu_util=10, cpu_per_core=(10,), mem_used=1, mem_total=2)


def bucket(**extra: object) -> dict:
    g = {"n": 6, "util": 50, "utilMax": 60, "vram": 10, "power": 5, "tempMax": 40, "idle": 0, "busy": 0, "heldIdle": 0}
    g.update(extra)
    return g


def test_buckets_store_sm_and_tensor_means() -> None:
    h = HistoryStore()
    for i in range(6):
        h.add(sample(T0 + i * 10, 90, sm=40 + i * 4, tensor=20))
    h.add(sample(T0 + 60, 10))  # next minute flushes the first one
    b = h.minutes[0]["gpus"]["G"]
    assert b["sm"] == 50.0 and b["tensor"] == 20.0
    st = h.stats("1m", 2, "G", T0 + 60)
    assert st.sm == 50.0 and st.tensor == 20.0
    _, cells = h.heatmap("G", 1, now=T0 + 60, metric="sm")
    assert cells[-2] == 50.0 and cells[-1] is None  # second minute has no GPM data
    _, util = h.heatmap("G", 1, now=T0 + 60)
    assert util[-1] == 10.0


def test_no_gpm_means_no_keys() -> None:
    h = HistoryStore()
    for i in range(7):
        h.add(sample(T0 + i * 10, 50))
    b = h.minutes[0]["gpus"]["G"]
    assert "sm" not in b and "tensor" not in b
    assert h.stats("1m", 1, "G", T0).sm is None


def test_normalize_keeps_valid_and_drops_invalid() -> None:
    h = HistoryStore()
    h.load(
        {
            "version": 1,
            "minutes": [
                {"t": T0 * 1000, "n": 6, "cpu": 1, "mem": 1, "gpus": {"G": bucket(sm=42.5, tensor=7)}},
                {"t": T0 * 1000 + 60_000, "n": 6, "cpu": 1, "mem": 1, "gpus": {"G": bucket(sm="x", tensor=150)}},
                {"t": T0 * 1000 + 120_000, "n": 6, "cpu": 1, "mem": 1, "gpus": {"G": bucket()}},  # extension format
            ],
        }
    )
    assert h.minutes[0]["gpus"]["G"]["sm"] == 42.5 and h.minutes[0]["gpus"]["G"]["tensor"] == 7
    assert "sm" not in h.minutes[1]["gpus"]["G"] and "tensor" not in h.minutes[1]["gpus"]["G"]
    assert h.minutes[2]["gpus"]["G"]["util"] == 50
    assert h.stats("3m", 3, "G", T0 + 120).sm == 42.5  # weighted only over minutes that have it


def test_merge_keeps_sm_from_either_side() -> None:
    h = HistoryStore()
    for i in range(3):
        h.add(sample(T0 + i * 10, 90, sm=60))
    h.add(sample(T0 + 70, 90))  # flush minute 0 (with sm)
    for i in range(3):  # same minute again, but without GPM: merges into the stored bucket
        h.add(sample(T0 + 30 + i * 5, 90))
    h.add(sample(T0 + 130, 90))
    assert h.minutes[0]["gpus"]["G"]["sm"] == 60.0
