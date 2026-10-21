"""Regression tests for the code-review findings (bad files, clears, locking)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gpu_pulse.history import HistoryFile, HistoryStore

from .test_history import T0, at, fill, sample


def test_malformed_buckets_are_dropped_not_fatal() -> None:
    h = HistoryStore()
    good_gpu = {
        "n": 6,
        "util": 50,
        "utilMax": 60,
        "vram": 10,
        "power": 5,
        "tempMax": 40,
        "idle": 0,
        "busy": 0,
        "heldIdle": 0,
    }
    h.load(
        {
            "version": 1,
            "clearedAt": "garbage",
            "minutes": [
                {"t": T0 * 1000 + 0.5, "n": 6, "cpu": 1, "mem": 1, "gpus": {"G": good_gpu}},  # float t -> normalized
                {"t": T0 * 1000 + 60_000, "cpu": 1, "mem": 1, "gpus": {}},  # missing n
                {"t": T0 * 1000 + 120_000, "n": 6, "cpu": 1, "mem": 1, "gpus": {"G": {"n": 6}}},  # partial gpu
                {"t": True, "n": 6, "cpu": 1, "mem": 1, "gpus": {}},  # bool is not a number
            ],
        }
    )
    # Minute 1 (no n) is dropped; minute 2 keeps CPU/RAM but loses its incomplete GPU entry.
    assert h._times == [int(T0 * 1000), int(T0 * 1000) + 120_000]
    assert h.minutes[1]["gpus"] == {}
    s = h.stats("1h", 60, "G", at(3))  # must not raise
    assert s.util == 50
    start, cells = h.heatmap("G", 1, now=T0 + 59 * 60)
    assert cells[0] == 50


def test_clear_from_other_process_keeps_newer_samples(tmp_path: Path) -> None:
    f = HistoryFile(tmp_path / "h.json")
    logger = HistoryStore()
    fill(logger, T0, 3, lambda i: 40)
    clear_ms = int((T0 + 150) * 1000)  # mid-minute 2
    clearer = HistoryStore()
    clearer.clear(clear_ms)
    assert f.save(clearer)
    logger.add(sample(T0 + 170, 90))  # after the clear, same minute
    logger.add(sample(T0 + 185, 90))  # next minute (open)
    assert f.save(logger)
    assert logger.stats("1h", 60, "G", at(4)).util_peak == 90  # the minute after the clear survived
    assert logger.stats("3m", 3, "G", at(3)).covered_minutes == 0  # minutes 0-2 (through the clear) dropped
    assert logger.stats("1m", 1, "G", at(4)).covered_minutes == 1


def test_save_skips_when_locked(tmp_path: Path) -> None:
    f = HistoryFile(tmp_path / "h.json")
    f.LOCK_WAIT_S = 0.2
    f.lock_path.write_text("held")
    h = HistoryStore()
    fill(h, T0, 1, lambda i: 10)
    assert f.save(h) is False
    assert not f.path.exists()
    os.utime(f.lock_path, (0, 0))  # make it stale
    assert f.save(h) is True
    assert not f.lock_path.exists()


def test_corrupt_file_is_backed_up(tmp_path: Path) -> None:
    f = HistoryFile(tmp_path / "h.json")
    f.path.write_text("{not json")
    h = HistoryStore()
    fill(h, T0, 1, lambda i: 10)
    assert f.save(h)
    assert json.loads(f.path.read_text())["version"] == 1
    assert list(tmp_path.glob("h.json.corrupt-*"))


@pytest.mark.skipif(os.name != "nt", reason="exclusive file locks are Windows-specific")
def test_unreadable_file_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = HistoryFile(tmp_path / "h.json")
    f.path.write_text(json.dumps({"version": 1, "minutes": []}))

    def deny(*_a: object, **_k: object) -> str:
        raise PermissionError("locked by another process")

    monkeypatch.setattr(Path, "read_text", deny)
    h = HistoryStore()
    fill(h, T0, 1, lambda i: 10)
    assert f.save(h) is False
