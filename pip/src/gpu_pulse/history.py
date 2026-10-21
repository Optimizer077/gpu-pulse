"""Per-minute usage history with weighted window stats.

The on-disk format matches the VS Code extension's ``history.json`` (version 1,
timestamps in epoch milliseconds), so both tools describe usage the same way.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Optional

from .models import Sample, WindowStats

logger = logging.getLogger(__name__)

MINUTE_MS = 60_000
IDLE_UTIL = 10.0
BUSY_UTIL = 70.0
HELD_VRAM = 50.0  # above a desktop's own compositor/app usage
RECENT_S = 3600


def minute_start(t_ms: float) -> int:
    return int(t_ms // MINUTE_MS * MINUTE_MS)


def _r2(x: float) -> float:
    return round(x, 2)


@dataclass
class _GpuAcc:
    n: int = 0
    util: float = 0.0
    util_max: float = 0.0
    vram: float = 0.0
    power: float = 0.0
    temp_max: float = 0.0
    idle: int = 0
    busy: int = 0
    held_idle: int = 0
    sm: float = 0.0  # GPM true-usage sums, counted separately (GPM may be absent or start late)
    sm_n: int = 0
    tensor: float = 0.0
    tensor_n: int = 0


def _gpu_bucket(g: _GpuAcc) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": g.n,
        "util": _r2(g.util / g.n),
        "utilMax": g.util_max,
        "vram": _r2(g.vram / g.n),
        "power": _r2(g.power / g.n),
        "tempMax": g.temp_max,
        "idle": g.idle,
        "busy": g.busy,
        "heldIdle": g.held_idle,
    }
    # Optional keys: absent when GPM is unsupported, and absent in files written by the VS Code extension.
    if g.sm_n:
        out["sm"] = _r2(g.sm / g.sm_n)
    if g.tensor_n:
        out["tensor"] = _r2(g.tensor / g.tensor_n)
    return out


@dataclass
class _Acc:
    t: int
    n: int = 0
    cpu: float = 0.0
    mem: float = 0.0
    gpus: dict[str, _GpuAcc] = field(default_factory=dict)

    def bucket(self) -> dict[str, Any]:
        gpus = {uid: _gpu_bucket(g) for uid, g in self.gpus.items() if g.n}
        n = self.n or 1
        return {"t": self.t, "n": self.n, "cpu": _r2(self.cpu / n), "mem": _r2(self.mem / n), "gpus": gpus}


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Combine two aggregates of the same minute, weighting means by sample count."""
    gpus = dict(a["gpus"])
    for uid, g in b["gpus"].items():
        x = gpus.get(uid)
        if not x:
            gpus[uid] = g
            continue
        n = x["n"] + g["n"]

        def w(k: str, x: dict[str, Any] = x, g: dict[str, Any] = g, n: int = n) -> float:
            return _r2((x[k] * x["n"] + g[k] * g["n"]) / n)

        merged = {
            "n": n,
            "util": w("util"),
            "utilMax": max(x["utilMax"], g["utilMax"]),
            "vram": w("vram"),
            "power": w("power"),
            "tempMax": max(x["tempMax"], g["tempMax"]),
            "idle": x["idle"] + g["idle"],
            "busy": x["busy"] + g["busy"],
            "heldIdle": x["heldIdle"] + g["heldIdle"],
        }
        for k in _OPT_KEYS:  # weighted by sample count when both sides have it, else whichever does
            if k in x and k in g:
                merged[k] = w(k)
            elif k in x or k in g:
                merged[k] = x[k] if k in x else g[k]
        gpus[uid] = merged
    n = a["n"] + b["n"]
    mean: Callable[[str], float] = lambda k: _r2((a[k] * a["n"] + b[k] * b["n"]) / n) if n else 0.0  # noqa: E731
    return {"t": a["t"], "n": n, "cpu": mean("cpu"), "mem": mean("mem"), "gpus": gpus}


_GPU_KEYS = ("n", "util", "utilMax", "vram", "power", "tempMax", "idle", "busy", "heldIdle")
_OPT_KEYS = ("sm", "tensor")  # optional per-minute GPM means (%), see _gpu_bucket


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None


def _normalize(m: Any) -> Optional[dict[str, Any]]:
    """Validate a bucket from disk (possibly written by another tool); None if unusable."""
    if not isinstance(m, dict) or not isinstance(m.get("gpus"), dict):
        return None
    t, n, cpu, mem = (_num(m.get(k)) for k in ("t", "n", "cpu", "mem"))
    if t is None or n is None or n < 1 or cpu is None or mem is None:
        return None
    gpus: dict[str, Any] = {}
    for uid, g in m["gpus"].items():
        raw = [_num(g.get(k)) for k in _GPU_KEYS] if isinstance(g, dict) else [None]
        vals = [v for v in raw if v is not None]
        if len(vals) == len(_GPU_KEYS) and vals[0] >= 1:
            entry = {k: (int(v) if k in ("n", "idle", "busy", "heldIdle") else v) for k, v in zip(_GPU_KEYS, vals)}
            for k in _OPT_KEYS:
                o = _num(g.get(k))
                if o is not None and 0 <= o <= 100:
                    entry[k] = o
            gpus[str(uid)] = entry
    return {"t": minute_start(t), "n": int(n), "cpu": cpu, "mem": mem, "gpus": gpus}


def _int(v: Any) -> int:
    n = _num(v)
    return int(n) if n is not None else 0


class HistoryStore:
    """Raw samples for the last hour plus sorted per-minute buckets."""

    def __init__(self, retention_days: int = 7) -> None:
        self.retention_days = retention_days
        self.recent: deque[Sample] = deque()
        self.minutes: list[dict[str, Any]] = []
        self._times: list[int] = []  # parallel to minutes, for bisect
        self._acc: Optional[_Acc] = None
        self.cleared_at = 0

    # ---- ingest ----
    def add(self, s: Sample) -> None:
        self.recent.append(s)
        while self.recent and self.recent[0].t < s.t - RECENT_S:
            self.recent.popleft()
        m = minute_start(s.t * 1000)
        if self._acc and self._acc.t != m:
            self._flush()
        if not self._acc:
            self._acc = _Acc(t=m)
        a = self._acc
        a.n += 1
        a.cpu += s.cpu_util
        a.mem += s.mem_pct
        for g in s.gpus:
            if g.util is None:
                continue
            v = g.vram_pct or 0.0
            ga = a.gpus.setdefault(g.uuid, _GpuAcc())
            ga.n += 1
            ga.util += g.util
            ga.util_max = max(ga.util_max, g.util)
            ga.vram += v
            ga.power += g.power_w or 0.0
            ga.temp_max = max(ga.temp_max, g.temp_c or 0.0)
            ga.idle += g.util < IDLE_UTIL
            ga.busy += g.util >= BUSY_UTIL
            ga.held_idle += g.util < IDLE_UTIL and v >= HELD_VRAM
            if g.sm_active is not None:
                ga.sm += g.sm_active
                ga.sm_n += 1
            if g.tensor_active is not None:
                ga.tensor += g.tensor_active
                ga.tensor_n += 1

    def _insert(self, b: dict[str, Any], merge: bool) -> None:
        i = bisect.bisect_left(self._times, b["t"])
        if i < len(self._times) and self._times[i] == b["t"]:
            if merge:
                self.minutes[i] = _merge(self.minutes[i], b)
            elif b["n"] > self.minutes[i]["n"]:
                self.minutes[i] = b
        else:
            self.minutes.insert(i, b)
            self._times.insert(i, b["t"])

    def _flush(self) -> None:
        if self._acc and self._acc.n:
            self._insert(self._acc.bucket(), merge=True)
        self._acc = None
        cutoff = time.time() * 1000 - self.retention_days * 86_400_000
        drop = bisect.bisect_left(self._times, cutoff)
        if drop:
            del self.minutes[:drop], self._times[:drop]

    def _each_since(self, frm: int) -> Iterator[dict[str, Any]]:
        """Buckets with t >= frm, newest first, including the open minute."""
        i = len(self.minutes) - 1
        if self._acc and self._acc.n:
            open_b = self._acc.bucket()
            if i >= 0 and self._times[i] == open_b["t"]:
                yield _merge(self.minutes[i], open_b)
                i -= 1
            else:
                yield open_b
        while i >= 0 and self._times[i] >= frm:
            yield self.minutes[i]
            i -= 1

    # ---- queries ----
    def recent_since(self, t0: float) -> list[Sample]:
        """Raw samples with ``t >= t0``, oldest first, without walking the whole hour."""
        out: list[Sample] = []
        for s in reversed(self.recent):
            if s.t < t0:
                break
            out.append(s)
        out.reverse()
        return out

    def recent_last(self, n: int) -> list[Sample]:
        """The newest ``n`` raw samples, oldest first (no copy of the full deque)."""
        out = list(islice(reversed(self.recent), max(0, n)))
        out.reverse()
        return out

    def stats(self, label: str, minutes: int, gpu: Optional[str], now: Optional[float] = None) -> WindowStats:
        to = minute_start((now if now is not None else time.time()) * 1000)
        frm = to - (minutes - 1) * MINUTE_MS
        covered = n = cn = idle = busy = held = sm_n = tensor_n = 0
        util = vram = power = cpu = mem = sm = tensor = 0.0
        peak: Optional[float] = None
        temp: Optional[float] = None
        for b in self._each_since(frm):
            if not frm <= b["t"] <= to:
                continue
            covered += 1
            cn += b["n"]
            cpu += b["cpu"] * b["n"]
            mem += b["mem"] * b["n"]
            g = b["gpus"].get(gpu) if gpu else next(iter(b["gpus"].values()), None)
            if not g:
                continue
            n += g["n"]
            util += g["util"] * g["n"]
            vram += g["vram"] * g["n"]
            power += g["power"] * g["n"]
            idle, busy, held = idle + g["idle"], busy + g["busy"], held + g["heldIdle"]
            peak = max(peak or 0.0, g["utilMax"])
            temp = max(temp or 0.0, g["tempMax"])
            if "sm" in g:
                sm, sm_n = sm + g["sm"] * g["n"], sm_n + g["n"]
            if "tensor" in g:
                tensor, tensor_n = tensor + g["tensor"] * g["n"], tensor_n + g["n"]
        return WindowStats(
            label=label,
            minutes=minutes,
            covered_minutes=covered,
            util=util / n if n else None,
            util_peak=peak,
            vram=vram / n if n else None,
            power=power / n if n else None,
            temp_peak=temp,
            idle_share=idle / n if n else None,
            busy_share=busy / n if n else None,
            held_idle_share=held / n if n else None,
            cpu=cpu / cn if cn else None,
            mem=mem / cn if cn else None,
            sm=sm / sm_n if sm_n else None,
            tensor=tensor / tensor_n if tensor_n else None,
            sm_coverage=min(1.0, sm_n / n) if n else None,
        )

    def heatmap(
        self, gpu: Optional[str], hours: int = 25, now: Optional[float] = None, metric: str = "util"
    ) -> tuple[int, list[Optional[float]]]:
        """(start_ms, per-minute mean of ``metric``) for the last ``hours`` hours; None = not recorded.

        ``metric`` is a GPU bucket key: "util" (default), "vram", "power", or the optional GPM keys
        "sm" / "tensor" (None for minutes recorded without GPM).
        """
        count = hours * 60
        start = minute_start((now if now is not None else time.time()) * 1000) - (count - 1) * MINUTE_MS
        cells: list[Optional[float]] = [None] * count
        for b in self._each_since(start):
            i = (b["t"] - start) // MINUTE_MS
            g = b["gpus"].get(gpu) if gpu else next(iter(b["gpus"].values()), None)
            if 0 <= i < count and g:
                cells[i] = g.get(metric)
        return start, cells

    # ---- persistence ----
    def snapshot(self) -> dict[str, Any]:
        out = list(self._each_since(0))
        out.reverse()
        d: dict[str, Any] = {"version": 1, "minutes": out}
        if self.cleared_at:
            d["clearedAt"] = self.cleared_at
        return d

    @staticmethod
    def _keep_from(cleared_at: int) -> int:
        """A clear removes everything through the end of its minute: a bucket can't be split,
        so up to 60 s recorded right after the clear is dropped rather than leaving old data."""
        return minute_start(cleared_at) + MINUTE_MS if cleared_at else 0

    def _apply_clear(self, cleared_at: int) -> None:
        """Drop data from before another process's clear, keeping later minutes."""
        self.cleared_at = cleared_at
        keep_from = self._keep_from(cleared_at)
        i = bisect.bisect_left(self._times, keep_from)
        del self.minutes[:i], self._times[:i]
        while self.recent and self.recent[0].t * 1000 < cleared_at:
            self.recent.popleft()
        if self._acc and self._acc.t < keep_from:
            self._acc = None

    def load(self, data: Any) -> None:
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("minutes"), list):
            return
        cleared = _int(data.get("clearedAt"))
        if cleared > self.cleared_at:
            self._apply_clear(cleared)
        keep_from = self._keep_from(self.cleared_at)
        for raw in data["minutes"]:
            m = _normalize(raw)
            if m and m["t"] >= keep_from and not (self._acc and m["t"] >= self._acc.t):
                self._insert(m, merge=False)

    def clear(self, now_ms: Optional[int] = None) -> None:
        self.recent.clear()
        self.minutes, self._times, self._acc = [], [], None
        self.cleared_at = now_ms or int(time.time() * 1000)

    def to_csv(self) -> str:
        rows = [
            "minute_utc,gpu_uuid,samples,gpu_util_mean,gpu_util_max,vram_pct_mean,power_w_mean,"
            "temp_c_max,cpu_pct_mean,ram_pct_mean"
        ]
        for b in self.snapshot()["minutes"]:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(b["t"] / 1000))
            for uid, g in b["gpus"].items() or [("", None)]:
                vals = (
                    [g["n"], g["util"], g["utilMax"], g["vram"], g["power"], g["tempMax"]]
                    if g
                    else [b["n"], "", "", "", "", ""]
                )
                rows.append(",".join(str(x) for x in [ts, uid, *vals, b["cpu"], b["mem"]]))
        return "\n".join(rows) + "\n"


class _ReadError(Exception):
    pass


class HistoryFile:
    """JSON persistence: atomic writes, merge-on-save, and a lock file so concurrent
    writers (dashboard, ``log``, ``clear``) never drop each other's minutes."""

    LOCK_WAIT_S = 3.0
    LOCK_STALE_S = 15.0

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_name(path.name + ".lock")

    def _read_strict(self) -> Any:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except ValueError as e:  # corrupt JSON: keep a copy, then start fresh
            backup = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
            logger.warning("Corrupt history file %s (%s); moved to %s", self.path, e, backup)
            try:
                os.replace(self.path, backup)
            except OSError:
                pass
            return None
        except OSError as e:  # transient (locked, permissions): don't overwrite what we couldn't read
            raise _ReadError(str(e)) from e

    def read(self) -> Any:
        try:
            return self._read_strict()
        except _ReadError as e:
            logger.warning("Could not read history file %s: %s", self.path, e)
            return None

    @contextmanager
    def _locked(self) -> Iterator[bool]:
        deadline = time.monotonic() + self.LOCK_WAIT_S
        fd = None
        while fd is None:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - self.lock_path.stat().st_mtime > self.LOCK_STALE_S:
                        self.lock_path.unlink()  # holder crashed
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    yield False
                    return
                time.sleep(0.05)
        try:
            yield True
        finally:
            os.close(fd)
            try:
                self.lock_path.unlink()
            except OSError:
                pass

    def save(self, store: HistoryStore) -> bool:
        """Merge the on-disk history into ``store`` and write it back. False if skipped or failed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked() as ok:
            if not ok:
                logger.warning("History file is locked by another process; will retry on next save")
                return False
            try:
                store.load(self._read_strict())
            except _ReadError as e:
                logger.warning("Skipping save, could not read %s: %s", self.path, e)
                return False
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(store.snapshot(), separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self.path)
            return True
