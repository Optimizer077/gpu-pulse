"""Cross-GPU analysis for multi-GPU machines.

``intel.diagnose`` looks at one GPU at a time. This module compares the GPUs with each other over
the last two minutes of raw samples and reports what only shows up side by side:

* straggler / imbalance – one GPU far less active while its siblings are busy (data-parallel
  ranks wait for the slowest one every step)
* uneven VRAM            – the fullest GPU runs out of memory first and caps everyone's batch
* lone throttling        – one GPU power- or thermal-limited while its siblings are not
* mismatched PCIe links  – one GPU on a narrower or slower link than the others

It is honest about what it cannot know: the GPUs may run unrelated jobs, so findings say what
the pattern means *if* they share one job. Only GPUs that show signs of taking part in work
(VRAM in use, or a compute process) are compared; when the process list is known, GPUs are
only compared with siblings that run a process of the same name. An idle GPU next to a busy
one is reported as "not in use", never as a straggler. Cheap enough for the dashboard's 2 s path.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from ._intel_gpu import POWER_REASONS, THERMAL_REASONS
from .history import HistoryStore
from .models import GpuDetails, GpuProcess, GpuSample, Insight, InsightLevel

FLEET_WINDOW_S = 120.0
MIN_SAMPLES = 10  # per GPU, before it is compared with the others ...
MIN_SPAN_S = 30.0  # ... and at least this many seconds of them
IN_USE_VRAM = 10.0  # VRAM use (%) that marks a GPU as taking part in work (no process list needed)
BUSY = 50.0  # siblings' mean activity (%) before an imbalance is judged
LAG_RATIO, LAG_GAP = 0.5, 30.0  # a GPU lags when below half the others' mean and 30 points under it
VRAM_SPREAD = 30.0  # percentage points between the fullest and emptiest GPU
THROTTLE_ALONE, THROTTLE_QUIET = 0.3, 0.05  # share of samples throttled: this GPU vs every sibling
LINK_BUSY = 30.0  # PCIe generation drops at idle to save power: compare generations only under load
LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}  # per lane, per direction
_RANK = {"good": 0, "info": 1, "warning": 2, "serious": 3, "critical": 4}


@dataclass(frozen=True)
class GpuStats:
    """One GPU over the analysis window."""

    index: int
    uuid: str
    samples: int
    activity: float  # mean SM activity (basis "sm") or utilization (basis "util"), %
    busy_share: float  # share of samples with utilization >= 50%
    vram: Optional[float]  # latest VRAM use, %
    temp_max: Optional[float]
    power_share: float  # share of samples power-capped
    thermal_share: float  # share of samples clock-limited by temperature
    power_limit: Optional[float]
    link: Optional[tuple[int, int]]  # latest (PCIe generation, width)
    util_now: float
    span_s: float = 0.0  # seconds between this GPU's first and last sample in the window
    procs: Optional[frozenset[str]] = None  # compute process names on it; None = process list unknown

    @property
    def in_use(self) -> bool:
        """Shows signs of taking part in work: VRAM in use or a compute process on it."""
        return bool(self.procs) or (self.vram or 0.0) >= IN_USE_VRAM


def related(a: GpuStats, b: GpuStats) -> bool:
    """Could ``a`` and ``b`` run one job? Both in use, and a shared process name when both lists are known."""
    if not (a.in_use and b.in_use):
        return False
    return not (a.procs and b.procs) or bool(a.procs & b.procs)


def span_text(seconds: float) -> str:
    """The real length of the compared window: "47 s", "2 min"."""
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"


@dataclass(frozen=True)
class GpuFlag:
    """Per-GPU marker for the All GPUs panel: a health level and short tags."""

    level: InsightLevel = "good"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fleet:
    findings: tuple[Insight, ...] = ()
    flags: dict[str, GpuFlag] = field(default_factory=dict)  # by GPU uuid
    basis: str = "util"  # "sm" when every GPU reports GPM SM activity
    stats: tuple[GpuStats, ...] = ()


@dataclass(frozen=True)
class Aggregate:
    """Machine-wide totals from one sample (``status`` and the All GPUs panel)."""

    count: int
    util_mean: Optional[float]
    vram_used_gib: float
    vram_total_gib: float
    power_w: Optional[float]
    power_limit_w: Optional[float]
    hottest: Optional[tuple[int, float]]  # (index, °C)
    throttled: tuple[int, ...]  # indices with an active power/thermal clock limit


def aggregate(gpus: Sequence[GpuSample]) -> Aggregate:
    utils = [g.util for g in gpus if g.util is not None]
    temps = [(g.index, g.temp_c) for g in gpus if g.temp_c is not None]
    powers = [g.power_w for g in gpus if g.power_w is not None]
    limits = [g.power_limit_w for g in gpus if g.power_limit_w]
    limited = POWER_REASONS | THERMAL_REASONS
    return Aggregate(
        count=len(gpus),
        util_mean=sum(utils) / len(utils) if utils else None,
        vram_used_gib=sum(g.mem_used_mib or 0.0 for g in gpus) / 1024,
        vram_total_gib=sum(g.mem_total_mib or 0.0 for g in gpus) / 1024,
        power_w=sum(powers) if powers else None,
        power_limit_w=sum(limits) if limits else None,
        hottest=max(temps, key=lambda x: x[1]) if temps else None,
        throttled=tuple(g.index for g in gpus if any(r in limited for r in g.throttle)),
    )


def _stats(
    history: HistoryStore, now: float, processes: Optional[Sequence[GpuProcess]] = None
) -> tuple[list[GpuStats], str]:
    per: dict[str, list[GpuSample]] = {}
    times: dict[str, list[float]] = {}
    for s in history.recent_since(now - FLEET_WINDOW_S):
        if s.t > now + 1:
            continue
        for g in s.gpus:
            if g.util is not None:
                per.setdefault(g.uuid, []).append(g)
                times.setdefault(g.uuid, []).append(s.t)
    spans = {u: ts[-1] - ts[0] for u, ts in times.items()}
    groups = [rows for u, rows in per.items() if len(rows) >= MIN_SAMPLES and spans[u] >= MIN_SPAN_S]
    use_sm = bool(groups) and all(sum(g.sm_active is not None for g in rows) >= 0.8 * len(rows) for rows in groups)
    names: dict[str, set[str]] = {}
    for p in processes or ():
        names.setdefault(p.gpu_uuid, set()).add(p.name)
    out: list[GpuStats] = []
    for rows in groups:
        last = rows[-1]
        n = len(rows)
        if use_sm:
            vals = [g.sm_active for g in rows if g.sm_active is not None]
        else:
            vals = [g.util for g in rows if g.util is not None]
        temps = [g.temp_c for g in rows if g.temp_c is not None]
        out.append(
            GpuStats(
                index=last.index,
                uuid=last.uuid,
                samples=n,
                activity=sum(vals) / len(vals) if vals else 0.0,
                busy_share=sum((g.util or 0) >= BUSY for g in rows) / n,
                vram=last.vram_pct,
                temp_max=max(temps) if temps else None,
                power_share=sum(any(r in POWER_REASONS for r in g.throttle) for g in rows) / n,
                thermal_share=sum(any(r in THERMAL_REASONS for r in g.throttle) for g in rows) / n,
                power_limit=last.power_limit_w,
                link=(last.pcie_gen, last.pcie_width) if last.pcie_gen and last.pcie_width else None,
                util_now=last.util or 0.0,
                span_s=spans[last.uuid],
                # A known process list with nothing on this GPU is an empty set; no list at all is None.
                procs=frozenset(names.get(last.uuid, ())) if processes is not None else None,
            )
        )
    out.sort(key=lambda x: x.index)
    return out, "sm" if use_sm else "util"


Found = tuple[Insight, Optional[str], str]  # (finding, uuid it flags or None, short tag)


def analyze_fleet(
    history: HistoryStore,
    details: Sequence[GpuDetails] = (),
    now: Optional[float] = None,
    processes: Optional[Sequence[GpuProcess]] = None,
) -> Fleet:
    """Compare every GPU with its siblings; an empty ``Fleet`` for fewer than two GPUs.

    ``processes`` (compute processes with their GPU uuid) narrows comparisons to GPUs that run a
    process of the same name; ``None`` means the process list is unavailable.
    """
    now = time.time() if now is None else now
    stats, basis = _stats(history, now, processes)
    if len(stats) < 2:
        return Fleet(stats=tuple(stats), basis=basis)
    found: list[Found] = []
    found += _stragglers(stats, basis)
    found += _uneven_vram(stats)
    found += _lone_throttle(stats)
    found += _links(stats, {d.uuid: d for d in details})
    tags: dict[str, list[str]] = {s.uuid: [] for s in stats}
    worst: dict[str, InsightLevel] = {s.uuid: "good" for s in stats}
    for ins, uid, tag in found:
        if uid is None:
            continue
        tags[uid].append(tag)
        if _RANK[ins.level] > _RANK[worst[uid]]:
            worst[uid] = ins.level
    order = {"critical": 0, "serious": 1, "warning": 2, "info": 3, "good": 4}
    findings = tuple(sorted((f for f, _, _ in found), key=lambda i: order.get(i.level, 5)))
    flags = {s.uuid: GpuFlag(worst[s.uuid], tuple(tags[s.uuid])) for s in stats}
    return Fleet(findings=findings, flags=flags, basis=basis, stats=tuple(stats))


def _ids(gs: Sequence[GpuStats]) -> str:
    """Name GPUs for a sentence: "GPU 0", "GPUs 0, 2"."""
    return ("GPU " if len(gs) == 1 else "GPUs ") + ", ".join(str(g.index) for g in gs)


def _stragglers(stats: Sequence[GpuStats], basis: str) -> list[Found]:
    out: list[Found] = []
    label = "SM activity" if basis == "sm" else "Utilization"
    for s in stats:
        others = [o for o in stats if o is not s and related(s, o)]
        if not others:
            continue
        mean = sum(o.activity for o in others) / len(others)
        if mean < BUSY or s.activity > LAG_RATIO * mean or mean - s.activity < LAG_GAP:
            continue
        bursty = 0.05 <= s.busy_share <= 0.5
        shared = sorted(set.intersection(set(s.procs or ()), *(set(o.procs or ()) for o in others)))
        # Same process name on every compared GPU is real evidence of one job; VRAM alone is not.
        level: InsightLevel = "warning" if shared else "info"
        span = min(g.span_s for g in (s, *others))
        evidence = (
            f"All of them run {shared[0]}."
            if shared
            else "All of them hold VRAM, but the process list can't confirm they run one job."
        )
        detail = (
            f"{label} {s.activity:.0f}% on GPU {s.index} vs {mean:.0f}% on {_ids(others)} over the last "
            f"{span_text(span)}{', in short bursts' if bursty else ''}. {evidence} "
            "If these GPUs run one data-parallel job, every step waits for this rank, so its idle time "
            "is lost on all of them. Typical causes: an uneven batch or shard split, this rank's data "
            "loader falling behind, or the rank blocked in an NCCL collective waiting on a peer. "
            "Compare per-rank step times (torch.profiler, NCCL_DEBUG=INFO). Unrelated jobs per GPU: ignore this."
        )
        title = f"GPU {s.index} lags its siblings: a straggler if they run one job"
        out.append((Insight(level, title, detail), s.uuid, "straggler"))
    out += _idle(stats, label)
    return out


def _idle(stats: Sequence[GpuStats], label: str) -> list[Found]:
    """GPUs not in use next to busy ones: worth knowing, but nobody is to blame."""
    workers = [s for s in stats if s.in_use and s.activity >= BUSY]
    idle = [s for s in stats if not s.in_use and s.activity < LAG_RATIO * BUSY]
    if not workers or not idle:
        return []
    known = all(s.procs is not None for s in idle)
    one = len(idle) == 1
    vram = ", ".join(f"{s.vram:.0f}%" for s in idle if s.vram is not None) or "none"
    detail = (
        f"{_ids(idle)} {'uses' if one else 'use'} {vram} of {'its' if one else 'their'} VRAM"
        f"{(' and runs' if one else ' and run') + ' no compute process' if known else ''}, so "
        f"{'it is' if one else 'they are'} "
        f"not taking part in the work on {_ids(workers)} ({label} "
        f"{max(w.activity for w in workers):.0f}%). Fine if that job needs fewer GPUs. If these GPUs should "
        "run one job, the launcher isn't using them: check CUDA_VISIBLE_DEVICES and the world size "
        "(torchrun --nproc_per_node, accelerate --num_processes)."
    )
    verb = "works" if len(workers) == 1 else "work"
    title = f"{_ids(idle)} idle while {_ids(workers)} {verb}"
    return [(Insight("info", title, detail), None, "")]


def _uneven_vram(stats: Sequence[GpuStats]) -> list[Found]:
    known = [s for s in stats if s.vram is not None and s.in_use]
    if len(known) < 2 or not any(s.util_now >= BUSY or s.busy_share >= 0.5 for s in known):
        return []
    hi = max(known, key=lambda s: s.vram or 0.0)
    peers = [o for o in known if o is not hi and related(hi, o)]
    if not peers:
        return []
    lo = min(peers, key=lambda s: s.vram or 0.0)
    if (hi.vram or 0) < 50 or (hi.vram or 0) - (lo.vram or 0) < VRAM_SPREAD:
        return []
    level: InsightLevel = "warning" if (hi.vram or 0) >= 90 else "info"
    detail = (
        f"GPU {hi.index} holds {hi.vram:.0f}% of its VRAM, GPU {lo.index} only {lo.vram:.0f}%. If these GPUs "
        "run one job, the fullest GPU runs out of memory first and caps the batch size for every rank. In "
        "data-parallel jobs this usually means extra state on one rank (rank-0 eval/optimizer buffers, or other "
        "ranks opening CUDA contexts on GPU 0: call torch.cuda.set_device(local_rank) first); in model or "
        "pipeline parallel jobs, an unbalanced split. FSDP / ZeRO sharding or moving layers between stages "
        "evens it out. Unrelated jobs per GPU: ignore this."
    )
    pair = f"GPU {hi.index} {hi.vram:.0f}% vs GPU {lo.index} {lo.vram:.0f}%"
    title = f"VRAM uneven across GPUs: {pair} (if they run one job)"
    return [(Insight(level, title, detail), hi.uuid, f"VRAM {hi.vram:.0f}%")]


def _lone_throttle(stats: Sequence[GpuStats]) -> list[Found]:
    out: list[Found] = []
    for kind in ("thermal", "power"):
        share = (lambda s: s.thermal_share) if kind == "thermal" else (lambda s: s.power_share)
        for s in stats:
            if share(s) < THROTTLE_ALONE:
                continue
            # Only busy siblings that could share its job count: an idle GPU never throttles.
            others = [o for o in stats if o is not s and related(s, o) and o.busy_share >= 0.5]
            if not others or any(share(o) > THROTTLE_QUIET for o in others):
                continue
            out.append(_throttle_finding(kind, s, others, share(s)))
    return out


def _throttle_finding(kind: str, s: GpuStats, others: Sequence[GpuStats], share: float) -> Found:
    who = _ids(others)
    if kind == "thermal":
        peers = max((o.temp_max for o in others if o.temp_max is not None), default=None)
        temps = f" (peak {s.temp_max:.0f}°C vs {peers:.0f}°C on {who})" if s.temp_max and peers else ""
        detail = (
            f"GPU {s.index} is clock-limited by temperature in {share:.0%} of samples{temps}; busy {who} "
            "run unthrottled. A throttled GPU runs slower; if these GPUs run one job, it becomes the straggler "
            "every step waits for. Check its fan and airflow (cards in adjacent slots run hottest), dust and "
            f"thermal paste; a slightly lower power limit (nvidia-smi -i {s.index} -pl <watts>) trades a little "
            "peak for steady clocks."
        )
        return (
            Insight("warning", f"GPU {s.index} thermal-throttles while its siblings don't", detail),
            s.uuid,
            "thermal",
        )
    caps = [o.power_limit for o in others if o.power_limit]
    if s.power_limit and caps and s.power_limit < max(caps) - 5:
        why = (
            f"Its power limit is {s.power_limit:.0f} W vs {max(caps):.0f} W on {who}: "
            f"raise it with nvidia-smi -i {s.index} -pl {max(caps):.0f} if the board allows."
        )
    else:
        why = (
            "Same limit as its siblings: it draws more power at its load (board variation, higher temperature) "
            "or runs heavier work than they do."
        )
    detail = (
        f"GPU {s.index} runs at its power cap in {share:.0%} of samples while busy {who} don't. If these GPUs "
        f"run one job, capped clocks make it the slow rank in synchronized training. {why}"
    )
    return Insight("warning", f"GPU {s.index} power-capped while its siblings aren't", detail), s.uuid, "power cap"


def _bandwidth(link: tuple[int, int]) -> float:
    gen, width = link
    return LANE_GBS.get(gen, LANE_GBS[6] if gen > 6 else 0.25) * width


def _links(stats: Sequence[GpuStats], details: dict[str, GpuDetails]) -> list[Found]:
    links = {s.uuid: s.link for s in stats if s.link is not None}
    linked = [s for s in stats if s.uuid in links]
    if len(linked) < 2:
        return []
    busy = {s.uuid for s in linked if s.util_now >= LINK_BUSY}
    max_w = max(w for _, w in links.values())
    max_gen = max((links[u][0] for u in busy), default=0)

    def effective(s: GpuStats) -> tuple[int, int]:
        """An idle GPU's generation reads low (power saving): judge it at the busy GPUs' generation."""
        gen, width = links[s.uuid]
        return (gen if s.uuid in busy or not max_gen else max(gen, max_gen)), width

    out: list[Found] = []
    for s in linked:
        gen, width = links[s.uuid]
        narrow = width < max_w
        slow = s.uuid in busy and len(busy) >= 2 and gen < max_gen
        if not (narrow or slow):
            continue
        ref = max((o for o in linked if o is not s), key=lambda o: _bandwidth(effective(o)))
        ref_link = effective(ref)
        mine, theirs = _bandwidth(effective(s)), _bandwidth(ref_link)
        if mine >= theirs:
            continue
        out.append(_link_finding(s, gen, width, narrow, mine, (ref, ref_link, theirs), details.get(s.uuid)))
    return out


def _link_finding(
    s: GpuStats,
    gen: int,
    width: int,
    narrow: bool,
    mine: float,
    ref: tuple[GpuStats, tuple[int, int], float],
    d: Optional[GpuDetails],
) -> Found:
    """A narrower / slower link than a sibling: a slot problem only if the card itself can do more."""
    other, (ref_gen, ref_width), theirs = ref
    busy = s.util_now >= LINK_BUSY
    card_w, card_gen = (d.pcie_max_width, d.pcie_max_gen) if d else (None, None)
    # Below the card's own maximum (generation only under load: links downshift at idle to save power).
    below = bool(card_w and width < card_w) or bool(card_gen and busy and gen < card_gen)
    at_max = not below and bool(card_w if narrow else card_gen)
    kind = "narrower" if narrow else "slower"
    compare = (
        f"GPU {s.index} links at PCIe Gen{gen} x{width}, about {mine:.0f} GB/s each way, vs Gen{ref_gen} "
        f"x{ref_width} (~{theirs:.0f} GB/s) on GPU {other.index}. If these GPUs run one job, host-to-GPU copies "
        "(input batches, checkpoint loads) take longer on this one."
    )
    if at_max:
        limit = f"x{width}" if narrow else f"Gen{gen}"
        detail = (
            f"{compare} That is the card's own maximum (Gen{card_gen or gen} x{card_w or width}), not the slot: "
            "nothing to fix unless the copies are a bottleneck. GPU-to-GPU NVLink traffic is unaffected."
        )
        title = f"GPU {s.index} on a {kind} PCIe link (Gen{gen} x{width}): card limit ({limit} by design)"
        return Insight("info", title, detail), None, ""
    if below:
        can = f"x{card_w}" if card_w and width < card_w else f"Gen{card_gen}"
        cause = (
            f"The card supports {can}, so it is usually the slot: chipset-attached or shared-lane slots, a riser "
            "cable, or BIOS lane bifurcation."
        )
    else:
        cause = (
            "If the card supports a wider or faster link (nvidia-smi -q, 'Max' under PCIe), it is usually the "
            "slot: chipset-attached or shared-lane slots, a riser cable, or BIOS lane bifurcation."
        )
    detail = f"{compare} {cause} GPU-to-GPU NVLink traffic is unaffected."
    level: InsightLevel = "warning" if below and busy and mine <= 0.5 * theirs else "info"
    title = f"GPU {s.index} on a {kind} PCIe link (Gen{gen} x{width})"
    return Insight(level, title, detail), s.uuid, f"x{width} link" if narrow else f"Gen{gen} link"


def gpu_flag(g: GpuSample, fleet: Optional[Fleet], temp_warn: float = 80.0) -> GpuFlag:
    """Health dot + tags for one GPU row: cross-GPU flags combined with its own live state."""
    base = fleet.flags.get(g.uuid, GpuFlag()) if fleet else GpuFlag()
    level: InsightLevel = base.level
    if g.util is None:
        return GpuFlag("info", base.tags)
    own: InsightLevel = "good"
    t, v = g.temp_c or 0.0, g.vram_pct or 0.0
    if t >= temp_warn:
        own = "critical"
    elif v >= 95:
        own = "serious"
    elif t >= temp_warn - 10 or any(r in POWER_REASONS | THERMAL_REASONS for r in g.throttle):
        own = "warning"
    if _RANK[own] > _RANK[level]:
        level = own
    return GpuFlag(level, base.tags)
