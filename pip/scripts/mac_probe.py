"""Read-only diagnostic for macOS: what gpu-pulse sees on this Mac, and what reading it costs.

    uv run python scripts/mac_probe.py [--runs 20] [--save-fixtures tests/fixtures]

Prints:
1. the machine (macOS version, CPU architecture, Python, total memory),
2. every IOAccelerator in ``ioreg -r -d 1 -w 0 -c IOAccelerator -a``: its class, top-level key names,
   the raw PerformanceStatistics keys with their values, and what gpu-pulse parsed from it,
3. ``system_profiler SPDisplaysDataType -json``: each GPU's keys and values (display lists and
   serial numbers removed) and the parsed result,
4. the cost of one ioreg read over ``--runs`` runs (wall time, and CPU time of the child process),
5. which collector and sampling interval gpu-pulse picks, and ``gpu-pulse status`` (host name hidden).

Nothing is written unless ``--save-fixtures DIR`` is given; it then saves ``ioreg_mac_capture.plist``
and ``system_profiler_mac_capture.json`` there for the test suite. Serial numbers are never printed
or saved: display entries (``spdisplays_ndrvs``) are dropped and any key containing "serial" or
"uuid" is removed at every depth.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import plistlib
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from gpu_pulse.collectors.apple import (
    IOREG_ARGS,
    IOREG_TIMEOUT_S,
    MIN_INTERVAL_S,
    PROFILER_ARGS,
    PROFILER_TIMEOUT_S,
    _tool,
    apple_details,
    parse_ioreg,
    parse_profiler_json,
    run_command,
    system_memory_mib,
)
from gpu_pulse.monitor import Monitor, MonitorConfig

PRIVATE = ("serial", "uuid")
DROP = ("spdisplays_ndrvs",)


def scrub(v: Any) -> Any:
    """Copy of ``v`` without display lists or keys that may hold serial numbers / hardware ids."""
    if isinstance(v, dict):
        return {k: scrub(x) for k, x in v.items() if k not in DROP and not any(p in str(k).lower() for p in PRIVATE)}
    if isinstance(v, list):
        return [scrub(x) for x in v]
    return v


def show(v: Any) -> str:
    if isinstance(v, bytes):
        return f"<data {len(v)} bytes: {v[:48]!r}>"
    if isinstance(v, (dict, list)):
        return f"<{type(v).__name__} of {len(v)}>"
    return repr(v)


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def machine() -> None:
    section("machine")
    mem = system_memory_mib()
    print(f"macOS {platform.mac_ver()[0] or '?'} · {platform.machine()} · Python {platform.python_version()}")
    print(f"total memory (unified total on Apple Silicon): {mem / 1024:.1f} GB" if mem else "total memory: unknown")


def probe_ioreg(save: Optional[Path]) -> Optional[bytes]:
    section("ioreg " + " ".join(IOREG_ARGS))
    t0 = time.perf_counter()
    data = run_command((_tool("ioreg"), *IOREG_ARGS), IOREG_TIMEOUT_S)
    print(f"{'failed' if data is None else f'{len(data)} bytes'} in {(time.perf_counter() - t0) * 1000:.0f} ms")
    if not data:
        return None
    try:
        entries = plistlib.loads(data)
    except Exception as e:  # noqa: BLE001 - report anything the parser would also reject
        print(f"not a plist: {e}")
        return data
    entries = entries if isinstance(entries, list) else [entries]
    for i, e in enumerate(x for x in entries if isinstance(x, dict)):
        e = scrub(e)
        cores = e.get("gpu-core-count")
        print(f"\n[{i}] IOClass={e.get('IOClass')!r} model={show(e.get('model'))} gpu-core-count={cores}")
        print("  top-level keys: " + ", ".join(sorted(map(str, e))))
        ps = e.get("PerformanceStatistics")
        if isinstance(ps, dict):
            print("  PerformanceStatistics:")
            for k in sorted(ps, key=str):
                print(f"    {k!s:<36} {show(ps[k])}")
        else:
            print("  PerformanceStatistics: missing")
    print("\nparsed by gpu-pulse:")
    rows = parse_ioreg(data, system_memory_mib())
    for g, info in rows:
        shown = {k: v for k, v in vars(g).items() if v not in (None, ())}
        print(f"  {shown}\n  {info}")
    if not rows:
        print("  (no GPU) -> gpu-pulse would fall back to NVIDIA/AMD detection and show CPU/RAM only")
    if save:
        clean = [scrub(x) for x in entries if isinstance(x, dict)]
        (save / "ioreg_mac_capture.plist").write_bytes(plistlib.dumps(clean))
        print(f"saved {save / 'ioreg_mac_capture.plist'}")
    return data


def probe_profiler(ioreg: Optional[bytes], save: Optional[Path]) -> None:
    section("system_profiler " + " ".join(PROFILER_ARGS))
    t0 = time.perf_counter()
    data = run_command((_tool("system_profiler"), *PROFILER_ARGS), PROFILER_TIMEOUT_S)
    print(f"{'failed' if data is None else f'{len(data)} bytes'} in {(time.perf_counter() - t0) * 1000:.0f} ms")
    if not data:
        return
    try:
        root = scrub(json.loads(data.decode("utf-8", "replace")))
    except ValueError as e:
        print(f"not JSON: {e}")
        return
    for gpu in root.get("SPDisplaysDataType", []) if isinstance(root, dict) else []:
        print("")
        for k in sorted(gpu, key=str):
            print(f"  {k!s:<40} {show(gpu[k])}")
    prof = parse_profiler_json(data)
    print("\nparsed by gpu-pulse:")
    for p in prof:
        print(f"  {p}")
    rows = parse_ioreg(ioreg, system_memory_mib()) if ioreg else []
    for d in apple_details([g for g, _ in rows], [i for _, i in rows], prof):
        known = {k: v for k, v in vars(d).items() if v is not None}
        print(f"  details: {known}")
    if save:
        root["_fixture_note"] = f"real capture (scripts/mac_probe.py, {time.strftime('%Y-%m-%d')}), serials removed"
        (save / "system_profiler_mac_capture.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
        print(f"saved {save / 'system_profiler_mac_capture.json'}")


def child_cpu_s() -> Optional[float]:
    try:
        import resource
    except ImportError:  # not POSIX
        return None
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def cost(runs: int) -> None:
    section(f"ioreg cost over {runs} runs")
    walls: list[float] = []
    cpus: list[float] = []
    parse_ms: list[float] = []
    mem = system_memory_mib()
    for _ in range(runs):
        c0, t0 = child_cpu_s(), time.perf_counter()
        data = run_command((_tool("ioreg"), *IOREG_ARGS), IOREG_TIMEOUT_S)
        t1, c1 = time.perf_counter(), child_cpu_s()
        walls.append((t1 - t0) * 1000)
        if c0 is not None and c1 is not None:
            cpus.append((c1 - c0) * 1000)
        if data:
            p0 = time.perf_counter()
            parse_ioreg(data, mem)
            parse_ms.append((time.perf_counter() - p0) * 1000)

    def stats(xs: list[float]) -> str:
        if not xs:
            return "n/a"
        p95 = sorted(xs)[max(0, round(0.95 * len(xs)) - 1)]
        return f"median {statistics.median(xs):.1f} ms · p95 {p95:.1f} ms · max {max(xs):.1f} ms"

    print(f"wall (spawn + read): {stats(walls)}")
    print(f"child CPU:           {stats(cpus)}")
    print(f"parse in Python:     {stats(parse_ms)}")
    if cpus:
        per = statistics.median(cpus) + (statistics.median(parse_ms) if parse_ms else 0.0)
        for interval in (0.5, MIN_INTERVAL_S, 2.0):
            print(f"  sampling every {interval:g} s ≈ {per / (interval * 1000) * 100:.1f}% of one core for GPU reads")
    print(f"current floor: MIN_INTERVAL_S = {MIN_INTERVAL_S:g} s (collectors/apple.py)")


def status() -> None:
    section("gpu-pulse status")
    mon = Monitor(MonitorConfig(interval_s=0.5, persist=False))
    print(f"collector: {mon.vendor} · dashboard interval at -i 0.5: {mon.cfg.interval_s:g} s")
    mon.close()
    env = dict(os.environ, COLUMNS="120", NO_COLOR="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, "-m", "gpu_pulse", "status"], capture_output=True, timeout=60, env=env, check=False
    )
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    host = socket.gethostname()
    print(out.replace(host, "<host>") if host else out)


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only report of what gpu-pulse sees on this Mac.")
    p.add_argument("--runs", type=int, default=20, help="ioreg runs for the cost measurement (default 20)")
    p.add_argument("--save-fixtures", type=Path, metavar="DIR", help="also save scrubbed captures for the tests")
    a = p.parse_args()
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:  # a non-UTF-8 console must not crash the report
        reconfigure(errors="replace")
    if sys.platform != "darwin":
        print("This probe is for macOS; on other systems gpu-pulse does not use ioreg.")
    if a.save_fixtures:
        a.save_fixtures.mkdir(parents=True, exist_ok=True)
    machine()
    ioreg = probe_ioreg(a.save_fixtures)
    probe_profiler(ioreg, a.save_fixtures)
    cost(max(1, a.runs))
    status()


if __name__ == "__main__":
    main()
