"""Measure the terminal dashboard's CPU cost at the shipped defaults.

    uv run --with psutil python scripts/bench.py [--seconds 15] [--runs 3]

Runs the dashboard headless (Textual ``run_test``) and reports process CPU time over a
window as a share of one core. Scenarios: real GPU Overview, 4-GPU demo Overview, real GPU
History tab. Headless runs skip writing to a real terminal, so real use costs a bit more.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from typing import Optional

import psutil

from gpu_pulse.app import GpuDashboard
from gpu_pulse.monitor import Monitor, MonitorConfig


async def measure(demo: bool, gpus: int, key: Optional[str], seconds: float) -> float:
    me = psutil.Process()
    mon = Monitor(MonitorConfig(interval_s=0.5, demo=demo, demo_gpus=gpus, persist=False))
    app = GpuDashboard(mon)
    async with app.run_test(size=(180, 52)) as pilot:
        await pilot.pause(4)  # let start-up (boot sweep, specs probe) finish
        if key:
            await pilot.press(key)
            await pilot.pause(1)
        c0 = sum(me.cpu_times()[:2])
        await pilot.pause(seconds)
        c1 = sum(me.cpu_times()[:2])
    mon.close()
    return 100 * (c1 - c0) / seconds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--runs", type=int, default=3)
    a = p.parse_args()
    for label, demo, gpus, key in (
        ("real GPU, Overview", False, 1, None),
        ("demo, 4 GPUs, Overview", True, 4, None),
        ("real GPU, History tab", False, 1, "4"),
    ):
        vals = [asyncio.run(measure(demo, gpus, key, a.seconds)) for _ in range(a.runs)]
        runs = ", ".join(f"{v:.1f}" for v in vals)
        print(f"{label:24s} median {statistics.median(vals):5.1f}% of a core  (runs: {runs})")


if __name__ == "__main__":
    main()
