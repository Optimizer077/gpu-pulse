"""Dev-only: run the dashboard headless and save an SVG screenshot.

    uv run python scripts/screenshot.py out.svg [--demo | --live-demo [--demo-vendor apple]] [--size 160x62]
        [--keys 2] [--wait 4]

--keys presses keys before the capture (comma-separated Textual key names, e.g. "2" for the
GPU tab or "question_mark" for the help overlay).

--demo seeds a temporary history file with a synthetic day of training-like usage
so the averages, heatmap and sparklines have something to show. Live readings
(gauges, processes, specs) always come from this machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import tempfile
import time
from pathlib import Path

from gpu_pulse.app import GpuDashboard
from gpu_pulse.monitor import Monitor, MonitorConfig


def seed(path: Path, gpu_uuid: str) -> None:
    rnd = random.Random(7)
    now_min = int(time.time() // 60 * 60_000)
    minutes = []
    for i in range(25 * 60, 0, -1):
        t = now_min - i * 60_000
        hour = time.localtime(t / 1000).tm_hour
        if 2 <= hour < 8:
            continue
        phase = (t // 60_000) % 240
        util = (
            3 + rnd.random() * 4 if phase < 40 else max(0, min(100, 78 + 14 * math.sin(t / 2.8e6) - rnd.random() * 15))
        )
        n = 60
        busy = int(n * (util / 100) ** 1.5)
        idle = n - busy if util < 10 else int(n * 0.05)
        minutes.append(
            {
                "t": t,
                "n": n,
                "cpu": 30 + rnd.random() * 20,
                "mem": 48,
                "gpus": {
                    gpu_uuid: {
                        "n": n,
                        "util": round(util, 1),
                        "utilMax": min(100, util + 12),
                        "vram": 70 if util > 10 else 34,
                        "power": 20 + util * 1.4,
                        "tempMax": 40 + util * 0.35,
                        "idle": idle,
                        "busy": busy,
                        "heldIdle": 0,
                    }
                },
            }
        )
    path.write_text(json.dumps({"version": 1, "minutes": minutes}), encoding="utf-8")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--live-demo", action="store_true", help="use the app's --demo mode (simulated GPU)")
    p.add_argument("--demo-gpus", type=int, default=1, help="with --live-demo: number of simulated GPUs")
    p.add_argument(
        "--demo-vendor", choices=("nvidia", "apple"), default="nvidia", help="with --live-demo: simulated GPU kind"
    )
    p.add_argument("--size", default="160x62")
    p.add_argument("--keys", default="", help="comma-separated keys to press before the capture")
    p.add_argument("--wait", type=float, default=4.0, help="seconds to let the dashboard run first")
    args = p.parse_args()
    w, h = (int(x) for x in args.size.split("x"))
    hist = Path(tempfile.mkdtemp()) / "history.json"
    probe = Monitor(MonitorConfig(history_path=hist, persist=False))
    first = probe.tick()
    uuid = first.gpus[0].uuid if first.gpus else "GPU-demo"
    probe.close()
    if args.demo:
        seed(hist, uuid)
    mon = Monitor(
        MonitorConfig(
            history_path=hist,
            persist=False,
            interval_s=0.5,
            demo=args.live_demo,
            demo_gpus=args.demo_gpus,
            demo_vendor=args.demo_vendor,
        )
    )
    app = GpuDashboard(mon)
    async with app.run_test(size=(w, h)) as pilot:
        await pilot.pause(args.wait)
        for key in filter(None, args.keys.split(",")):
            await pilot.press(key)
            await pilot.pause(0.6)
        app.save_screenshot(str(args.out))
    mon.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
