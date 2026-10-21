"""Command-line entry point: ``gpu-pulse [dashboard|status|log|export|clear|where]``."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import __version__
from .monitor import Monitor, MonitorConfig, default_history_path

logger = logging.getLogger(__name__)

DEFAULT_FPS = 12.0  # smooth enough for eased gauges and the pulse; ~40% cheaper than 20 fps
MAX_DEMO_GPUS = 16


def _gpu_count(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if not 1 <= n <= MAX_DEMO_GPUS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_DEMO_GPUS}")
    return n


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpu-pulse", description="Terminal GPU dashboard with usage history.")
    p.add_argument("--version", action="version", version=f"gpu-pulse {__version__}")
    p.add_argument("--history", type=Path, default=None, help=f"history file (default: {default_history_path()})")
    p.add_argument("--temp-warn", type=float, default=80.0, help="temperature warning threshold in °C")
    p.add_argument("-i", "--interval", type=float, default=0.5, help="dashboard: seconds between samples (default 0.5)")
    p.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help="dashboard: animation frames per second, 0 = off (default 12)"
    )
    p.add_argument(
        "--demo", action="store_true", help="dashboard/status: simulated GPU load to preview the UI (nothing is saved)"
    )
    p.add_argument(
        "--demo-gpus",
        type=_gpu_count,
        default=1,
        metavar="N",
        help="with --demo: number of simulated GPUs, 1-16 (default 1); GPU 2 plays a straggler",
    )
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("dashboard", help="full-screen live dashboard (default)")
    sub.add_parser("status", help="print a one-shot snapshot and exit")
    lg = sub.add_parser("log", help="record usage history in the background (no UI)")
    lg.add_argument(
        "-i", "--interval", dest="log_interval", type=float, default=5.0, help="seconds between samples (default 5)"
    )
    ex = sub.add_parser("export", help="write per-minute history as CSV")
    ex.add_argument("output", nargs="?", type=Path, help="CSV path (default: stdout)")
    sub.add_parser("clear", help="delete recorded history")
    sub.add_parser("where", help="print the history file location")
    return p


def _monitor(args: argparse.Namespace, interval: float = 1.0, persist: bool = True, demo: bool = False) -> Monitor:
    interval = min(60.0, max(0.1, interval))
    return Monitor(
        MonitorConfig(
            interval_s=interval,
            temp_warn=args.temp_warn,
            history_path=args.history,
            persist=persist and not demo,
            demo=demo,
            demo_gpus=args.demo_gpus if demo else 1,
        )
    )


def _log_forever(mon: Monitor) -> None:
    # An Event (not time.sleep) so a stop signal interrupts the wait at once and the final
    # save completes inside short shutdown grace periods (docker stop, console close).
    stop = threading.Event()

    def _stop(*_: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGBREAK"):  # Windows: Ctrl+Break / console close from a service wrapper
        signal.signal(signal.SIGBREAK, _stop)
    logger.info("Recording every %.1fs to %s (Ctrl+C to stop)", mon.cfg.interval_s, mon.file.path)
    try:
        while not stop.is_set():
            t0 = time.monotonic()
            mon.tick()
            stop.wait(max(0.0, mon.cfg.interval_s - (time.monotonic() - t0)))
    finally:
        mon.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.demo_gpus > 1 and not args.demo:
        parser.error("--demo-gpus only applies with --demo (real GPUs are always all shown)")
    cmd = args.cmd or "dashboard"
    if cmd == "dashboard":
        # stderr output would draw over the full-screen UI; route it to Textual's log instead.
        from textual.logging import TextualHandler

        logging.basicConfig(level=logging.WARNING, handlers=[TextualHandler()])
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if cmd == "where":
        print(args.history or default_history_path())
        return 0
    if cmd == "status":
        from .status import run_status

        mon = _monitor(args, persist=False, demo=args.demo)
        try:
            run_status(mon)
        finally:
            mon.nvidia.close()
        return 0
    if cmd == "export":
        mon = _monitor(args, persist=False)
        csv = mon.history.to_csv()
        if args.output:
            args.output.write_text(csv, encoding="utf-8")
            logger.info("Wrote %s", args.output)
        else:
            sys.stdout.write(csv)
        return 0
    if cmd == "clear":
        mon = _monitor(args, persist=False)  # no final save on close; we save explicitly below
        mon.history.clear()
        ok = mon.save()
        mon.close()
        if not ok:
            logger.error("Could not clear history in %s (see message above)", mon.file.path)
            return 1
        logger.info("Cleared history in %s", mon.file.path)
        return 0
    if cmd == "log":
        _log_forever(_monitor(args, args.log_interval))
        return 0

    from .app import GpuDashboard

    mon = _monitor(args, args.interval, demo=args.demo)
    try:
        GpuDashboard(mon, fps=args.fps).run()
    finally:
        mon.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
