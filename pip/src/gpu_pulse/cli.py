"""Command-line entry point: ``gpu-pulse [dashboard|status|log|export|clear|where]``."""

from __future__ import annotations

import argparse
import logging
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import __version__
from .alerts import Alert, AlertConfig, AlertEngine, format_line
from .models import Sample
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


def _minutes(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected minutes, got {text!r}") from None
    if not 0 < v <= 24 * 60:
        raise argparse.ArgumentTypeError("must be more than 0 and at most 1440 minutes")
    return v


def _alert_rule_options(p: argparse.ArgumentParser, inherit: bool = False) -> None:
    """Alert thresholds; on a subcommand (``inherit``) they only override when given."""
    kw: dict[str, Any] = {"default": argparse.SUPPRESS} if inherit else {}
    p.add_argument(
        "--idle-held-min",
        type=_minutes,
        metavar="MIN",
        help="alert when a GPU idles while holding VRAM this long (default 15)",
        **({"default": 15.0} | kw),
    )
    p.add_argument(
        "--job-min",
        type=_minutes,
        metavar="MIN",
        help="alert when a busy stretch at least this long finishes (default 10)",
        **({"default": 10.0} | kw),
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpu-pulse", description="Terminal GPU dashboard with usage history.")
    p.add_argument("--version", action="version", version=f"gpu-pulse {__version__}")
    p.add_argument("--history", type=Path, default=None, help=f"history file (default: {default_history_path()})")
    p.add_argument("--temp-warn", type=float, default=80.0, help="temperature warning threshold in °C")
    p.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.5,
        help="dashboard: seconds between samples (default 0.5; at least 1 on Apple GPUs, where each read runs ioreg)",
    )
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
    p.add_argument(
        "--demo-vendor",
        choices=("nvidia", "apple"),
        default="nvidia",
        help="with --demo: simulate an NVIDIA GPU (default) or an Apple Silicon GPU (no temperature/power)",
    )
    p.add_argument(
        "--alerts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="dashboard: alert toasts for hot GPUs, full VRAM, idle-but-held VRAM and finished jobs (default on)",
    )
    p.add_argument("--bell", action="store_true", help="dashboard: also ring the terminal bell once per alert")
    _alert_rule_options(p)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("dashboard", help="full-screen live dashboard (default)")
    sub.add_parser("status", help="print a one-shot snapshot and exit")
    lg = sub.add_parser("log", help="record usage history in the background (no UI)")
    lg.add_argument(
        "-i", "--interval", dest="log_interval", type=float, default=5.0, help="seconds between samples (default 5)"
    )
    lg.add_argument(
        "--alerts", action="store_true", default=argparse.SUPPRESS, help="print one timestamped line per alert"
    )
    lg.add_argument(
        "--notify",
        action="store_true",
        help="with alerts: also send a desktop notification (Linux, needs notify-send; implies --alerts)",
    )
    _alert_rule_options(lg, inherit=True)
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
            demo_vendor=args.demo_vendor if demo else "nvidia",
        )
    )


def _alert_config(args: argparse.Namespace, interval_s: float) -> AlertConfig:
    return AlertConfig(
        temp_warn=args.temp_warn, interval_s=interval_s, idle_held_min=args.idle_held_min, job_min=args.job_min
    )


def desktop_notifier() -> Optional[Callable[[Alert], None]]:
    """A no-dependency desktop notification sender, or None when this platform has none (Linux: notify-send)."""
    if not sys.platform.startswith("linux"):
        return None
    exe = shutil.which("notify-send")
    if exe is None:
        return None

    def send(a: Alert) -> None:
        urgency = "normal" if a.level == "warning" else "low"
        try:  # fire and forget: a slow notification daemon must not delay sampling
            subprocess.Popen(
                [exe, "-a", "GPU Pulse", "-u", urgency, a.title, a.detail],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            logger.warning("notify-send failed: %s", e)

    return send


def alert_printer(args: argparse.Namespace, interval_s: float) -> Optional[Callable[[Sample], None]]:
    """``log --alerts / --notify``: a per-sample callback printing one stdout line per alert, or None."""
    notify = bool(getattr(args, "notify", False))
    if not (args.alerts or notify):
        return None
    engine = AlertEngine(_alert_config(args, interval_s))
    sender = desktop_notifier() if notify else None
    if notify and sender is None:
        logger.info("Desktop notifications need notify-send on Linux; alerts are printed only.")

    def on_sample(s: Sample) -> None:
        for a in engine.update(s):
            sys.stdout.write(format_line(a) + "\n")
            sys.stdout.flush()
            if sender is not None:
                sender(a)

    return on_sample


def _log_forever(mon: Monitor, on_sample: Optional[Callable[[Sample], None]] = None) -> None:
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
            s = mon.tick()
            if on_sample is not None:
                try:
                    on_sample(s)
                except Exception as e:  # noqa: BLE001 - an alert bug must not stop recording
                    logger.warning("Alert check failed: %s", e)
            stop.wait(max(0.0, mon.cfg.interval_s - (time.monotonic() - t0)))
    finally:
        mon.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.demo_gpus > 1 and not args.demo:
        parser.error("--demo-gpus only applies with --demo (real GPUs are always all shown)")
    if args.demo_vendor != "nvidia" and not args.demo:
        parser.error("--demo-vendor only applies with --demo (real GPUs are detected)")
    if args.demo_vendor == "apple" and args.demo_gpus > 1:
        parser.error("--demo-vendor apple simulates one GPU (Apple Silicon Macs have one); drop --demo-gpus")
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
        mon = _monitor(args, args.log_interval)
        _log_forever(mon, alert_printer(args, mon.cfg.interval_s))
        return 0

    from .app import GpuDashboard

    mon = _monitor(args, args.interval, demo=args.demo)
    try:
        GpuDashboard(
            mon,
            fps=args.fps,
            alerts=args.alerts is not False,
            alert_config=_alert_config(args, mon.cfg.interval_s),
            bell=args.bell,
        ).run()
    finally:
        mon.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
