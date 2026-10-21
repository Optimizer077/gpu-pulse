"""End-to-end CLI checks on the real machine (GPU optional)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from gpu_pulse.cli import main


def test_log_records_and_saves_on_signal(tmp_path: Path) -> None:
    hist = tmp_path / "history.json"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    p = subprocess.Popen(
        [sys.executable, "-m", "gpu_pulse", "--history", str(hist), "log", "-i", "0.25"],
        creationflags=flags,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)
    p.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    out, _ = p.communicate(timeout=20)
    assert p.returncode == 0, out.decode(errors="replace")
    data = json.loads(hist.read_text())
    assert data["version"] == 1
    assert sum(m["n"] for m in data["minutes"]) >= 5


def test_export_and_clear(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    hist = tmp_path / "h.json"
    t = int(time.time() // 60 * 60_000)
    hist.write_text(json.dumps({"version": 1, "minutes": [{"t": t, "n": 3, "cpu": 5, "mem": 6, "gpus": {}}]}))
    assert main(["--history", str(hist), "export"]) == 0
    assert ",5.0,6.0" in capsys.readouterr().out
    assert main(["--history", str(hist), "clear"]) == 0
    data = json.loads(hist.read_text())
    assert data.get("clearedAt") and all(m["t"] >= data["clearedAt"] for m in data["minutes"])
