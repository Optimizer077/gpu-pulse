"""Modal screens: the ``?`` help overlay (keys and what every metric means)."""

from __future__ import annotations

from rich.console import Group
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from . import render as R

KEYS = (
    ("1 – 5", "Overview · GPU · System · History · Diagnosis"),
    ("← →  [ ]", "previous / next tab"),
    ("g / G", "next / previous GPU (multi-GPU machines); click a line in ALL GPUS to pick one"),
    ("g then 0 – 9", "jump to the GPU with that index (as nvidia-smi numbers them); g 1 2 → GPU 12"),
    ("r", "timeline range: 1 / 5 / 15 / 60 min"),
    ("e", "export per-minute history as CSV"),
    ("m", "mute alerts for 1 hour / unmute (alerts are listed on the Diagnosis tab)"),
    ("?", "this help · Esc closes"),
    ("q", "quit"),
)

METRICS = (
    ("Utilization", "Share of time at least one kernel was running. 100% can still mean one tiny kernel."),
    ("SM active", "Share of streaming multiprocessors doing work (NVML GPM). The honest 'how busy' number."),
    ("SM occupancy", "Resident warps vs the maximum. Low occupancy limits latency hiding."),
    ("Tensor cores", "Activity of the matrix units used by mixed-precision training and inference."),
    ("DRAM bandwidth", "Memory bandwidth used vs peak. High here + low SM = memory-bound kernels."),
    ("FP32/FP16/FP64/INT", "Activity of each arithmetic pipe; shows which precision is doing the work."),
    ("VRAM", "Video memory allocated. 'Held idle' = VRAM allocated while utilization < 10%."),
    ("Idle / busy", "Share of samples under 10% / at or above 70% utilization in a window."),
    ("Efficiency", "0–100 score for the last hour from utilization, SM activity and throttling."),
    ("Bottleneck", "What is holding the GPU back: input pipeline, CPU, power cap, thermal, memory…"),
    ("Throttle", "Clock limits reported by the driver right now, e.g. POWER CAP or THERMAL."),
    ("◆ marker", "On the GPU ring: the 1-hour average utilization."),
    ("Pulse", "Heartbeat whose rate and height follow GPU utilization."),
    ("All GPUs", "One line per GPU: bright bar = SM activity, dim = kernel running but SMs idle. Cross-GPU "
     "checks flag stragglers, uneven VRAM, a GPU throttling alone and mismatched PCIe links."),
    ("Alerts", "Toasts when a GPU runs hot (30 s), VRAM is nearly full (20 s), a GPU idles while holding VRAM, "
     "or a long job finishes. Each fires once, re-arms after clearing, and waits 30 min before repeating."),
    ("Rings", "Temperature turns amber 10 °C below the warning threshold and red at it; power turns amber "
     "while the power cap throttles clocks."),
)  # fmt: skip


def help_text(shared_memory: bool = False) -> Group:
    keys = Table.grid(padding=(0, 2))
    keys.add_column(style=Style(color=R.GPU_GLOW, bold=True), no_wrap=True)
    keys.add_column(style=Style(color=R.INK))
    for k, v in KEYS:
        keys.add_row(k, v)
    metrics = Table.grid(padding=(0, 2))
    metrics.add_column(style=Style(color=R.INK, bold=True), no_wrap=True)
    metrics.add_column(style=Style(color=R.MUTED))
    for k, v in METRICS:
        if shared_memory and k in ("SM active", "SM occupancy", "Tensor cores", "DRAM bandwidth",
                                   "FP32/FP16/FP64/INT", "VRAM", "Throttle", "Rings", "All GPUs", "Alerts"):
            continue
        metrics.add_row(k, v)
    if shared_memory:
        metrics.add_row("GPU memory", "GPU allocations in GB, shared with the CPU. "
                        "There is no dedicated VRAM capacity.")
        metrics.add_row("Swap", "System swap usage, shown with RAM; rising swap can affect GPU workloads.")
        metrics.add_row("Alerts", "Notifies when a sustained GPU workload finishes; m mutes notifications.")
    return Group(
        Text("KEYS", Style(color=R.MUTED, bold=True)),
        keys,
        Text(),
        Text("WHAT THE NUMBERS MEAN", Style(color=R.MUTED, bold=True)),
        metrics,
    )


class HelpScreen(ModalScreen[None]):
    def __init__(self, shared_memory: bool = False) -> None:
        super().__init__()
        self.shared_memory = shared_memory

    DEFAULT_CSS = """
    HelpScreen { align: center middle; background: #080a0e 70%; }
    #help-box { width: 96; max-width: 96%; height: auto; max-height: 90%; border: round #1e9ccc;
                border-title-color: #5fd4ff; border-subtitle-color: #7d8696; background: #0d1117; padding: 1 2; }
    """
    BINDINGS = [("escape", "close", "Close"), ("question_mark", "close", "Close"), ("q", "close", "Close")]

    def compose(self) -> ComposeResult:
        box = VerticalScroll(Static(help_text(self.shared_memory)), id="help-box")
        box.border_title = "GPU PULSE · HELP"
        box.border_subtitle = "Esc to close"
        yield box

    def action_close(self) -> None:
        self.dismiss()
