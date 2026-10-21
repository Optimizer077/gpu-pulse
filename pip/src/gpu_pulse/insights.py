"""Plain-language observations about recent usage (same rules as the VS Code extension)."""

from __future__ import annotations

from typing import Optional

from .history import HELD_VRAM, IDLE_UTIL
from .models import GpuSample, Insight, WindowStats


def grade(util: Optional[float]) -> Optional[tuple[str, str]]:
    """(label, level) verdict from mean utilization; level in idle/low/good/max."""
    if util is None:
        return None
    if util < 20:
        return "Mostly idle", "idle"
    if util < 50:
        return "Underused", "low"
    if util < 85:
        return "Well used", "good"
    return "Saturated", "max"


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def build_insights(hour: WindowStats, half: WindowStats, gpu: Optional[GpuSample], temp_warn: float) -> list[Insight]:
    out: list[Insight] = []
    if gpu is None:
        out.append(
            Insight(
                "info",
                "No supported GPU detected",
                "NVML is unavailable and no amdgpu device exposes utilization. CPU and memory are still tracked.",
            )
        )
    if hour.covered_minutes < 5:
        out.append(
            Insight(
                "info",
                "Building history",
                f"{hour.covered_minutes} min recorded so far. Averages become meaningful after a few minutes.",
            )
        )
    if gpu is not None:
        v = gpu.vram_pct
        if v is not None and v >= 95:
            out.append(
                Insight(
                    "serious",
                    "VRAM nearly full",
                    f"{v:.0f}% of video memory is allocated. New allocations may fail with out-of-memory errors.",
                )
            )
        if gpu.temp_c is not None and gpu.temp_c >= temp_warn:
            out.append(
                Insight(
                    "critical",
                    f"Running hot: {gpu.temp_c:.0f} °C",
                    f"At or above the {temp_warn:.0f} °C threshold. Sustained heat can lower clocks.",
                )
            )
        elif hour.temp_peak is not None and hour.temp_peak >= temp_warn:
            out.append(
                Insight(
                    "warning",
                    f"Peaked at {hour.temp_peak:.0f} °C this hour",
                    f"Above the {temp_warn:.0f} °C threshold at least once in the last hour.",
                )
            )

    if hour.covered_minutes >= 5 and hour.util is not None:
        if hour.held_idle_share is not None and hour.held_idle_share >= 0.25:
            out.append(
                Insight(
                    "warning",
                    "Memory held while GPU idles",
                    f"For {_pct(hour.held_idle_share)} of the last hour, ≥{HELD_VRAM:.0f}% of VRAM was allocated "
                    f"while utilization stayed under {IDLE_UTIL:.0f}%. An idle process may be holding memory.",
                )
            )
        if hour.util < 20:
            out.append(
                Insight(
                    "info",
                    "GPU mostly idle",
                    f"Averaged {hour.util:.0f}% over {hour.covered_minutes} recorded minutes; "
                    f"idle {_pct(hour.idle_share or 0)} of the time.",
                )
            )
        elif hour.busy_share is not None and hour.busy_share >= 0.6:
            out.append(
                Insight(
                    "good",
                    "GPU kept busy",
                    f"Above 70% utilization for {_pct(hour.busy_share)} of the last hour (mean {hour.util:.0f}%).",
                )
            )
        elif hour.util < 50:
            out.append(
                Insight(
                    "info",
                    "Utilization is intermittent",
                    f"Mean {hour.util:.0f}% with frequent idle gaps. In training loops this often points at "
                    "data loading or CPU-side work between steps.",
                )
            )
        if half.cpu is not None and half.cpu >= 85 and half.util is not None and half.util < 50:
            out.append(
                Insight(
                    "warning",
                    "CPU saturated, GPU waiting",
                    f"Last 30 min: CPU {half.cpu:.0f}% vs GPU {half.util:.0f}%. The workload may be CPU-bound.",
                )
            )
        if gpu and gpu.power_limit_w and hour.power and hour.util >= 70 and hour.power / gpu.power_limit_w < 0.5:
            out.append(
                Insight(
                    "info",
                    "High utilization, low power draw",
                    f"Utilization counts time any kernel runs, not how many cores are busy. Drawing "
                    f"{hour.power:.0f} W of {gpu.power_limit_w:.0f} W suggests small or memory-bound kernels.",
                )
            )
    return out
