# Changelog

CPU figures are measured on one machine (RTX 5060 Ti, Core Ultra 7 265F, Windows 11) and expressed as a share of one CPU core. Your numbers will differ.

## 0.2.0

### VS Code extension
- **Pinned side-bar view:** GPU Pulse now has its own Activity Bar icon with a compact live monitor: per-GPU rows, a 5-minute sparkline, CPU and RAM bars, and the top insights. You can drag it to the Secondary Side Bar to keep it visible while you work. New command: **GPU Pulse: Show in Side Bar**.
- **Faster and cheaper sampling:** one long-lived `nvidia-smi … -lms` process replaces one process per sample.
  - The default refresh is now **500 ms** (was 1 s), and it can go as low as 250 ms.
  - Total CPU at 500 ms dropped from ~5.7% to **~0.2–0.6%** of a core.
  - The stream restarts automatically. It falls back to per-sample polling if it keeps failing, and retries streaming every 10 minutes.
- **Multiple GPUs:**
  - An **All GPUs** strip of selectable cards.
  - An aggregate status bar (`GPU ×4 avg 67% · max 98% · hottest 71°C`) with a per-GPU tooltip.
  - Per-GPU history.
  - Cross-GPU insights: a GPU well below the others (possible straggler), uneven VRAM, all GPUs busy, a hot GPU that isn't selected.
- **Lighter dashboard:**
  - Window stats and insights are computed at most every 2 s, and tick messages are compact.
  - The dashboard only touches the page where something changed, and redraws charts only when their data changed. On an idle machine it does 3–5× less work.
  - Nothing is sent while both views are hidden.
- **Safety:** each webview may only run its own commands. Synthetic test GPUs are disabled in installed builds.

### Terminal dashboard (`gpu-pulse`)
- **CPU overhaul:** the Overview went from **~57% to ~5%** of a core with an idle GPU (3.5–6.7% across runs), and to about 5–10% with four simulated GPUs all moving. Other tabs use ~0.5–1%.
  - Rings and the pulse strip render directly to terminal lines with cached styles, and only changed rows are repainted (ring row repaints per 15 s: 990 → 117).
  - Rings stop drawing once they reach their value. A ring only shimmers after a visible move, at most every 9 s. Animation pauses off the Overview tab.
  - The default frame rate is now 12 fps (was 20).
- **Multiple GPUs:**
  - An **ALL GPUS** panel: one line per GPU with health, a utilization bar with SM activity, VRAM, temperature, power, throttle badges and tags. Select a GPU with `g`/`G`, by clicking, or with `g` then a digit.
  - Cross-GPU findings: straggler, uneven VRAM, one GPU throttling alone, mismatched PCIe link.
    - Only GPUs that are actually in use (≥10% VRAM or a compute process; same process name when known) are compared, each needs at least 30 s of data, and the wording stays conditional ("if they run one job").
    - An idle GPU next to a busy one is reported as idle, not as a straggler.
    - A card that is x8 by design is reported as a card limit, not a slot problem.
  - The selected GPU is remembered by UUID, so the view stays on the same physical GPU if another drops out.
  - `gpu-pulse status` prints a totals line.
  - `--demo --demo-gpus N` simulates N GPUs (including a straggler) for previews and testing.
- **Lower sampling cost:** fan and energy counters are read every 2 s, and the PCIe-throughput poller is duty-capped. One GPU's sample went from 0.44 to 0.10 ms.
- **Diagnosis fix:** right after start, minute averages can no longer report "idle" while the live reading shows the GPU working.
- **Python 3.9 is verified:** the full test suite passes on 3.9 and 3.12.

### Project
- **Tests:** 67 for the extension (was 21) and 193 for the terminal tool (was 140). The terminal suite also passes on Python 3.9.

## 0.1.0

- **First release:**
  - live gauges and a GPU timeline
  - 5 min / 30 min / 1 h / 24 h usage windows and a 24-hour heatmap
  - insights, CPU & memory, GPU processes and machine specs
  - status bar
  - persistent history with CSV export
- **Platforms:** NVIDIA on Windows and Linux, AMD on Linux (sysfs, best effort), Remote-SSH support.
- **Terminal dashboard:**
  - true-usage (GPM) metrics
  - bottleneck diagnosis with an efficiency score
  - system health and GPU architecture details
