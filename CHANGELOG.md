# Changelog

CPU figures are measured on one machine (RTX 5060 Ti, Core Ultra 7 265F, Windows 11) and expressed as a share of one CPU core. Your numbers will differ.

## 0.4.0 (beta: macOS support, not yet verified on Mac hardware)

### macOS / Apple GPUs (Metal), both tools
- **Apple Silicon (M-series):** GPU utilization (device, renderer, tiler) and GPU memory in use (unified memory, against total RAM), read from `ioreg` (IOAccelerator). No admin rights needed.
- **Specs:** GPU name, GPU core count and Metal version from `system_profiler`.
- **Intel Macs:** the AMD/Intel GPU through the same `ioreg` interface (VRAM used/total where reported).
- **Shown as n/a on macOS:** temperature, power, clocks, fan, PCIe, the process list, and NVIDIA-only true-usage (SM/Tensor) counters. Heat, power and PCIe rules never fire on Apple GPUs.
- **VS Code:** "GPU memory (unified)" wording, and "16-core GPU · Metal 3" in the dashboard and Machine card.
- **Terminal:**
  - a GPU LIVE panel (device / renderer / tiler)
  - `--demo --demo-vendor apple`
  - Apple GPUs sampled at most every 1 s until the real `ioreg` cost is measured
- **Diagnostics for testing on a Mac:** `scripts/mac-probe.sh` (extension) and `pip/scripts/mac_probe.py` (terminal). Both are read-only and drop monitor serial numbers.
- **Caveat:** the `ioreg` key names come from documentation and memory, and tests run against reconstructed sample output. This beta exists to verify them on a real Mac.

### Fixes
- Terminal test suite: widget unit tests now pass on Python 3.9 too. Only the tests were affected, not the app.

## 0.3.0

### Alerts (both tools, same rules)
- **Running hot:** temperature ≥ your warning threshold for 30 s. Clears 3 °C below it.
- **VRAM nearly full:** ≥ 95% for 20 s. Clears below 90%.
- **VRAM held while idle:** utilization < 10% with VRAM ≥ 50% for 15 min (configurable).
- **Job finished:** a busy stretch of ≥ 10 min (configurable) ends. Reports the duration and average utilization.
- Each alert fires once, re-arms only after its condition clears, and repeats at most every 30 min per GPU. Gaps in the data reset the timers, so stale samples never trigger alerts.

### VS Code extension
- Non-modal notifications with **Open Dashboard** and **Mute 1 hour**.
- New commands: `GPU Pulse: Mute Alerts` / `Unmute Alerts`.
- An alert badge in the dashboard and side bar; active alerts listed in the status-bar tooltip.
- Settings: `gpuPulse.alerts.enabled`, `.hot`, `.vramFull`, `.idleHeld`, `.jobFinished`, `.idleHeldMinutes`, `.jobFinishedMinutes`.
- Alerts work with the dashboard closed, as long as sampling runs.
- The alert engine costs ~0.1 µs per sample for one GPU.

### Terminal dashboard
- **Alert toasts** in the dashboard, an alert list on the Diagnosis tab and a count in the header. `m` mutes alerts for 1 hour.
- **Options:** `--alerts/--no-alerts`, `--bell`, `--idle-held-min`, `--job-min`.
- **No UI needed:** `gpu-pulse log --alerts` prints one line per alert. `--notify` also sends a desktop notification on Linux via `notify-send`.
- **Resting pulse:** while the GPU is idle, the pulse strip scrolls slowly (4 columns/s instead of 36) and speeds up with load. On an idle GPU, pulse repaints over 15 s went from 181 to 60.

### Project
- Tests: extension 67 → 102, terminal 193 → 227.

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
