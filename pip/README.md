# gpu-pulse

A GPU dashboard that runs in your terminal. It shows live usage and how well the GPU has been used over the last 5 minutes, 30 minutes, hour and day. It works on Windows, Linux and macOS, including over plain SSH. This is the terminal companion to the **GPU Pulse** VS Code extension.

![Animated ring gauges](https://raw.githubusercontent.com/Optimizer077/gpu-pulse/main/docs/images/terminal-rings.gif)

```bash
pip install "git+https://github.com/Optimizer077/gpu-pulse.git#subdirectory=pip"   # or a wheel from Releases
gpu-pulse                  # full-screen live dashboard
gpu-pulse --demo           # preview every feature with simulated load
```

![Overview tab](https://raw.githubusercontent.com/Optimizer077/gpu-pulse/main/docs/images/terminal-overview.png)

Full documentation, screenshots and the bottleneck reference are in the [project README](https://github.com/Optimizer077/gpu-pulse#readme).

## Apple Silicon / Metal

✅ **Live GPU utilization** · ✅ **Shared GPU allocation** · ✅ **History and CSV export** ·
✅ **Job-finished alerts** · ✅ **CPU and RAM monitoring**

![Terminal dashboard on an Apple M2 Pro during a real Metal compute workload](https://raw.githubusercontent.com/Optimizer077/gpu-pulse/main/docs/images/terminal-metal.png)

Real M2 Pro readings; no simulated GPU data. The macOS backend uses built-in `ioreg`, requires no `sudo`,
and leaves unsupported GPU sensors unavailable. Apple Silicon support is available from `main` using
the source install above; it is not included in the v0.3.0 release files. See [supported hardware](#supported-hardware)
for the available metrics and shared-memory behavior.

## Commands

| Command | What it does |
|---------|--------------|
| `gpu-pulse` | Full-screen live dashboard. Keys: `1`–`5` or `←`/`→` switch tabs, `?` help, `g` next GPU, `r` timeline range (1/5/15/60 min), `e` export CSV, `q` quit |
| `gpu-pulse status` | One-shot snapshot, for SSH sessions, scripts and quick checks |
| `gpu-pulse log [-i 5]` | Records history in the background with no UI, so averages also cover time the dashboard was closed |
| `gpu-pulse export [file.csv]` | Writes per-minute history as CSV (to stdout if no file is given) |
| `gpu-pulse clear` | Deletes recorded history |
| `gpu-pulse where` | Prints the history file location |

Global options:
- `-i 0.5`: dashboard sampling interval in seconds (minimum 0.1)
- `--fps 20`: animation frame rate; `--fps 0` turns animation off, which helps on slow SSH links
- `--demo`: simulated GPU load, so you can preview the dashboard on an idle machine (nothing is saved)
- `--history PATH`: custom history file
- `--temp-warn 80`: temperature warning threshold in °C

## Layout

The dashboard has five tabs. Press `?` for a key list and a short definition of every metric.

| Tab | What it shows |
|-----|---------------|
| **1 Overview** | Six ring gauges (GPU, VRAM, temperature, power, CPU, RAM), the pulse strip, GPU timeline, **true usage**, system health, the diagnosis card, usage windows and disk/network throughput |
| **2 GPU** | Architecture card (compute, memory, PCIe link, power limits, board), live clocks/encoder/decoder/PCIe throughput, true-usage gauges, per-metric traces and GPU processes |
| **3 System** | CPU history and per-core load, RAM and swap, disk and network traces, machine specs |
| **4 History** | 5 min / 30 min / 1 h / 24 h windows, 24-hour heatmap (plus SM activity), detected sessions |
| **5 Diagnosis** | Efficiency score, bottleneck with confidence, evidence, suggested actions, throttle reasons, system health and all insights |

The layout follows the terminal size. The rings stay round and take about a third of the screen. They sit 6 in a row, 3×2 or 2×3, whichever gives the largest rings. Ring text steps down from large digits to value-only so it never overlaps the ring. When a ring has spare width, extra readings appear beside it. Below 80×24 the dashboard switches to a compact layout with only the essential panels.

## Animation

The ring gauges are drawn in braille dots. They sweep to full on startup and then settle on the live value. Between samples they ease smoothly, a light travels around each filled arc, and the tip of each arc glows. On the GPU ring, a ◆ marks the 1-hour average. The temperature ring turns amber 10 °C below `--temp-warn` and red at it. The power ring turns amber while the power cap throttles clocks. The **pulse strip** draws an ECG-style heartbeat whose beats get faster and taller as the GPU works harder. Only the Overview animates; other tabs refresh at the sample rate, and slow panels refresh every 2 s.

## What the dashboard shows

- **True usage** (NVIDIA GPUs with NVML GPM support): SM active, SM occupancy, Tensor Core, DRAM bandwidth and FP32/FP16/FP64/INT pipe activity. Utilization only says that some kernel was running. SM activity shows how much of the GPU was actually computing, and the dashboard flags when the two disagree.
- **Diagnosis**: an efficiency score for the last hour (based on SM activity when available), the likely bottleneck (input pipeline, CPU, power cap, thermal, memory, launch-bound, RAM, disk, network…) with a confidence level, evidence, suggested actions, the current session (with energy used) and a VRAM forecast.
- **System health**: one status row each for GPU, VRAM, CPU, CPU temperature, RAM, disk and network.
- **Usage over time**: for 5 min / 30 min / 1 h / 24 h, shows mean and peak utilization, the share of time idle (<10%) vs busy (≥70%), average power, peak temperature, and "held idle" (VRAM allocated while the GPU sat idle).
- **24-hour heatmap**: one cell per minute in clock-aligned hourly rows, with sessions detected from the same data.
- **Insights**: for example "Memory held while GPU idles", "CPU saturated, GPU waiting", "VRAM nearly full" or "Running hot".
- **GPU architecture**, **CPU & memory**, **disk & network**, **GPU processes** (with per-process VRAM on Linux) and **machine specs**.

## Supported hardware

| GPU | Windows | Linux | macOS |
|-----|---------|-------|-------|
| NVIDIA | NVML (ships with the driver) | NVML (ships with the driver) | — |
| AMD | not yet | `amdgpu` via `/sys/class/drm` (best effort) | — |
| Apple Silicon / Metal | — | — | ✅ Utilization and allocated GPU memory via `ioreg` |
| none | CPU / RAM / specs only | CPU / RAM / specs only | CPU / RAM / specs only |

Apple Silicon support requires no `sudo`, helper installation or extra Python dependency. IORegistry counters
are best effort and can vary by macOS version. Allocated GPU memory is **shared system memory**, so there is
no dedicated VRAM total or percentage. Unsupported metrics (GPU temperature, power, clocks, GPM and per-process
GPU usage) remain unavailable. Memory-capacity alerts need a known total and do not fire for this backend;
utilization history and job-finished alerts work normally. Intel and AMD GPUs on macOS are not supported.

On Windows, per-process VRAM shows as *n/a*. The Windows display driver model (WDDM) does not report it.

## Recording history all the time

Averages only cover time when `gpu-pulse` (or `gpu-pulse log`) was running. To record continuously on a Linux server, use a systemd user service:

```ini
# ~/.config/systemd/user/gpu-pulse.service
[Unit]
Description=GPU Pulse usage logger

[Service]
ExecStart=%h/.local/bin/gpu-pulse log -i 5
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now gpu-pulse
```

The dashboard and the logger can run at the same time. Saves merge per minute and are written atomically, so neither one overwrites the other.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests && uv run mypy src
uv run python scripts/screenshot.py out.svg --live-demo --size 130x40 --keys 2   # headless screenshot (GPU tab)
uv build
```
