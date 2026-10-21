# GPU Pulse

A live GPU, CPU and memory dashboard inside VS Code. It also shows how well the GPU has been used over the last 5 minutes, 30 minutes, hour and day.

Works on **Windows** and **Linux**, including Linux GPU servers you reach through **Remote-SSH** or WSL. The extension runs on the machine that owns the GPU.

> **Prefer the terminal?** The same project includes **`gpu-pulse`**, a terminal dashboard in the `pip/` folder of this repository. It has animated ring gauges, true GPU usage (SM, Tensor Core and DRAM activity), bottleneck diagnosis and system health. Install it with `pip install ./pip` or `uv tool install ./pip`.

## Features

- **Live gauges**: GPU utilization (with a marker for the 1-hour average), VRAM, temperature, and power against the power cap. Also shows clocks, memory-bus load, fan, P-state and PCIe link.
- **GPU timeline**: GPU utilization and VRAM over the last 1, 5, 15 or 60 minutes, with a hover crosshair.
- **Usage over time**: for 5 min / 30 min / 1 h / 24 h, shows mean and peak utilization, the share of time the GPU was idle (<10%), active or busy (≥70%), average power, peak temperature, and how often VRAM was held while the GPU sat idle.
- **24-hour activity heatmap**: one cell per minute, one row per clock hour.
- **Insights**: plain-language notes, such as "Memory held while GPU idles", "CPU saturated, GPU waiting", "VRAM nearly full" or "Running hot".
- **CPU & memory**: CPU/RAM timeline, per-core load bars, RAM usage.
- **GPU processes**: compute processes and their VRAM use (NVIDIA).
- **Machine specs**: CPU, GPUs, driver, CUDA version, memory modules, disks, OS, board and uptime.
- **Status bar**: `GPU 37% · 5.1/16 GB · 54°C`. Click it to open the dashboard; it turns amber above your temperature threshold.
- **History** is saved per minute and survives VS Code restarts. Export it with **GPU Pulse: Export Usage History (CSV)**.

## Usage

Run **GPU Pulse: Open Dashboard** from the Command Palette, or click the status bar item.

## Requirements

| GPU | Windows | Linux |
|-----|---------|-------|
| NVIDIA | `nvidia-smi` (ships with the driver) | `nvidia-smi` (ships with the driver) |
| AMD | specs only | `amdgpu` driver via `/sys/class/drm` (best effort) |
| Other / none | CPU, RAM and specs only | CPU, RAM and specs only |

On Windows, per-process VRAM shows as *n/a*. The Windows display driver model (WDDM) does not report it to `nvidia-smi`.

## Settings

| Setting | Default | Meaning |
|---------|---------|---------|
| `gpuPulse.sampleIntervalMs` | `1000` | Sampling interval while the dashboard is visible |
| `gpuPulse.backgroundIntervalMs` | `5000` | Sampling interval while hidden (history keeps recording) |
| `gpuPulse.historyDays` | `7` | Days of per-minute history to keep |
| `gpuPulse.statusBar.enabled` | `true` | Show the status bar item |
| `gpuPulse.nvidiaSmiPath` | `""` | Custom `nvidia-smi` path |
| `gpuPulse.temperatureWarning` | `80` | Warning threshold in °C |

## How the numbers are measured

- **Utilization** comes from `nvidia-smi`'s `utilization.gpu`: the share of time at least one kernel was running. It does not measure how many cores were busy, so a GPU can read 100% while it draws little power.
- Averages are weighted by sample count. A window only covers the time VS Code was running; each card shows "x / y min recorded".
- History is stored as JSON in the extension's global storage on the GPU machine. **GPU Pulse: Clear Usage History** deletes it.

## Development

```bash
npm install
npm run build        # bundle to dist/
npm test             # unit tests (vitest)
npm run smoke        # run the collectors against this machine's hardware
npm run package      # produce gpu-pulse-<version>.vsix
```

Open `scripts/preview.html` in a browser to see the dashboard with synthetic data (add `?light` for the light theme).

## License

MIT. See the `LICENSE` file. The bundled fonts (Chakra Petch and Azeret Mono) are under the SIL Open Font License 1.1; see `media/fonts/`.
