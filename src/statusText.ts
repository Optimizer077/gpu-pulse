import type { GpuSample, Sample } from './types';

/** Status bar content as plain strings (no vscode dependency, so it is unit-testable). */
export interface StatusView {
  text: string;
  /** Markdown with $(icon) syntax. */
  tooltip: string;
  warn: boolean;
}

/** Alert state shown in the tooltip (the status bar text never changes for alerts). */
export interface AlertSummary {
  /** Titles of alerts whose condition still holds. */
  active: string[];
  /** Epoch ms until which notifications are muted, or 0. */
  mutedUntil: number;
}

const hhmm = (t: number) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

/** Tooltip lines for active alerts and the mute state; empty when there is nothing to say. */
export function alertLines(a: AlertSummary | undefined, now = Date.now()): string {
  if (!a) return '';
  let out = '';
  if (a.active.length) {
    const n = a.active.length;
    out += `$(bell) **${n} active alert${n > 1 ? 's' : ''}**\n\n` + a.active.map((t) => `- ${t}\n`).join('') + '\n';
  }
  if (a.mutedUntil > now) out += `$(bell-slash) Alerts muted until ${hhmm(a.mutedUntil)}\n\n`;
  return out;
}

const dash = (v: number | null | undefined, d = 0) => (v === null || v === undefined ? '–' : v.toFixed(d));
const shortName = (n: string) => n.replace(/^NVIDIA (GeForce )?/, '');

/**
 * Used/total for VRAM. Unified memory (Apple Silicon) shows only what the GPU holds: the
 * total is all of system RAM, which reads like free VRAM that is not really the GPU's.
 */
function vramGb(g: GpuSample): string {
  if (g.unifiedMemory) return g.memUsedMiB !== null ? `${(g.memUsedMiB / 1024).toFixed(1)} GB` : '–';
  return g.memUsedMiB !== null && g.memTotalMiB ? `${(g.memUsedMiB / 1024).toFixed(1)}/${(g.memTotalMiB / 1024).toFixed(0)} GB` : '–';
}

/** Tooltip memory phrase: "12.0/24 GB" (VRAM) or "GPU mem 2.1 GB of 32 GB unified". */
function memPhrase(g: GpuSample): string {
  if (!g.unifiedMemory) return vramGb(g);
  return `GPU mem ${vramGb(g)}${g.memTotalMiB ? ` of ${Math.round(g.memTotalMiB / 1024)} GB unified` : ''}`;
}

function cpuLine(s: Sample): string {
  const ram = s.mem.totalBytes ? Math.round((s.mem.usedBytes / s.mem.totalBytes) * 100) : 0;
  return `$(dashboard) CPU ${Math.round(s.cpu.util)}% · RAM ${ram}%\n\n`;
}

export function statusView(s: Sample, tempWarn: number, alerts?: AlertSummary): StatusView {
  const gs = s.gpus;
  if (!gs.length) {
    return { text: `$(pulse) CPU ${Math.round(s.cpu.util)}%`, tooltip: 'GPU Pulse — no GPU detected. Click to open the dashboard.', warn: false };
  }
  const temps = gs.map((g) => g.tempC).filter((t): t is number => t !== null);
  const hottest = temps.length ? Math.max(...temps) : null;
  const warn = hottest !== null && hottest >= tempWarn;

  if (gs.length === 1) {
    const g = gs[0];
    const vram = g.unifiedMemory
      ? g.memUsedMiB !== null ? ` · ${vramGb(g)}` : ''
      : g.memUsedMiB !== null && g.memTotalMiB ? ` · ${(g.memUsedMiB / 1024).toFixed(1)}/${Math.round(g.memTotalMiB / 1024)} GB` : '';
    const temp = g.tempC !== null ? ` · ${Math.round(g.tempC)}°C` : '';
    // Readings a backend never reports (macOS: temperature, power) are left out, not shown as "–".
    const parts = [`${dash(g.util)}% util`, memPhrase(g)];
    if (g.tempC !== null || g.vendor !== 'apple') parts.push(`${dash(g.tempC)} °C`);
    if (g.powerW !== null || g.vendor !== 'apple') parts.push(`${dash(g.powerW)} W`);
    const tooltip =
      '**GPU Pulse**\n\n' +
      `$(circuit-board) ${g.name} — ${parts.join(', ')}\n\n` +
      cpuLine(s) +
      alertLines(alerts, s.t) +
      '_Click to open the dashboard_';
    return { text: `$(pulse) GPU ${dash(g.util)}%${vram}${temp}`, tooltip, warn };
  }

  const utils = gs.map((g) => g.util).filter((u): u is number => u !== null);
  const avg = utils.length ? utils.reduce((a, b) => a + b, 0) / utils.length : null;
  const max = utils.length ? Math.max(...utils) : null;
  const text =
    `$(pulse) GPU ×${gs.length} avg ${dash(avg)}% · max ${dash(max)}%` + (hottest !== null ? ` · hottest ${Math.round(hottest)}°C` : '');
  // macOS reports neither temperature nor power for most GPUs: drop columns no GPU fills.
  const showTemp = temps.length > 0 || gs.some((g) => g.vendor !== 'apple');
  const showPower = gs.some((g) => g.powerW !== null || g.vendor !== 'apple');
  const memHead = gs.some((g) => g.unifiedMemory) ? 'GPU mem' : 'VRAM';
  const rows = gs.map((g) => {
    const hot = g.tempC !== null && g.tempC >= tempWarn ? ' $(flame)' : '';
    return `| ${g.index} · ${shortName(g.name)} | ${dash(g.util)}% | ${vramGb(g)} |` +
      (showTemp ? ` ${dash(g.tempC)} °C${hot} |` : '') + (showPower ? ` ${dash(g.powerW)} W |` : '');
  });
  const head = `| GPU | Util | ${memHead} |${showTemp ? ' Temp |' : ''}${showPower ? ' Power |' : ''}\n` +
    `|:--|--:|--:|${showTemp ? '--:|' : ''}${showPower ? '--:|' : ''}\n`;
  const tooltip =
    `**GPU Pulse** — ${gs.length} GPUs\n\n` +
    head +
    rows.join('\n') +
    '\n\n' +
    cpuLine(s) +
    alertLines(alerts, s.t) +
    '_Click to open the dashboard_';
  return { text, tooltip, warn };
}
