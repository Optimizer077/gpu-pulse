import { HELD_VRAM, IDLE_UTIL } from './history';
import type { GpuSample, Insight, WindowStats } from './types';

export interface Grade {
  label: string;
  level: 'idle' | 'low' | 'good' | 'max';
}

/** Coarse verdict from mean utilization. */
export function grade(util: number | null): Grade | null {
  if (util === null) return null;
  if (util < 20) return { label: 'Mostly idle', level: 'idle' };
  if (util < 50) return { label: 'Underused', level: 'low' };
  if (util < 85) return { label: 'Well used', level: 'good' };
  return { label: 'Saturated', level: 'max' };
}

const pct = (x: number) => `${Math.round(x * 100)}%`;

/** One GPU's recent averages, for comparing GPUs with each other. */
export interface FleetEntry {
  uuid: string;
  index: number;
  util: number | null; // mean % over the fleet window
  vram: number | null; // mean VRAM % over the fleet window
  covered: number; // minutes with data in the window
  tempC: number | null; // live
}

export const FLEET_WINDOW_MIN = 10;
const FLEET_MIN_COVERED = 3;
const LEVEL_ORDER: Record<Insight['level'], number> = { critical: 0, serious: 1, warning: 2, info: 3, good: 4 };

/** Stable sort so critical items lead. */
export function bySeverity(list: Insight[]): Insight[] {
  return list
    .map((x, i) => [x, i] as const)
    .sort((a, b) => LEVEL_ORDER[a[0].level] - LEVEL_ORDER[b[0].level] || a[1] - b[1])
    .map(([x]) => x);
}

const gpuList = (xs: FleetEntry[]) => {
  const ids = xs.map((x) => x.index);
  return ids.length === 1 ? `GPU ${ids[0]}` : `GPUs ${ids.slice(0, -1).join(', ')} and ${ids[ids.length - 1]}`;
};

/**
 * Observations that need more than one GPU: one GPU far below the others while they are
 * busy, uneven memory, heat on a GPU other than the one being viewed. The data cannot say
 * whether the GPUs share a job, so the wording names both readings.
 */
export function buildFleetInsights(fleet: FleetEntry[], selected: string | undefined, tempWarn: number): Insight[] {
  const out: Insight[] = [];
  if (fleet.length < 2) return out;

  for (const f of fleet) {
    if (f.uuid !== selected && f.tempC !== null && f.tempC >= tempWarn) {
      out.push({
        level: 'critical',
        title: `GPU ${f.index} running hot: ${Math.round(f.tempC)} °C`,
        detail: `At or above your ${tempWarn} °C threshold. Select GPU ${f.index} above for details.`,
      });
    }
  }

  const ready = fleet.filter((f): f is FleetEntry & { util: number } => f.util !== null && f.covered >= FLEET_MIN_COVERED);
  if (ready.length < 2) return out;
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  // State the data actually behind each claim (3–10 recorded minutes), not the nominal window.
  const span = (fs: FleetEntry[]) => Math.min(FLEET_WINDOW_MIN, ...fs.map((f) => f.covered));
  const lastMin = (fs: FleetEntry[]) => `Last ${span(fs)} recorded min`;

  // A GPU is lagging when the others average >= 50% and it sits under half of that, at least 25 points lower.
  const lagging = ready.filter((f) => {
    const others = mean(ready.filter((o) => o !== f).map((o) => o.util));
    return others >= 50 && f.util < others * 0.5 && others - f.util >= 25;
  });
  if (lagging.length && lagging.length < ready.length) {
    const rest = ready.filter((f) => !lagging.includes(f));
    const restMean = Math.round(mean(rest.map((f) => f.util)));
    const lagText = lagging.map((f) => `GPU ${f.index} ${Math.round(f.util)}%`).join(', ');
    out.push({
      level: 'warning',
      title: `${gpuList(lagging)} well below the others`,
      detail:
        `${lastMin(ready)}: ${lagText} vs ${restMean}% mean on the other GPUs. ` +
        'If one job spans these GPUs (e.g. data-parallel training), this fits a straggler, an uneven data split, ' +
        'or a rank that is not using its GPU. If they run separate jobs, it is expected.',
    });
  } else if (ready.length === fleet.length && ready.every((f) => f.util >= 70)) {
    out.push({
      level: 'good',
      title: `All ${fleet.length} GPUs kept busy`,
      detail: `${lastMin(ready)}, means: ${ready.map((f) => `GPU ${f.index} ${Math.round(f.util)}%`).join(', ')}.`,
    });
  }

  const withMem = ready.filter((f): f is typeof f & { vram: number } => f.vram !== null);
  const inUse = ready.filter((f) => f.util >= 20).length;
  if (withMem.length >= 2 && inUse >= 2) {
    const hi = withMem.reduce((a, b) => (b.vram > a.vram ? b : a));
    const lo = withMem.reduce((a, b) => (b.vram < a.vram ? b : a));
    if (hi.vram >= 50 && hi.vram - lo.vram >= 30) {
      out.push({
        level: 'info',
        title: 'Uneven memory across GPUs',
        detail:
          `${lastMin(withMem)}: GPU ${hi.index} averaged ${Math.round(hi.vram)}% VRAM, GPU ${lo.index} ${Math.round(lo.vram)}%. ` +
          (hi.index === 0
            ? 'In data-parallel jobs rank 0 often holds extra state (gathered outputs, logging); a large gap can also mean an unbalanced model split.'
            : 'A large gap can mean an unbalanced model or data split, or different jobs on each GPU.'),
      });
    }
  }
  return out;
}

/**
 * Plain-language observations about the last hour. Thresholds are deliberately
 * conservative and every message says what was measured, not why.
 */
export function buildInsights(
  hour: WindowStats,
  halfHour: WindowStats,
  gpu: GpuSample | undefined,
  tempWarn: number,
): Insight[] {
  const out: Insight[] = [];
  if (!gpu) {
    out.push({
      level: 'info',
      title: 'No supported GPU detected',
      detail:
        process.platform === 'darwin'
          ? 'ioreg lists no GPU accelerator (common in virtual machines). CPU and memory are still tracked.'
          : 'nvidia-smi was not found and no amdgpu device exposes utilization. CPU and memory are still tracked.',
    });
  }
  if (hour.coveredMinutes < 5) {
    out.push({
      level: 'info',
      title: 'Building history',
      detail: `${hour.coveredMinutes} min recorded so far. Averages become meaningful after a few minutes; history persists across restarts.`,
    });
  }

  // Unified memory (Apple Silicon): the share is GPU-allocated memory of all system RAM.
  const unified = !!gpu?.unifiedMemory;
  if (gpu) {
    const vram = gpu.memUsedMiB !== null && gpu.memTotalMiB ? gpu.memUsedMiB / gpu.memTotalMiB : null;
    if (vram !== null && vram >= 0.95) {
      out.push({
        level: 'serious',
        title: unified ? 'GPU memory nearly full' : 'VRAM nearly full',
        detail: `${pct(vram)} of ${unified ? 'unified (system) memory is allocated by the GPU' : 'video memory is allocated'}. New allocations may fail with out-of-memory errors.`,
      });
    }
    if (gpu.tempC !== null && gpu.tempC >= tempWarn) {
      out.push({
        level: 'critical',
        title: `Running hot: ${Math.round(gpu.tempC)} °C`,
        detail: `At or above your warning threshold of ${tempWarn} °C. Sustained heat can lower clocks (thermal throttling).`,
      });
    } else if (hour.tempPeak !== null && hour.tempPeak >= tempWarn) {
      out.push({
        level: 'warning',
        title: `Peaked at ${Math.round(hour.tempPeak)} °C this hour`,
        detail: `Above your ${tempWarn} °C threshold at least once in the last hour.`,
      });
    }
  }

  if (hour.coveredMinutes >= 5 && hour.util !== null) {
    if (hour.heldIdleShare !== null && hour.heldIdleShare >= 0.25) {
      out.push({
        level: 'warning',
        title: 'Memory held while GPU idles',
        detail: `For ${pct(hour.heldIdleShare)} of the last hour, at least ${HELD_VRAM}% of ${unified ? 'unified memory was allocated by the GPU' : 'VRAM was allocated'} while utilization stayed under ${IDLE_UTIL}%. An open notebook or idle process may be holding memory.`,
      });
    }
    if (hour.util < 20) {
      out.push({
        level: 'info',
        title: 'GPU mostly idle',
        detail: `Averaged ${Math.round(hour.util)}% over ${hour.coveredMinutes} recorded minutes; idle (<10%) ${pct(hour.idleShare ?? 0)} of the time.`,
      });
    } else if (hour.busyShare !== null && hour.busyShare >= 0.6) {
      out.push({
        level: 'good',
        title: 'GPU kept busy',
        detail: `Above 70% utilization for ${pct(hour.busyShare)} of the last hour (mean ${Math.round(hour.util)}%).`,
      });
    } else if (hour.util < 50) {
      out.push({
        level: 'info',
        title: 'Utilization is intermittent',
        detail: `Mean ${Math.round(hour.util)}% with frequent idle gaps. In training loops this pattern often points at data loading or CPU-side work between steps.`,
      });
    }
    if (
      halfHour.cpu !== null && halfHour.cpu >= 85 &&
      halfHour.util !== null && halfHour.util < 50
    ) {
      out.push({
        level: 'warning',
        title: 'CPU saturated, GPU waiting',
        detail: `Last 30 min: CPU ${Math.round(halfHour.cpu)}% vs GPU ${Math.round(halfHour.util)}%. The workload may be CPU-bound.`,
      });
    }
    if (gpu?.powerLimitW && hour.power && hour.util >= 70 && hour.power / gpu.powerLimitW < 0.5) {
      out.push({
        level: 'info',
        title: 'High utilization, low power draw',
        detail: `Utilization counts time any kernel runs, not how many cores are busy. Drawing ${Math.round(hour.power)} W of ${Math.round(gpu.powerLimitW)} W suggests small or memory-bound kernels.`,
      });
    }
  }
  return out;
}
