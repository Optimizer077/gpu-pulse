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
      detail: 'nvidia-smi was not found and no amdgpu device exposes utilization. CPU and memory are still tracked.',
    });
  }
  if (hour.coveredMinutes < 5) {
    out.push({
      level: 'info',
      title: 'Building history',
      detail: `${hour.coveredMinutes} min recorded so far. Averages become meaningful after a few minutes; history persists across restarts.`,
    });
  }

  if (gpu) {
    const vram = gpu.memUsedMiB !== null && gpu.memTotalMiB ? gpu.memUsedMiB / gpu.memTotalMiB : null;
    if (vram !== null && vram >= 0.95) {
      out.push({
        level: 'serious',
        title: 'VRAM nearly full',
        detail: `${pct(vram)} of video memory is allocated. New allocations may fail with out-of-memory errors.`,
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
        detail: `For ${pct(hour.heldIdleShare)} of the last hour, at least ${HELD_VRAM}% of VRAM was allocated while utilization stayed under ${IDLE_UTIL}%. An open notebook or idle process may be holding memory.`,
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
