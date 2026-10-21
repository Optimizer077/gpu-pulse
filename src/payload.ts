import { toPoint, type HistoryStore, type RecentPoint } from './history';
import { buildFleetInsights, buildInsights, bySeverity, FLEET_WINDOW_MIN, grade, type FleetEntry } from './insights';
import type { Alert } from './alerts';
import type { GpuProcess, GpuSample, Sample, Specs } from './types';

const WINDOWS: [string, number][] = [
  ['5 min', 5],
  ['30 min', 30],
  ['1 hour', 60],
  ['24 hours', 1440],
];
const LIVE_WINDOW_MS = 60 * 60_000;
// The UI grid aligns on clock hours, so it needs up to one extra hour of minutes.
export const HEATMAP_HOURS = 25;
/** Window stats and insights change slowly; recompute them at most this often, not every tick. */
export const ANALYSIS_EVERY_MS = 2000;

/** Pick the selected GPU if it still exists, else the first one. */
export function resolveGpu(s: Sample | undefined, selected: string | undefined): string | undefined {
  const ids = s?.gpus.map((g) => g.uuid) ?? [];
  return selected && ids.includes(selected) ? selected : ids[0];
}

/** Per-GPU recent averages for every GPU in the sample (multi-GPU strip and fleet insights). */
export function fleetSummary(history: HistoryStore, s: Sample | undefined, now = Date.now()): FleetEntry[] {
  return (s?.gpus ?? []).map((g) => {
    const w = history.stats('fleet', FLEET_WINDOW_MIN, g.uuid, now);
    return { uuid: g.uuid, index: g.index, util: w.util, vram: w.vram, covered: w.coveredMinutes, tempC: g.tempC };
  });
}

export function analysis(history: HistoryStore, s: Sample | undefined, id: string | undefined, tempWarn: number) {
  const windows = WINDOWS.map(([label, min]) => history.stats(label, min, id));
  const gpu = s?.gpus.find((g) => g.uuid === id);
  const fleet = s && s.gpus.length > 1 ? fleetSummary(history, s) : [];
  return {
    windows,
    grade: grade(windows[2].util),
    insights: bySeverity([...buildInsights(windows[2], windows[1], gpu, tempWarn), ...buildFleetInsights(fleet, id, tempWarn)]),
    fleet: fleet.map((f) => ({ uuid: f.uuid, util: f.util === null ? null : Math.round(f.util * 10) / 10 })),
  };
}

export type Analysis = ReturnType<typeof analysis>;

/** What the views show for an active alert (rendered as text, escaped in the webview). */
export type AlertView = Pick<Alert, 'level' | 'title' | 'detail'>;

export const alertViews = (active: Alert[]): AlertView[] => active.map((a) => ({ level: a.level, title: a.title, detail: a.detail }));

export interface InitContext {
  history: HistoryStore;
  sample: Sample | undefined;
  selected: string | undefined;
  vendor: string;
  processes: GpuProcess[];
  specs: Specs | undefined;
  tempWarn: number;
  intervalMs: number;
  /** Alerts whose condition still holds. */
  alerts: AlertView[];
  /** Side bar: 5 minutes of live points, no heatmap / processes / specs. */
  compact?: boolean;
}

const SIDEBAR_WINDOW_MS = 5 * 60_000;

/** Full state message sent when a webview (re)loads or the selected GPU changes. */
export function initPayload(c: InitContext) {
  const id = resolveGpu(c.sample, c.selected);
  return {
    type: 'init',
    selected: id,
    vendor: c.vendor,
    platform: process.platform,
    config: { tempWarn: c.tempWarn, intervalMs: c.intervalMs },
    sample: c.sample && compactSample(c.sample),
    recent: c.history.recentPoints(id, Date.now() - (c.compact ? SIDEBAR_WINDOW_MS : LIVE_WINDOW_MS)),
    heatmap: c.compact ? null : c.history.heatmap(id, HEATMAP_HOURS),
    processes: c.compact ? [] : c.processes,
    specs: c.compact ? undefined : c.specs,
    analysis: analysis(c.history, c.sample, id, c.tempWarn),
    alerts: c.alerts,
  };
}

const r1 = (x: number | null) => (x === null ? null : Math.round(x * 10) / 10);

/** Same shape as Sample, with display precision only (smaller messages, fewer changed digits). */
export function compactSample(s: Sample): Sample {
  return {
    t: s.t,
    gpus: s.gpus.map((g): GpuSample => ({ ...g, powerW: r1(g.powerW), memUsedMiB: g.memUsedMiB === null ? null : Math.round(g.memUsedMiB) })),
    cpu: { util: Math.round(s.cpu.util * 10) / 10, perCore: s.cpu.perCore.map(Math.round) },
    mem: s.mem,
  };
}

export interface TickMessage {
  type: 'tick';
  sample: Sample;
  point: RecentPoint;
  selected: string | undefined;
  /** Present only when recomputed (at most every ANALYSIS_EVERY_MS) and different from the last one sent. */
  analysis?: Analysis;
  /** Present only when the set of active alerts changed. */
  alerts?: AlertView[];
}

/**
 * Builds the per-tick dashboard message. The live part (sample + one chart point) is
 * O(GPUs); window stats and insights are recomputed at most every ANALYSIS_EVERY_MS
 * and only sent when they changed.
 */
export class TickComposer {
  private lastAt = 0;
  private lastKey = '';

  constructor(
    private readonly history: HistoryStore,
    private readonly tempWarn: () => number,
  ) {}

  /** Record the analysis that went out with an init message (single consumer). */
  sent(a: Analysis, t = Date.now()): void {
    this.lastAt = t;
    this.lastKey = JSON.stringify(a);
  }

  /** Include analysis in the next tick (a view (re)loaded or the selection changed; views may hold different copies). */
  invalidate(): void {
    this.lastAt = 0;
    this.lastKey = '';
  }

  tick(s: Sample, selected: string | undefined): TickMessage {
    const id = resolveGpu(s, selected);
    const msg: TickMessage = { type: 'tick', sample: compactSample(s), point: toPoint(s, id), selected: id };
    if (s.t - this.lastAt >= ANALYSIS_EVERY_MS || s.t < this.lastAt) {
      this.lastAt = s.t;
      const a = analysis(this.history, s, id, this.tempWarn());
      const key = JSON.stringify(a);
      if (key !== this.lastKey) {
        this.lastKey = key;
        msg.analysis = a;
      }
    }
    return msg;
  }
}
