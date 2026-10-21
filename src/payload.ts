import type { HistoryStore } from './history';
import { buildInsights, grade } from './insights';
import type { GpuProcess, Sample, Specs } from './types';

const WINDOWS: [string, number][] = [
  ['5 min', 5],
  ['30 min', 30],
  ['1 hour', 60],
  ['24 hours', 1440],
];
const LIVE_WINDOW_MS = 60 * 60_000;
// The UI grid aligns on clock hours, so it needs up to one extra hour of minutes.
export const HEATMAP_HOURS = 25;

/** Pick the selected GPU if it still exists, else the first one. */
export function resolveGpu(s: Sample | undefined, selected: string | undefined): string | undefined {
  const ids = s?.gpus.map((g) => g.uuid) ?? [];
  return selected && ids.includes(selected) ? selected : ids[0];
}

export function analysis(history: HistoryStore, s: Sample | undefined, id: string | undefined, tempWarn: number) {
  const windows = WINDOWS.map(([label, min]) => history.stats(label, min, id));
  const gpu = s?.gpus.find((g) => g.uuid === id);
  return {
    windows,
    grade: grade(windows[2].util),
    insights: buildInsights(windows[2], windows[1], gpu, tempWarn),
  };
}

export interface InitContext {
  history: HistoryStore;
  sample: Sample | undefined;
  selected: string | undefined;
  vendor: string;
  processes: GpuProcess[];
  specs: Specs | undefined;
  tempWarn: number;
  intervalMs: number;
}

/** Full state message sent when the webview (re)loads or the selected GPU changes. */
export function initPayload(c: InitContext) {
  const id = resolveGpu(c.sample, c.selected);
  return {
    type: 'init',
    selected: id,
    vendor: c.vendor,
    platform: process.platform,
    config: { tempWarn: c.tempWarn, intervalMs: c.intervalMs },
    sample: c.sample,
    recent: c.history.recentPoints(id, Date.now() - LIVE_WINDOW_MS),
    heatmap: c.history.heatmap(id, HEATMAP_HOURS),
    processes: c.processes,
    specs: c.specs,
    ...analysis(c.history, c.sample, id, c.tempWarn),
  };
}
