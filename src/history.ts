import type { GpuMinute, MinuteBucket, Sample, WindowStats } from './types';

const MINUTE = 60_000;
const RECENT_MS = 60 * MINUTE;
export const IDLE_UTIL = 10;
export const BUSY_UTIL = 70;
// High enough that a desktop's own compositor/app usage (~20–35%) doesn't count as "held".
// On unified memory (Apple Silicon) the percentage is GPU-allocated memory of all system RAM
// (memTotalMiB = RAM size), so "held" means memory the GPU driver has allocated.
export const HELD_VRAM = 50;

interface GpuAcc {
  n: number;
  util: number;
  utilMax: number;
  vram: number;
  power: number;
  tempMax: number;
  idle: number;
  busy: number;
  heldIdle: number;
}

interface Acc {
  t: number;
  n: number;
  cpu: number;
  mem: number;
  gpus: Record<string, GpuAcc>;
}

export interface SerializedHistory {
  version: 1;
  minutes: MinuteBucket[];
  /** Buckets older than this were deleted by "Clear history" (respected when merging other windows' saves). */
  clearedAt?: number;
}

/** A compact point for the live chart: [t, gpuUtil, vram%, cpu%, ram%]. */
export type RecentPoint = [number, number | null, number | null, number, number];

/**
 * One raw sample reduced to what the live chart needs. At 250 ms an hour is 14 400
 * entries, so full Sample objects (per-core arrays, names, clocks) are not kept.
 * `ids` is shared between consecutive entries while the GPU set is unchanged.
 */
interface RecentEntry {
  t: number;
  cpu: number;
  ram: number;
  ids: string[];
  util: (number | null)[];
  vram: (number | null)[];
}

const r1 = (x: number) => Math.round(x * 10) / 10;

function ramPct(s: Sample): number {
  return s.mem.totalBytes ? r1((s.mem.usedBytes / s.mem.totalBytes) * 100) : 0;
}

/** Live-chart point for one GPU of a sample (falls back to the first GPU). */
export function toPoint(s: Sample, gpuUuid: string | undefined): RecentPoint {
  const g = s.gpus.find((x) => x.uuid === gpuUuid) ?? s.gpus[0];
  const v = g ? vramPct(g.memUsedMiB, g.memTotalMiB) : null;
  return [s.t, g?.util ?? null, v === null ? null : r1(v), r1(s.cpu.util), ramPct(s)];
}

export function minuteStart(t: number): number {
  return Math.floor(t / MINUTE) * MINUTE;
}

const r2 = (x: number) => Math.round(x * 100) / 100;

/**
 * A clear removes everything through the end of its minute: a bucket can't be split, so
 * up to 60 s recorded right after the clear is dropped rather than leaving old data.
 */
function keepFromClear(clearedAt: number): number {
  return clearedAt ? minuteStart(clearedAt) + MINUTE : 0;
}

function vramPct(used: number | null, total: number | null): number | null {
  return used !== null && total ? (used / total) * 100 : null;
}

function finalize(acc: Acc): MinuteBucket {
  const gpus: Record<string, GpuMinute> = {};
  for (const [id, g] of Object.entries(acc.gpus)) {
    if (!g.n) continue;
    gpus[id] = {
      n: g.n,
      util: r2(g.util / g.n),
      utilMax: g.utilMax,
      vram: r2(g.vram / g.n),
      power: r2(g.power / g.n),
      tempMax: g.tempMax,
      idle: g.idle,
      busy: g.busy,
      heldIdle: g.heldIdle,
    };
  }
  return { t: acc.t, n: acc.n, cpu: acc.n ? r2(acc.cpu / acc.n) : 0, mem: acc.n ? r2(acc.mem / acc.n) : 0, gpus };
}

function isBucket(m: unknown): m is MinuteBucket {
  const b = m as MinuteBucket;
  return !!b && typeof b.t === 'number' && typeof b.n === 'number' && !!b.gpus && typeof b.gpus === 'object';
}

/** Index of the bucket with time `t`, or -(insertion point) - 1. Buckets are sorted by t. */
function search(list: MinuteBucket[], t: number): number {
  let lo = 0, hi = list.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid].t === t) return mid;
    if (list[mid].t < t) lo = mid + 1;
    else hi = mid - 1;
  }
  return -lo - 1;
}

/** Combine two aggregates of the same minute, weighting means by sample count. */
function mergeBuckets(a: MinuteBucket, b: MinuteBucket): MinuteBucket {
  const n = a.n + b.n;
  const gpus: Record<string, GpuMinute> = { ...a.gpus };
  for (const [id, g] of Object.entries(b.gpus)) {
    const x = gpus[id];
    if (!x) {
      gpus[id] = g;
      continue;
    }
    const gn = x.n + g.n;
    const w = (p: number, q: number) => r2((p * x.n + q * g.n) / gn);
    gpus[id] = {
      n: gn,
      util: w(x.util, g.util),
      utilMax: Math.max(x.utilMax, g.utilMax),
      vram: w(x.vram, g.vram),
      power: w(x.power, g.power),
      tempMax: Math.max(x.tempMax, g.tempMax),
      idle: x.idle + g.idle,
      busy: x.busy + g.busy,
      heldIdle: x.heldIdle + g.heldIdle,
    };
  }
  const w = (p: number, q: number) => (n ? r2((p * a.n + q * b.n) / n) : 0);
  return { t: a.t, n, cpu: w(a.cpu, b.cpu), mem: w(a.mem, b.mem), gpus };
}

/**
 * Usage history: raw samples for the last hour (live chart) and per-minute
 * buckets for long windows (averages, heatmap). Only minute buckets persist.
 */
export class HistoryStore {
  private recent: RecentEntry[] = [];
  private lastIds: string[] = [];
  private minutes: MinuteBucket[] = [];
  private acc: Acc | null = null;
  private clearedAt = 0;

  constructor(private retentionDays: number) {}

  setRetention(days: number): void {
    this.retentionDays = days;
  }

  load(data: unknown): void {
    const d = data as SerializedHistory | undefined;
    if (d?.version !== 1 || !Array.isArray(d.minutes)) return;
    this.clearedAt = Math.max(this.clearedAt, d.clearedAt ?? 0);
    const keepFrom = keepFromClear(this.clearedAt);
    this.minutes = d.minutes.filter(isBucket).filter((m) => m.t >= keepFrom).sort((a, b) => a.t - b.t);
  }

  /**
   * Fold in history saved by another VS Code window. For a minute both recorded,
   * keep the bucket with more samples (both describe the same wall-clock minute).
   */
  mergeFrom(data: unknown): void {
    const d = data as SerializedHistory | undefined;
    if (d?.version !== 1 || !Array.isArray(d.minutes)) return;
    if ((d.clearedAt ?? 0) > this.clearedAt) {
      this.clearedAt = d.clearedAt ?? 0;
      const from = keepFromClear(this.clearedAt);
      this.minutes = this.minutes.filter((m) => m.t >= from);
      this.recent = this.recent.filter((s) => s.t >= this.clearedAt);
      if (this.acc && this.acc.t < from) this.acc = null;
    }
    const keepFrom = keepFromClear(this.clearedAt);
    for (const m of d.minutes) {
      // The open minute is still accumulating here; a saved partial copy of it would be double-counted.
      if (!isBucket(m) || m.t < keepFrom || (this.acc && m.t >= this.acc.t)) continue;
      const i = search(this.minutes, m.t);
      if (i < 0) this.minutes.splice(-i - 1, 0, m);
      else if (m.n > this.minutes[i].n) this.minutes[i] = m;
    }
  }

  clear(now = Date.now()): void {
    this.recent = [];
    this.minutes = [];
    this.acc = null;
    this.clearedAt = now;
  }

  add(s: Sample): void {
    let ids = this.lastIds;
    if (ids.length !== s.gpus.length || s.gpus.some((g, i) => g.uuid !== ids[i])) ids = this.lastIds = s.gpus.map((g) => g.uuid);
    this.recent.push({
      t: s.t,
      cpu: r1(s.cpu.util),
      ram: ramPct(s),
      ids,
      util: s.gpus.map((g) => g.util),
      vram: s.gpus.map((g) => {
        const v = vramPct(g.memUsedMiB, g.memTotalMiB);
        return v === null ? null : r1(v);
      }),
    });
    const cutoff = s.t - RECENT_MS;
    let drop = 0;
    while (drop < this.recent.length && this.recent[drop].t < cutoff) drop++;
    if (drop) this.recent.splice(0, drop);

    const m = minuteStart(s.t);
    if (this.acc && this.acc.t !== m) this.flush();
    if (!this.acc) this.acc = { t: m, n: 0, cpu: 0, mem: 0, gpus: {} };

    const a = this.acc;
    a.n++;
    a.cpu += s.cpu.util;
    a.mem += s.mem.totalBytes ? (s.mem.usedBytes / s.mem.totalBytes) * 100 : 0;
    for (const g of s.gpus) {
      if (g.util === null) continue;
      const v = vramPct(g.memUsedMiB, g.memTotalMiB) ?? 0;
      const ga = (a.gpus[g.uuid] ??= { n: 0, util: 0, utilMax: 0, vram: 0, power: 0, tempMax: 0, idle: 0, busy: 0, heldIdle: 0 });
      ga.n++;
      ga.util += g.util;
      ga.utilMax = Math.max(ga.utilMax, g.util);
      ga.vram += v;
      ga.power += g.powerW ?? 0;
      ga.tempMax = Math.max(ga.tempMax, g.tempC ?? 0);
      if (g.util < IDLE_UTIL) ga.idle++;
      if (g.util >= BUSY_UTIL) ga.busy++;
      if (g.util < IDLE_UTIL && v >= HELD_VRAM) ga.heldIdle++;
    }
  }

  /** Close the in-progress minute, merging into an existing bucket for that minute (restart, clock skew). */
  private flush(): void {
    if (!this.acc) return;
    if (this.acc.n) {
      const b = finalize(this.acc);
      const i = search(this.minutes, b.t);
      if (i >= 0) this.minutes[i] = mergeBuckets(this.minutes[i], b);
      else this.minutes.splice(-i - 1, 0, b);
    }
    this.acc = null;
    const cutoff = Date.now() - this.retentionDays * 24 * 60 * MINUTE;
    let drop = 0;
    while (drop < this.minutes.length && this.minutes[drop].t < cutoff) drop++;
    if (drop) this.minutes.splice(0, drop);
  }

  /** Visit buckets with t >= from, newest first, including the in-progress minute. */
  private eachSince(from: number, fn: (b: MinuteBucket) => void): void {
    const open = this.acc && this.acc.n ? finalize(this.acc) : null;
    let i = this.minutes.length - 1;
    if (open) {
      const same = i >= 0 && this.minutes[i].t === open.t;
      fn(same ? mergeBuckets(this.minutes[i], open) : open);
      if (same) i--;
    }
    for (; i >= 0 && this.minutes[i].t >= from; i--) fn(this.minutes[i]);
  }

  recentPoints(gpuUuid: string | undefined, sinceMs: number): RecentPoint[] {
    const out: RecentPoint[] = [];
    let i = this.recent.length;
    while (i > 0 && this.recent[i - 1].t >= sinceMs) i--;
    for (; i < this.recent.length; i++) {
      const e = this.recent[i];
      const k = Math.max(0, e.ids.indexOf(gpuUuid ?? ''));
      out.push([e.t, e.util[k] ?? null, e.vram[k] ?? null, e.cpu, e.ram]);
    }
    return out;
  }

  /** Weighted aggregate over the last `minutes` minutes for one GPU. */
  stats(label: string, minutes: number, gpuUuid: string | undefined, now = Date.now()): WindowStats {
    const to = minuteStart(now);
    const from = to - (minutes - 1) * MINUTE;
    let covered = 0, n = 0, util = 0, vram = 0, power = 0, idle = 0, busy = 0, held = 0;
    let peak: number | null = null, tempPeak: number | null = null;
    let cn = 0, cpu = 0, mem = 0;
    this.eachSince(from, (b) => {
      if (b.t < from || b.t > to) return;
      // With a GPU selected, only minutes that GPU recorded count (a GPU back from a driver
      // reset must not borrow coverage from its siblings); otherwise any recorded minute counts.
      if (!gpuUuid || b.gpus[gpuUuid]) covered++;
      cn += b.n;
      cpu += b.cpu * b.n;
      mem += b.mem * b.n;
      const g = gpuUuid ? b.gpus[gpuUuid] : Object.values(b.gpus)[0];
      if (!g) return;
      n += g.n;
      util += g.util * g.n;
      vram += g.vram * g.n;
      power += g.power * g.n;
      idle += g.idle;
      busy += g.busy;
      held += g.heldIdle;
      peak = Math.max(peak ?? 0, g.utilMax);
      tempPeak = Math.max(tempPeak ?? 0, g.tempMax);
    });
    return {
      label,
      minutes,
      coveredMinutes: covered,
      util: n ? util / n : null,
      utilPeak: peak,
      vram: n ? vram / n : null,
      power: n ? power / n : null,
      tempPeak,
      idleShare: n ? idle / n : null,
      busyShare: n ? busy / n : null,
      heldIdleShare: n ? held / n : null,
      cpu: cn ? cpu / cn : null,
      mem: cn ? mem / cn : null,
    };
  }

  /** Mean GPU util per minute for the last `hours` hours; null where nothing was recorded. */
  heatmap(gpuUuid: string | undefined, hours: number, now = Date.now()): { start: number; cells: (number | null)[] } {
    const count = hours * 60;
    const start = minuteStart(now) - (count - 1) * MINUTE;
    const cells: (number | null)[] = new Array(count).fill(null);
    this.eachSince(start, (b) => {
      const i = Math.round((b.t - start) / MINUTE);
      if (i < 0 || i >= count) return;
      const g = gpuUuid ? b.gpus[gpuUuid] : Object.values(b.gpus)[0];
      if (g) cells[i] = Math.round(g.util * 10) / 10;
    });
    return { start, cells };
  }

  /** Every persisted minute as CSV rows (one row per GPU per minute). */
  toCsv(): string {
    const rows = ['minute_utc,gpu_uuid,samples,gpu_util_mean,gpu_util_max,vram_pct_mean,power_w_mean,temp_c_max,cpu_pct_mean,ram_pct_mean'];
    const all: MinuteBucket[] = [];
    this.eachSince(0, (b) => all.push(b));
    for (const b of all.reverse()) {
      const ts = new Date(b.t).toISOString();
      const gpus = Object.entries(b.gpus);
      if (!gpus.length) rows.push([ts, '', b.n, '', '', '', '', '', b.cpu, b.mem].join(','));
      for (const [id, g] of gpus) {
        rows.push([ts, id, g.n, g.util, g.utilMax, g.vram, g.power, g.tempMax, b.cpu, b.mem].join(','));
      }
    }
    return rows.join('\n') + '\n';
  }

  /** Serializable state including the in-progress minute (merged, not double-counted). */
  snapshotForSave(): SerializedHistory {
    const all: MinuteBucket[] = [];
    this.eachSince(0, (b) => all.push(b));
    return { version: 1, minutes: all.reverse(), clearedAt: this.clearedAt || undefined };
  }
}
