import { describe, expect, it } from 'vitest';
import { HistoryStore } from '../src/history';
import { buildInsights, grade } from '../src/insights';
import type { GpuSample, Sample } from '../src/types';

const T0 = Date.UTC(2026, 9, 21, 10, 0, 0);

function gpu(util: number, usedMiB = 1000): GpuSample {
  return {
    index: 0, uuid: 'G', name: 'Test GPU', vendor: 'nvidia', util, memUtil: 0, memUsedMiB: usedMiB, memTotalMiB: 10000,
    tempC: 60, powerW: 100, powerLimitW: 200, clockMHz: null, clockMaxMHz: null, memClockMHz: null, fanPct: null,
    pstate: null, pcieGen: null, pcieWidth: null,
  };
}

function sample(t: number, util: number, usedMiB?: number, cpu = 20): Sample {
  return { t, gpus: [gpu(util, usedMiB)], cpu: { util: cpu, perCore: [cpu] }, mem: { usedBytes: 50, totalBytes: 100 } };
}

/** One sample every 10 s for `minutes` minutes. */
function fill(h: HistoryStore, from: number, minutes: number, util: (i: number) => number, usedMiB?: number) {
  for (let i = 0; i < minutes * 6; i++) h.add(sample(from + i * 10_000, util(i), usedMiB));
}

describe('HistoryStore', () => {
  it('computes weighted means, peaks and idle/busy shares', () => {
    const h = new HistoryStore(7);
    fill(h, T0, 10, (i) => (i % 2 ? 100 : 0)); // half busy, half idle
    const s = h.stats('10m', 10, 'G', T0 + 10 * 60_000 - 1);
    expect(s.coveredMinutes).toBe(10);
    expect(s.util).toBeCloseTo(50);
    expect(s.utilPeak).toBe(100);
    expect(s.idleShare).toBeCloseTo(0.5);
    expect(s.busyShare).toBeCloseTo(0.5);
    expect(s.mem).toBeCloseTo(50);
  });

  it('only counts minutes inside the window', () => {
    const h = new HistoryStore(7);
    fill(h, T0, 60, (i) => (i < 30 * 6 ? 0 : 80));
    const last30 = h.stats('30m', 30, 'G', T0 + 60 * 60_000 - 1);
    expect(last30.coveredMinutes).toBe(30);
    expect(last30.util).toBeCloseTo(80);
    const hour = h.stats('1h', 60, 'G', T0 + 60 * 60_000 - 1);
    expect(hour.util).toBeCloseTo(40);
  });

  it('flags VRAM held while idle', () => {
    const h = new HistoryStore(7);
    fill(h, T0, 5, () => 2, 6000); // 60% VRAM, util 2%
    expect(h.stats('5m', 5, 'G', T0 + 5 * 60_000 - 1).heldIdleShare).toBe(1);
  });

  it('round-trips through serialization including the open minute', () => {
    const h = new HistoryStore(7);
    fill(h, T0, 3, () => 40);
    h.add(sample(T0 + 3 * 60_000 + 5_000, 90)); // open minute
    const copy = new HistoryStore(7);
    copy.load(JSON.parse(JSON.stringify(h.snapshotForSave())));
    const s = copy.stats('5m', 5, 'G', T0 + 4 * 60_000 - 1);
    expect(s.coveredMinutes).toBe(4);
    expect(s.utilPeak).toBe(90);
  });

  it('merges a restart within the same minute instead of replacing it', () => {
    const before = new HistoryStore(7);
    fill(before, T0, 1, () => 20); // 6 samples, minute still open
    const after = new HistoryStore(7);
    after.load(JSON.parse(JSON.stringify(before.snapshotForSave())));
    after.add(sample(T0 + 50_000, 80)); // same minute, after "restart"
    after.add(sample(T0 + 61_000, 0)); // closes the minute
    const s = after.stats('1m', 1, 'G', T0 + 59_000);
    expect(s.util).toBeCloseTo((20 * 6 + 80) / 7);
  });

  it('merges another window without double-counting its own open minute', () => {
    const a = new HistoryStore(7);
    fill(a, T0, 2, () => 50);
    a.add(sample(T0 + 2 * 60_000 + 1000, 50)); // open minute
    const saved = JSON.parse(JSON.stringify(a.snapshotForSave()));
    a.mergeFrom(saved); // reading back our own file
    fill(a, T0 + 2 * 60_000 + 2000, 0.5, () => 50);
    a.add(sample(T0 + 3 * 60_000 + 1000, 50)); // close minute 2
    const b = JSON.parse(JSON.stringify(a.snapshotForSave()));
    const m2 = b.minutes.find((m: { t: number }) => m.t === T0 + 2 * 60_000);
    expect(m2.n).toBe(4); // 1 + 3 samples, not 1 + 1 + 3

    const other = new HistoryStore(7);
    fill(other, T0 - 10 * 60_000, 1, () => 90);
    a.mergeFrom(JSON.parse(JSON.stringify(other.snapshotForSave())));
    expect(a.stats('1h', 60, 'G', T0 + 4 * 60_000).utilPeak).toBe(90);
  });

  it('a clear in one window wins over older data from another', () => {
    const a = new HistoryStore(7);
    fill(a, T0, 3, () => 50);
    const staleOther = JSON.parse(JSON.stringify(a.snapshotForSave()));
    const b = new HistoryStore(7);
    b.load(staleOther);
    b.clear(T0 + 3 * 60_000);
    a.mergeFrom(JSON.parse(JSON.stringify(b.snapshotForSave())));
    expect(a.stats('1h', 60, 'G', T0 + 4 * 60_000).coveredMinutes).toBe(0);
  });

  it('ignores malformed buckets on load', () => {
    const h = new HistoryStore(7);
    h.load({ version: 1, minutes: [{ t: T0 }, null, { t: T0 + 60_000, n: 1, cpu: 1, mem: 1, gpus: {} }] });
    // The valid bucket counts as a recorded minute overall, but not for a GPU it has no data for.
    expect(h.stats('1h', 60, undefined, T0 + 2 * 60_000).coveredMinutes).toBe(1);
    expect(h.stats('1h', 60, 'G', T0 + 2 * 60_000).coveredMinutes).toBe(0);
  });

  it('counts coverage per GPU (a GPU back from a reset does not borrow its siblings\' minutes)', () => {
    const h = new HistoryStore(7);
    const two = (t: number, withB: boolean): Sample => ({
      t,
      gpus: withB ? [gpu(50), { ...gpu(10), uuid: 'B', index: 1 }] : [gpu(50)],
      cpu: { util: 1, perCore: [1] },
      mem: { usedBytes: 1, totalBytes: 2 },
    });
    for (let m = 0; m < 10; m++) h.add(two(T0 + m * 60_000, m === 9)); // B only in the last minute
    h.add(two(T0 + 10 * 60_000 + 1000, false)); // close minute 9
    expect(h.stats('10m', 10, 'G', T0 + 9 * 60_000).coveredMinutes).toBe(10);
    expect(h.stats('10m', 10, 'B', T0 + 9 * 60_000).coveredMinutes).toBe(1);
  });

  it('builds a heatmap with nulls for unrecorded minutes', () => {
    const h = new HistoryStore(7);
    fill(h, T0, 2, () => 50);
    const hm = h.heatmap('G', 1, T0 + 59 * 60_000);
    expect(hm.cells).toHaveLength(60);
    expect(hm.cells[0]).toBe(50);
    expect(hm.cells[1]).toBe(50);
    expect(hm.cells[2]).toBeNull();
  });

  it('exports CSV with one row per GPU minute', () => {
    const h = new HistoryStore(7);
    fill(h, T0, 2, () => 50);
    const lines = h.toCsv().trim().split('\n');
    expect(lines).toHaveLength(3);
    expect(lines[1]).toContain('2026-10-21T10:00:00.000Z,G,6,50');
  });
});

describe('insights', () => {
  it('grades mean utilization', () => {
    expect(grade(null)).toBeNull();
    expect(grade(5)?.level).toBe('idle');
    expect(grade(35)?.level).toBe('low');
    expect(grade(70)?.level).toBe('good');
    expect(grade(95)?.level).toBe('max');
  });

  it('reports held memory and idle GPU over the last hour', () => {
    const h = new HistoryStore(7);
    fill(h, T0, 30, () => 1, 8000);
    const now = T0 + 30 * 60_000 - 1;
    const titles = buildInsights(h.stats('1h', 60, 'G', now), h.stats('30m', 30, 'G', now), gpu(1, 8000), 80).map((i) => i.title);
    expect(titles).toContain('Memory held while GPU idles');
    expect(titles).toContain('GPU mostly idle');
  });

  it('warns on heat and near-full VRAM', () => {
    const h = new HistoryStore(7);
    const hot = { ...gpu(90, 9800), tempC: 85 };
    const ins = buildInsights(h.stats('1h', 60, 'G'), h.stats('30m', 30, 'G'), hot, 80);
    expect(ins.map((i) => i.level)).toEqual(expect.arrayContaining(['critical', 'serious']));
  });
});
