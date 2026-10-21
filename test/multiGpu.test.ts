import { describe, expect, it } from 'vitest';
import { fakeGpus } from '../src/collectors/fake';
import { HistoryStore } from '../src/history';
import { buildFleetInsights, bySeverity, type FleetEntry } from '../src/insights';
import { ANALYSIS_EVERY_MS, compactSample, fleetSummary, TickComposer } from '../src/payload';
import { samplingPlan } from '../src/rate';
import { statusView } from '../src/statusText';
import { gpu, sampleOf } from './helpers';

const T0 = Date.UTC(2026, 9, 21, 10, 0, 0);

const entry = (index: number, util: number | null, vram: number | null = 50, extra: Partial<FleetEntry> = {}): FleetEntry => ({
  uuid: `GPU-${index}`, index, util, vram, covered: 10, tempC: 60, ...extra,
});

describe('status bar text', () => {
  it('keeps the single-GPU format', () => {
    const v = statusView(sampleOf(T0, [gpu(0, 87, { memUsedMiB: 11_720, memTotalMiB: 16_311, tempC: 66 })]), 80);
    expect(v.text).toBe('$(pulse) GPU 87% · 11.4/16 GB · 66°C');
    expect(v.warn).toBe(false);
    expect(v.tooltip).toContain('NVIDIA GeForce RTX 4090 — 87% util');
  });

  it('aggregates several GPUs and tabulates them in the tooltip', () => {
    const s = sampleOf(T0, [gpu(0, 98, { tempC: 71 }), gpu(1, 63), gpu(2, 40), gpu(3, null, { tempC: null })]);
    const v = statusView(s, 80);
    expect(v.text).toBe('$(pulse) GPU ×4 avg 67% · max 98% · hottest 71°C');
    expect(v.tooltip).toContain('| GPU | Util | VRAM | Temp | Power |');
    expect(v.tooltip).toContain('| 0 · RTX 4090 | 98% | 12.0/24 GB | 71 °C | 300 W |');
    expect(v.tooltip).toContain('| 3 · RTX 4090 | –% |');
    expect(v.tooltip.split('\n').filter((l) => l.startsWith('| ') && /^\| \d/.test(l))).toHaveLength(4);
  });

  it('warns when any GPU, not just the first, is hot', () => {
    const v = statusView(sampleOf(T0, [gpu(0, 50), gpu(1, 50, { tempC: 88 })]), 80);
    expect(v.warn).toBe(true);
    expect(v.tooltip).toContain('$(flame)');
  });

  it('falls back to CPU without GPUs', () => {
    expect(statusView(sampleOf(T0, [], 33), 80).text).toBe('$(pulse) CPU 33%');
  });
});

describe('per-GPU history', () => {
  function twoGpus(): HistoryStore {
    const h = new HistoryStore(7);
    for (let i = 0; i < 60; i++) h.add(sampleOf(T0 + i * 10_000, [gpu(0, 90), gpu(1, 10, { memUsedMiB: 2457 })]));
    return h;
  }

  it('keeps stats, heatmap, live points and CSV separate per GPU', () => {
    const h = twoGpus();
    const now = T0 + 10 * 60_000 - 1;
    expect(h.stats('w', 10, 'GPU-0', now).util).toBeCloseTo(90);
    expect(h.stats('w', 10, 'GPU-1', now).util).toBeCloseTo(10);
    const hm = h.heatmap('GPU-1', 1, now); // 60 cells ending at minute T0+9, so T0 is cell 50
    expect(hm.cells[50]).toBe(10);
    expect(h.heatmap('GPU-0', 1, now).cells[50]).toBe(90);
    expect(h.recentPoints('GPU-1', 0)[0].slice(1, 3)).toEqual([10, 10]);
    expect(h.recentPoints('GPU-0', 0)[0].slice(1, 3)).toEqual([90, 50]);
    const csv = h.toCsv().trim().split('\n');
    expect(csv.filter((l) => l.includes(',GPU-0,'))).toHaveLength(10);
    expect(csv.filter((l) => l.includes(',GPU-1,'))).toHaveLength(10);
  });

  it('summarises every GPU for the strip and fleet insights', () => {
    const h = twoGpus();
    const s = sampleOf(T0 + 10 * 60_000 - 1, [gpu(0, 90), gpu(1, 10)]);
    const f = fleetSummary(h, s, s.t);
    expect(f.map((x) => [x.index, Math.round(x.util ?? -1)])).toEqual([[0, 90], [1, 10]]);
  });
});

describe('fleet insights', () => {
  const titles = (f: FleetEntry[], sel = 'GPU-0') => buildFleetInsights(f, sel, 80).map((i) => i.title);

  it('flags one GPU far below busy peers, with honest wording', () => {
    const ins = buildFleetInsights([entry(0, 92), entry(1, 88), entry(2, 90), entry(3, 21)], 'GPU-0', 80);
    const lag = ins.find((i) => i.title === 'GPU 3 well below the others');
    expect(lag?.level).toBe('warning');
    expect(lag?.detail).toContain('GPU 3 21% vs 90% mean');
    expect(lag?.detail).toContain('If they run separate jobs, it is expected.');
  });

  it('names several laggards', () => {
    expect(titles([entry(0, 95), entry(1, 91), entry(2, 5), entry(3, 4)])).toContain('GPUs 2 and 3 well below the others');
  });

  it('stays quiet when the busy GPU is the odd one out or data is thin', () => {
    expect(titles([entry(0, 95), entry(1, 2), entry(2, 3), entry(3, 1)])).toEqual([]); // one job on one GPU
    expect(titles([entry(0, 95), entry(1, 5, 50, { covered: 1 })])).toEqual([]); // not enough minutes
    expect(titles([entry(0, 60), entry(1, 45)])).toEqual([]); // mild difference
    expect(buildFleetInsights([entry(0, 90)], 'GPU-0', 80)).toEqual([]); // single GPU
  });

  it('reports balanced, busy GPUs as good', () => {
    expect(titles([entry(0, 85), entry(1, 80)])).toEqual(['All 2 GPUs kept busy']);
  });

  it('notes uneven memory only when the GPUs are in use', () => {
    expect(titles([entry(0, 80, 92), entry(1, 78, 40)])).toContain('Uneven memory across GPUs');
    expect(titles([entry(0, 2, 92), entry(1, 1, 40)])).not.toContain('Uneven memory across GPUs');
  });

  it('warns about heat on a GPU that is not selected', () => {
    const ins = buildFleetInsights([entry(0, 50), entry(1, 50, 50, { tempC: 86 })], 'GPU-0', 80);
    expect(ins[0]).toMatchObject({ level: 'critical', title: 'GPU 1 running hot: 86 °C' });
    expect(buildFleetInsights([entry(0, 50, 50, { tempC: 86 }), entry(1, 50)], 'GPU-0', 80).some((i) => i.level === 'critical')).toBe(false);
  });

  it('orders by severity, stably', () => {
    const out = bySeverity([
      { level: 'info', title: 'a', detail: '' }, { level: 'critical', title: 'b', detail: '' },
      { level: 'good', title: 'c', detail: '' }, { level: 'info', title: 'd', detail: '' },
    ]);
    expect(out.map((x) => x.title)).toEqual(['b', 'a', 'd', 'c']);
  });
});

describe('TickComposer', () => {
  it('sends the live part every tick and analysis at most every 2 s, only when changed', () => {
    const h = new HistoryStore(7);
    const c = new TickComposer(h, () => 80);
    const msgs = [];
    for (let i = 0; i < 20; i++) {
      const s = sampleOf(T0 + i * 250, [gpu(0, 50), gpu(1, 50)]);
      h.add(s);
      msgs.push(c.tick(s, 'GPU-1'));
    }
    expect(msgs.every((m) => m.selected === 'GPU-1' && m.point[1] === 50)).toBe(true);
    const withAnalysis = msgs.filter((m) => m.analysis).length;
    expect(withAnalysis).toBeGreaterThanOrEqual(1);
    expect(withAnalysis).toBeLessThanOrEqual(Math.ceil((20 * 250) / ANALYSIS_EVERY_MS));
    expect(msgs[0].analysis?.fleet.map((f) => f.uuid)).toEqual(['GPU-0', 'GPU-1']);
  });

  it('does not resend analysis that was just sent with init', () => {
    const h = new HistoryStore(7);
    const c = new TickComposer(h, () => 80);
    const s = sampleOf(T0, [gpu(0, 50)]);
    h.add(s);
    const first = c.tick(s, undefined);
    c.sent(first.analysis!, T0 - ANALYSIS_EVERY_MS); // as if init went out earlier with identical content
    expect(c.tick(sampleOf(T0 + 10, [gpu(0, 50)]), undefined).analysis).toBeUndefined();
  });

  it('rounds display-only precision', () => {
    const s = compactSample({ t: 1, gpus: [gpu(0, 5, { powerW: 18.3456, memUsedMiB: 1000.7 })], cpu: { util: 12.345, perCore: [1.4, 99.6] }, mem: { usedBytes: 1, totalBytes: 2 } });
    expect(s.gpus[0].powerW).toBe(18.3);
    expect(s.gpus[0].memUsedMiB).toBe(1001);
    expect(s.cpu).toEqual({ util: 12.3, perCore: [1, 100] });
  });
});

describe('sampling plan', () => {
  const base = { dashboardVisible: false, sidebarVisible: false, background: true, statusBar: true, fastMs: 500, slowMs: 5000 };

  it('samples fast once when the dashboard, the side bar or both are visible', () => {
    for (const [d, sb] of [[true, false], [false, true], [true, true]]) {
      expect(samplingPlan({ ...base, dashboardVisible: d, sidebarVisible: sb })).toMatchObject({ run: true, intervalMs: 500 });
    }
  });

  it('only the dashboard polls processes', () => {
    expect(samplingPlan({ ...base, sidebarVisible: true }).processes).toBe(false);
    expect(samplingPlan({ ...base, dashboardVisible: true }).processes).toBe(true);
  });

  it('records slowly in the background, or pauses when nothing needs samples', () => {
    expect(samplingPlan(base)).toEqual({ run: true, intervalMs: 5000, processes: false });
    expect(samplingPlan({ ...base, background: false, statusBar: false }).run).toBe(false);
    expect(samplingPlan({ ...base, background: false, statusBar: false, sidebarVisible: true }).run).toBe(true);
  });
});

describe('fakeGpus', () => {
  it('is deterministic and makes the last GPU lag', () => {
    const a = fakeGpus(4, T0), b = fakeGpus(4, T0);
    expect(a).toEqual(b);
    expect(a.map((g) => g.uuid)).toEqual(['GPU-fake-0', 'GPU-fake-1', 'GPU-fake-2', 'GPU-fake-3']);
    expect(a[3].util!).toBeLessThan(40);
    expect(Math.min(...a.slice(0, 3).map((g) => g.util!))).toBeGreaterThan(60);
  });
});

describe('fleet insight wording (review fixes)', () => {
  const entry = (index: number, util: number, vram: number, covered = 10): FleetEntry => ({
    uuid: `G${index}`, index, util, vram, covered, tempC: 60,
  });

  it('only mentions rank 0 when GPU 0 is the one holding extra memory', () => {
    const notZero = buildFleetInsights([entry(0, 80, 30), entry(1, 80, 35), entry(2, 80, 90)], 'G0', 80);
    const mem = notZero.find((i) => i.title === 'Uneven memory across GPUs');
    expect(mem?.detail).toContain('GPU 2 averaged 90%');
    expect(mem?.detail).not.toContain('rank 0');
    const zero = buildFleetInsights([entry(0, 80, 90), entry(1, 80, 35)], 'G0', 80);
    expect(zero.find((i) => i.title === 'Uneven memory across GPUs')?.detail).toContain('rank 0');
  });

  it('states the minutes actually recorded, not the nominal 10', () => {
    const ins = buildFleetInsights([entry(0, 90, 50, 3), entry(1, 85, 50, 4)], 'G0', 80);
    expect(ins.find((i) => i.title.startsWith('All 2 GPUs'))?.detail).toMatch(/^Last 3 recorded min/);
  });
});
