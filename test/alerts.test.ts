import { describe, expect, it } from 'vitest';
import { AlertEngine, COOLDOWN_MS, type Alert, type AlertConfig } from '../src/alerts';
import { HistoryStore } from '../src/history';
import { alertViews, initPayload } from '../src/payload';
import type { GpuSample } from '../src/types';
import { alertLines, statusView } from '../src/statusText';
import { gpu, sampleOf } from './helpers';

const T0 = Date.UTC(2026, 9, 21, 10, 0, 0);
const S = 1000;
const MIN = 60_000;

const config = (over: Partial<AlertConfig> = {}): AlertConfig => ({
  enabled: true, hot: true, vramFull: true, idleHeld: true, jobFinished: true,
  tempWarn: 80, idleHeldMinutes: 15, jobFinishedMinutes: 10, intervalMs: 500, ...over,
});

/** Feeds samples every `step` ms in [from, to) (relative to T0); returns every alert that fired. */
function run(e: AlertEngine, from: number, to: number, step: number, gpus: (t: number) => GpuSample[]): Alert[] {
  const out: Alert[] = [];
  for (let t = from; t < to; t += step) out.push(...e.update(sampleOf(T0 + t, gpus(t))));
  return out;
}

/** Relative time (ms after T0) of the first alert, found by stepping one sample at a time. */
function firstFireAt(e: AlertEngine, from: number, to: number, step: number, gpus: (t: number) => GpuSample[]): number | null {
  for (let t = from; t < to; t += step) if (e.update(sampleOf(T0 + t, gpus(t))).length) return t;
  return null;
}

const hot = (temp: number | null, i = 0) => [gpu(i, 50, { tempC: temp })];
const MIB_24G = 24_576;
const vram = (usedGb: number, i = 0, util: number | null = 50) => [gpu(i, util, { memUsedMiB: usedGb * 1024, memTotalMiB: MIB_24G })];

describe('alerts: hot', () => {
  it('fires once after 30 s at or above the threshold', () => {
    const e = new AlertEngine(config());
    expect(firstFireAt(e, 0, 60 * S, 500, () => hot(85))).toBe(30 * S);
    const e2 = new AlertEngine(config());
    const fired = run(e2, 0, 10 * MIN, 500, () => hot(85));
    expect(fired).toHaveLength(1);
    expect(fired[0]).toMatchObject({ rule: 'hot', gpuUuid: 'GPU-0', gpuIndex: 0, level: 'warning', title: 'GPU 0 running hot: 85 °C' });
    expect(e2.active().map((a) => a.rule)).toEqual(['hot']);
  });

  it('needs the temperature continuously: a dip below the threshold restarts the timer', () => {
    const e = new AlertEngine(config());
    const fired = run(e, 0, 2 * MIN, 500, (t) => hot(t % (20 * S) < 19 * S ? 85 : 79));
    expect(fired).toHaveLength(0);
  });

  it('does not re-arm until the temperature drops 3 °C below the threshold', () => {
    const e = new AlertEngine(config());
    expect(run(e, 0, 40 * S, 500, () => hot(85))).toHaveLength(1);
    // 78 °C is below 80 but not below 77: the alert stays latched, so heating up again is not a new alert.
    run(e, 40 * S, 60 * S, 500, () => hot(78));
    expect(e.active()).toHaveLength(1);
    expect(run(e, 60 * S, 40 * MIN, 500, () => hot(85))).toHaveLength(0);
    // Clears at 76 °C; after the cooldown a new hot stretch fires again.
    run(e, 40 * MIN, 41 * MIN, 500, () => hot(76));
    expect(e.active()).toHaveLength(0);
    expect(firstFireAt(e, 41 * MIN, 43 * MIN, 500, () => hot(85))).toBe(41 * MIN + 30 * S);
  });

  it('applies a 30 min cooldown per rule and GPU even after the condition cleared', () => {
    const e = new AlertEngine(config());
    expect(run(e, 0, 40 * S, 500, () => hot(85))).toHaveLength(1);
    run(e, 40 * S, 60 * S, 500, () => hot(70));
    // Hot again 1 min later: latched (active) but not shown again inside the cooldown.
    expect(run(e, 60 * S, 5 * MIN, 500, () => hot(90))).toHaveLength(0);
    expect(e.active()).toHaveLength(1);
    // The suppressed episode must clear before the next one; past the cooldown (30 min after 0:30) it fires.
    run(e, 5 * MIN, 6 * MIN, 500, () => hot(70));
    expect(firstFireAt(e, COOLDOWN_MS, COOLDOWN_MS + MIN, 500, () => hot(90))).toBe(COOLDOWN_MS + 30 * S);
  });

  it('cooldown expires exactly 30 min after the last fire; an episode completed inside it stays suppressed', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    expect(firstFireAt(e, 0, MIN, 5 * S, () => hot(85))).toBe(30 * S);
    run(e, 35 * S, 60 * S, 5 * S, () => hot(70));
    expect(firstFireAt(e, 60 * S, 35 * MIN, 5 * S, (t) => hot(t < 30 * MIN ? 70 : 85))).toBe(30 * S + COOLDOWN_MS);

    // Same, but the hot spell starts 1 min earlier: it completes at 29:30 (inside the cooldown) and is
    // latched without a notification; staying hot past 30:30 does not produce a late alert.
    const e2 = new AlertEngine(config({ intervalMs: 5000 }));
    run(e2, 0, 35 * S, 5 * S, () => hot(85));
    run(e2, 35 * S, 60 * S, 5 * S, () => hot(70));
    expect(run(e2, 60 * S, 40 * MIN, 5 * S, (t) => hot(t < 29 * MIN ? 70 : 85))).toHaveLength(0);
    expect(e2.active()).toHaveLength(1);
  });

  it('a sample gap longer than 3x the interval resets the timer (no firing on stale data)', () => {
    const e = new AlertEngine(config());
    run(e, 0, 20 * S, 500, () => hot(85));
    // Next sample 2 s after the last one (> 3 × 500 ms): the 20 s already seen no longer count.
    expect(firstFireAt(e, 21.5 * S, 2 * MIN, 500, () => hot(85))).toBe(51.5 * S);
  });

  it('works at the background interval (5 s)', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    expect(firstFireAt(e, 0, 2 * MIN, 5 * S, () => hot(81))).toBe(30 * S);
  });

  it('null temperature never satisfies the condition and breaks continuity', () => {
    const e = new AlertEngine(config());
    expect(run(e, 0, 5 * MIN, 500, () => hot(null))).toHaveLength(0);
    expect(run(e, 5 * MIN, 10 * MIN, 500, (t) => hot(t % (20 * S) === 0 ? null : 90))).toHaveLength(0);
  });

  it('uses the configured threshold and follows live config changes', () => {
    const e = new AlertEngine(config({ tempWarn: 90 }));
    expect(run(e, 0, 2 * MIN, 500, () => hot(85))).toHaveLength(0);
    e.setConfig(config({ tempWarn: 84 }), T0 + 2 * MIN);
    expect(firstFireAt(e, 2 * MIN, 4 * MIN, 500, () => hot(85))).toBe(2 * MIN + 30 * S);
  });
});

describe('alerts: vramFull', () => {
  it('fires after 20 s at >= 95% with used/total in the title', () => {
    const e = new AlertEngine(config());
    expect(firstFireAt(e, 0, MIN, 500, () => vram(23.5, 1))).toBe(20 * S);
    const e2 = new AlertEngine(config());
    const fired = run(e2, 0, 5 * MIN, 500, () => vram(23.5, 1));
    expect(fired).toHaveLength(1);
    expect(fired[0]).toMatchObject({ rule: 'vramFull', gpuIndex: 1, level: 'warning', title: 'GPU 1 VRAM nearly full (23.5/24 GB)' });
  });

  it('stays latched between 90% and 95% and re-arms below 90%', () => {
    const e = new AlertEngine(config());
    expect(run(e, 0, MIN, 500, () => vram(23.5))).toHaveLength(1);
    run(e, MIN, 2 * MIN, 500, () => vram(22.5)); // 93.75%
    expect(e.active()).toHaveLength(1);
    run(e, 2 * MIN, 3 * MIN, 500, () => vram(21)); // 87.5%
    expect(e.active()).toHaveLength(0);
    // Re-armed, but within the cooldown.
    expect(run(e, 3 * MIN, 4 * MIN, 500, () => vram(23.5))).toHaveLength(0);
  });

  it('ignores missing or zero memory totals', () => {
    const e = new AlertEngine(config());
    const fired = run(e, 0, 2 * MIN, 500, (t) => [gpu(0, 50, { memUsedMiB: t % 2 ? null : 24_000, memTotalMiB: t % 1000 ? 0 : null })]);
    expect(fired).toHaveLength(0);
  });
});

describe('alerts: idleHeld', () => {
  const idle = (util: number | null, usedGb = 12, i = 0) => vram(usedGb, i, util);

  it('fires after the configured minutes of low utilization with VRAM held', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    expect(firstFireAt(e, 0, 20 * MIN, 5 * S, () => idle(2))).toBe(15 * MIN);
    const e2 = new AlertEngine(config({ intervalMs: 5000 }));
    const fired = run(e2, 0, 60 * MIN, 5 * S, () => idle(2));
    expect(fired).toHaveLength(1);
    expect(fired[0]).toMatchObject({ rule: 'idleHeld', level: 'info', title: 'GPU 0 idle for 15 min while holding 12.0 GB of VRAM' });
    expect(fired[0].detail).toMatch(/notebook/);
  });

  it('honours gpuPulse.alerts.idleHeldMinutes', () => {
    const e = new AlertEngine(config({ intervalMs: 5000, idleHeldMinutes: 2 }));
    expect(firstFireAt(e, 0, 5 * MIN, 5 * S, () => idle(0))).toBe(2 * MIN);
  });

  it('does not fire when little VRAM is held or utilization is not low', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    expect(run(e, 0, 30 * MIN, 5 * S, () => idle(2, 11))).toHaveLength(0); // 45.8% VRAM
    expect(run(e, 30 * MIN, 60 * MIN, 5 * S, () => idle(12))).toHaveLength(0);
  });

  it('clears on util >= 15% or VRAM < 40%, not in between', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    expect(run(e, 0, 16 * MIN, 5 * S, () => idle(2))).toHaveLength(1);
    run(e, 16 * MIN, 17 * MIN, 5 * S, () => idle(12)); // neither idle nor cleared
    expect(e.active()).toHaveLength(1);
    run(e, 17 * MIN, 18 * MIN, 5 * S, () => idle(2, 9)); // 37.5% VRAM: cleared
    expect(e.active()).toHaveLength(0);
  });

  it('null utilization never counts as idle', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    expect(run(e, 0, 60 * MIN, 5 * S, () => idle(null))).toHaveLength(0);
  });
});

describe('alerts: jobFinished', () => {
  const util = (u: number | null, i = 0) => [gpu(i, u, { memUsedMiB: 2048 })]; // little VRAM: idleHeld stays quiet

  it('fires once when a long busy stretch ends with 60 s of low utilization', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    expect(run(e, 0, 12 * MIN, 5 * S, () => util(80))).toHaveLength(0);
    expect(firstFireAt(e, 12 * MIN, 15 * MIN, 5 * S, () => util(2))).toBe(13 * MIN);
    const e2 = new AlertEngine(config({ intervalMs: 5000 }));
    run(e2, 0, 12 * MIN, 5 * S, () => util(80));
    const fired = run(e2, 12 * MIN, 60 * MIN, 5 * S, () => util(2));
    expect(fired).toHaveLength(1);
    // Last busy sample at 11:55 after starting at 0:00 → 12 min rounded.
    expect(fired[0]).toMatchObject({ rule: 'jobFinished', level: 'info', title: 'Job on GPU 0 finished after 12 min (avg 80%)' });
    expect(e2.active()).toHaveLength(0); // an event, not a standing condition
  });

  it('ignores stretches shorter than gpuPulse.alerts.jobFinishedMinutes', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    run(e, 0, 9 * MIN, 5 * S, () => util(90));
    expect(run(e, 9 * MIN, 20 * MIN, 5 * S, () => util(0))).toHaveLength(0);
    const e2 = new AlertEngine(config({ intervalMs: 5000, jobFinishedMinutes: 5 }));
    run(e2, 0, 6 * MIN, 5 * S, () => util(90));
    expect(run(e2, 6 * MIN, 20 * MIN, 5 * S, () => util(0))).toHaveLength(1);
  });

  it('bridges dips shorter than 60 s and reports a time-weighted average', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    // 6 min at 100%, a 30 s dip to 0%, 6 min at 50%: one 12.5 min stretch.
    const u = (t: number) => (t < 6 * MIN ? 100 : t < 6.5 * MIN ? 0 : t < 12.5 * MIN ? 50 : 0);
    const fired = run(e, 0, 30 * MIN, 5 * S, (t) => util(u(t)));
    expect(fired).toHaveLength(1);
    expect(fired[0].title).toMatch(/^Job on GPU 0 finished after 12 min \(avg 7\d%\)$/);
  });

  it('a 60 s+ low phase in the middle splits the job into two short stretches', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    const u = (t: number) => (t < 6 * MIN ? 100 : t < 8 * MIN ? 0 : t < 14 * MIN ? 100 : 0);
    expect(run(e, 0, 30 * MIN, 5 * S, (t) => util(u(t)))).toHaveLength(0);
  });

  it('a stretch that idles at 10-15% for 60 s without going quiet is dropped, not reported later', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    const u = (t: number) => (t < 12 * MIN ? 90 : t < 14 * MIN ? 12 : 0);
    expect(run(e, 0, 30 * MIN, 5 * S, (t) => util(u(t)))).toHaveLength(0);
  });

  it('a short sample gap (< 60 s) keeps the busy stretch; the quiet timer restarts', () => {
    const e = new AlertEngine(config());
    run(e, 0, 12 * MIN, 500, () => util(90));
    // 20 s without samples (> 3 × 500 ms), then quiet: the job still finishes 60 s after quiet resumes.
    expect(firstFireAt(e, 12 * MIN + 20 * S, 15 * MIN, 500, () => util(2))).toBe(13 * MIN + 20 * S);
  });

  it('a sample gap during the job resets the stretch', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    run(e, 0, 8 * MIN, 5 * S, () => util(90));
    // Machine slept 10 min: the stretch restarts and is too short when it ends.
    run(e, 18 * MIN, 22 * MIN, 5 * S, () => util(90));
    expect(run(e, 22 * MIN, 30 * MIN, 5 * S, () => util(1))).toHaveLength(0);
  });
});

describe('alerts: common behaviour', () => {
  it('keeps GPUs independent (own timers, latches and cooldowns)', () => {
    const e = new AlertEngine(config());
    const both = (t: number) => [gpu(0, 50, { tempC: 85 }), gpu(1, 50, { tempC: t >= 60 * S ? 88 : 60 })];
    const fired = run(e, 0, 3 * MIN, 500, both);
    expect(fired.map((a) => [a.gpuIndex, a.rule])).toEqual([[0, 'hot'], [1, 'hot']]);
    expect(e.active().map((a) => a.gpuUuid).sort()).toEqual(['GPU-0', 'GPU-1']);
  });

  it('a GPU missing from samples is not evaluated, and returns with fresh timers', () => {
    const e = new AlertEngine(config());
    run(e, 0, 25 * S, 500, () => [gpu(0, 50, { tempC: 85 }), gpu(1, 50, { tempC: 85 })]);
    // GPU 1 drops out (failed read) for 10 s; GPU 0 keeps reporting and fires.
    const fired = run(e, 25 * S, 35 * S, 500, () => [gpu(0, 50, { tempC: 85 })]);
    expect(fired.map((a) => a.gpuIndex)).toEqual([0]);
    expect(firstFireAt(e, 35 * S, 2 * MIN, 500, () => [gpu(0, 50, { tempC: 85 }), gpu(1, 50, { tempC: 85 })])).toBe(65 * S);
  });

  it('all alerts off: nothing fires and no state is kept', () => {
    const e = new AlertEngine(config({ enabled: false }));
    expect(run(e, 0, 2 * MIN, 500, () => hot(95))).toHaveLength(0);
    const e2 = new AlertEngine(config());
    run(e2, 0, MIN, 500, () => hot(95));
    expect(e2.active()).toHaveLength(1);
    e2.setConfig(config({ enabled: false }), T0 + MIN);
    expect(e2.active()).toHaveLength(0);
  });

  it('a disabled rule drops its state and stops firing; others keep working', () => {
    const e = new AlertEngine(config());
    run(e, 0, MIN, 500, () => [gpu(0, 50, { tempC: 95, memUsedMiB: 24_000, memTotalMiB: MIB_24G })]);
    expect(e.active().map((a) => a.rule).sort()).toEqual(['hot', 'vramFull']);
    e.setConfig(config({ hot: false }), T0 + MIN);
    expect(e.active().map((a) => a.rule)).toEqual(['vramFull']);
    expect(run(e, MIN, 40 * MIN, 500, () => hot(95))).toHaveLength(0);
  });

  it('a slower-to-faster interval change does not reset timers while the sampler switches over', () => {
    const e = new AlertEngine(config({ intervalMs: 5000 }));
    run(e, 0, 20 * S, 5 * S, () => hot(85)); // samples at 0..15 s
    e.setConfig(config({ intervalMs: 500 }), T0 + 16 * S);
    // The first fast sample arrives 5 s after the last slow one (> 3 × 500 ms, inside the grace window).
    expect(firstFireAt(e, 20 * S, MIN, 500, () => hot(85))).toBe(30 * S);
  });

  it('time going backwards resets timers instead of firing', () => {
    const e = new AlertEngine(config());
    run(e, 0, 25 * S, 500, () => hot(85));
    expect(run(e, 10 * S, 30 * S, 500, () => hot(85))).toHaveLength(0);
  });

  it('cost per sample is O(GPUs): a day at 500 ms on 8 GPUs stays fast', () => {
    const e = new AlertEngine(config());
    const gs = Array.from({ length: 8 }, (_, i) => gpu(i, 40, { tempC: 70 }));
    const t0 = performance.now();
    for (let t = 0; t < 24 * 3600 * S; t += 500) e.update(sampleOf(T0 + t, gs));
    expect(performance.now() - t0).toBeLessThan(3000); // 172 800 samples; typically ~100 ms
  });
});

describe('alerts: status bar tooltip', () => {
  const s = sampleOf(T0, [gpu(0, 50)]);

  it('lists active alerts in the tooltip but never changes the text', () => {
    const plain = statusView(s, 80);
    const v = statusView(s, 80, { active: ['GPU 0 running hot: 85 °C', 'GPU 0 VRAM nearly full (23.5/24 GB)'], mutedUntil: 0 });
    expect(v.text).toBe(plain.text);
    expect(v.tooltip).toContain('$(bell) **2 active alerts**');
    expect(v.tooltip).toContain('- GPU 0 running hot: 85 °C\n');
    expect(v.tooltip).not.toContain('muted');
  });

  it('shows the mute state only while it lasts', () => {
    expect(alertLines({ active: [], mutedUntil: T0 + MIN }, T0)).toContain('$(bell-slash) Alerts muted until');
    expect(alertLines({ active: [], mutedUntil: T0 - 1 }, T0)).toBe('');
    expect(alertLines({ active: ['x'], mutedUntil: 0 }, T0)).toContain('**1 active alert**');
  });
});

describe('alerts: view payload', () => {
  it('init carries level/title/detail of the active alerts only', () => {
    const e = new AlertEngine(config());
    run(e, 0, MIN, 500, () => hot(90));
    const views = alertViews(e.active());
    expect(views).toEqual([{ level: 'warning', title: 'GPU 0 running hot: 90 °C', detail: expect.stringContaining('80 °C') }]);
    const init = initPayload({
      history: new HistoryStore(7), sample: undefined, selected: undefined, vendor: 'nvidia', processes: [], specs: undefined,
      tempWarn: 80, intervalMs: 500, alerts: views,
    });
    expect(init.alerts).toBe(views);
  });

  it('revision changes exactly when the active set may have changed', () => {
    const e = new AlertEngine(config());
    const r0 = e.revision;
    run(e, 0, 29 * S, 500, () => hot(90));
    expect(e.revision).toBe(r0);
    run(e, 29 * S, 31 * S, 500, () => hot(90));
    const r1 = e.revision;
    expect(r1).toBeGreaterThan(r0);
    run(e, 31 * S, MIN, 500, () => hot(90));
    expect(e.revision).toBe(r1);
    run(e, MIN, 2 * MIN, 500, () => hot(60));
    expect(e.revision).toBeGreaterThan(r1);
  });
});
