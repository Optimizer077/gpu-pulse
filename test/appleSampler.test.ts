import { afterEach, describe, expect, it, vi } from 'vitest';
import { AlertEngine, type AlertConfig } from '../src/alerts';
import { IOREG, IOREG_ARGS, parseIoregAccelerators } from '../src/collectors/apple';
import type { ExecFn } from '../src/collectors/exec';
import { HistoryStore } from '../src/history';
import { buildInsights } from '../src/insights';
import { initPayload, TickComposer } from '../src/payload';
import { Sampler } from '../src/sampler';
import { statusView } from '../src/statusText';
import type { GpuSample, Sample, WindowStats } from '../src/types';
import { appleGpu, fixture, gpu, sampleOf } from './helpers';

afterEach(() => vi.useRealTimers());

const T0 = Date.UTC(2026, 9, 26, 10, 0, 0);
const MIN = 60_000;

/** Fake runner: ioreg answers `text` (after `delayMs`), anything else fails like a missing binary. */
function ioregExec(text: string | (() => string), delayMs = 0) {
  const calls: string[] = [];
  let inFlight = 0;
  let maxInFlight = 0;
  const exec: ExecFn = async (cmd, args) => {
    calls.push(`${cmd} ${args.join(' ')}`);
    if (cmd !== IOREG) throw Object.assign(new Error(`spawn ${cmd} ENOENT`), { code: 'ENOENT' });
    inFlight++;
    maxInFlight = Math.max(maxInFlight, inFlight);
    try {
      if (delayMs) await new Promise((r) => setTimeout(r, delayMs));
      return typeof text === 'function' ? text() : text;
    } finally {
      inFlight--;
    }
  };
  return { exec, calls, maxInFlight: () => maxInFlight };
}

describe('Sampler on macOS: backend order', () => {
  it('uses ioreg first on darwin and never looks for nvidia-smi when an accelerator exists', async () => {
    vi.useFakeTimers();
    const { exec, calls } = ioregExec(fixture('ioreg-apple-m1pro.txt'));
    const logs: string[] = [];
    const s = new Sampler('', 1000, (m) => logs.push(m), { platform: 'darwin', exec });
    const nv = vi.spyOn(s.nvidia, 'detect').mockResolvedValue(true);
    const procs = vi.spyOn(s.nvidia, 'processes');
    const samples: Sample[] = [];
    s.onSample((x) => samples.push(x));
    await s.start();
    expect(s.vendor).toBe('apple');
    expect(s.mode).toBe('poll'); // no stream exists for ioreg
    expect(nv).not.toHaveBeenCalled();
    expect(calls[0]).toBe(`${IOREG} ${IOREG_ARGS.join(' ')}`);
    expect(logs).toContain('GPU backend: apple (ioreg once per sample)');
    expect(samples[0].gpus[0]).toMatchObject({ vendor: 'apple', name: 'Apple M1 Pro', util: 6, unifiedMemory: true });
    await vi.advanceTimersByTimeAsync(3000);
    expect(samples.length).toBeGreaterThanOrEqual(4);
    expect(procs).not.toHaveBeenCalled(); // no per-process listing on macOS
    expect(s.processes).toEqual([]);
    s.stop();
  });

  it('falls back to the other backends when ioreg lists no accelerator (VM)', async () => {
    const { exec } = ioregExec('');
    const s = new Sampler('', 1000, () => {}, { platform: 'darwin', exec });
    const nv = vi.spyOn(s.nvidia, 'detect').mockResolvedValue(false);
    await s.start();
    expect(nv).toHaveBeenCalledOnce();
    expect(s.vendor).toBe('none');
    s.stop();
  });

  it('never runs ioreg on other platforms', async () => {
    for (const platform of ['win32', 'linux'] as const) {
      const { exec, calls } = ioregExec(fixture('ioreg-apple-m1pro.txt'));
      const s = new Sampler('', 1000, () => {}, { platform, exec, stream: false });
      vi.spyOn(s.nvidia, 'detect').mockResolvedValue(true);
      vi.spyOn(s.nvidia, 'sample').mockResolvedValue([gpu(0, 5)]);
      s.setProcessPolling(false);
      await s.start();
      expect(s.vendor).toBe('nvidia');
      expect(calls).toEqual([]);
      s.stop();
    }
  });
});

describe('Sampler on macOS: polling', () => {
  it('keeps at most one ioreg run in flight however often now() is called', async () => {
    vi.useFakeTimers();
    const io = ioregExec(fixture('ioreg-apple-m1pro.txt'), 300); // slower than a tick would like
    const s = new Sampler('', 250, () => {}, { platform: 'darwin', exec: io.exec });
    let n = 0;
    s.onSample(() => n++);
    const started = s.start();
    await vi.advanceTimersByTimeAsync(700); // detect + first sample
    await started;
    for (let i = 0; i < 10; i++) void s.now();
    await vi.advanceTimersByTimeAsync(5000);
    s.stop();
    expect(io.maxInFlight()).toBe(1);
    // Each sample costs 300 ms of ioreg + a 250 ms gap: ~9 in 5 s; parallel loops would give far more.
    expect(n).toBeGreaterThanOrEqual(6);
    expect(n).toBeLessThanOrEqual(12);
  });

  it('a failed or timed-out ioreg yields a sample without GPUs, logged once', async () => {
    vi.useFakeTimers();
    let fail = false;
    const io = ioregExec(() => {
      if (fail) throw new Error('Command failed: ioreg (timeout)');
      return fixture('ioreg-apple-m1pro.txt');
    });
    const logs: string[] = [];
    const s = new Sampler('', 1000, (m) => logs.push(m), { platform: 'darwin', exec: io.exec });
    const samples: Sample[] = [];
    s.onSample((x) => samples.push(x));
    await s.start();
    fail = true;
    await vi.advanceTimersByTimeAsync(3000);
    s.stop();
    expect(samples[0].gpus).toHaveLength(1);
    expect(samples[samples.length - 1].gpus).toEqual([]);
    expect(logs.filter((l) => l.startsWith('GPU sample failed'))).toHaveLength(1);
  });

  it('an injected reader can stand in for Apple (benchmark, preview)', async () => {
    const s = new Sampler('', 1000, () => {}, { gpuReader: () => [appleGpu(0, 30)], gpuVendor: 'apple' });
    await s.start();
    expect(s.vendor).toBe('apple');
    s.stop();
  });
});

describe('status bar on Apple Silicon', () => {
  it('shows used unified memory and no temperature segment', () => {
    const v = statusView(sampleOf(T0, [appleGpu(0, 37, { memUsedMiB: 2150 })]), 80);
    expect(v.text).toBe('$(pulse) GPU 37% · 2.1 GB');
    expect(v.warn).toBe(false);
    expect(v.tooltip).toContain('Apple M1 Pro — 37% util, GPU mem 2.1 GB of 32 GB unified\n');
    expect(v.tooltip).not.toMatch(/°C| W\b/);
  });

  it('handles a null temperature on any vendor', () => {
    const v = statusView(sampleOf(T0, [gpu(0, 50, { tempC: null })]), 80);
    expect(v.text).toBe('$(pulse) GPU 50% · 12.0/24 GB');
    expect(v.tooltip).toContain('– °C'); // NVIDIA/AMD keep the field, it is normally reported
    expect(statusView(sampleOf(T0, [appleGpu(0, null, { memUsedMiB: null })]), 80).text).toBe('$(pulse) GPU –%');
  });

  it('tabulates an Intel Mac (iGPU + AMD) with a GPU-memory column', () => {
    const gs = parseIoregAccelerators(fixture('ioreg-intel-amd.txt'), { totalMemBytes: 32 * 2 ** 30 });
    const v = statusView(sampleOf(T0, gs), 80);
    expect(v.text).toBe('$(pulse) GPU ×2 avg 20% · max 37% · hottest 58°C');
    expect(v.tooltip).toContain('| GPU | Util | GPU mem | Temp | Power |');
    expect(v.tooltip).toContain('| 0 · Intel Graphics | 2% | 0.3 GB | – °C | – W |');
    expect(v.tooltip).toContain('| 1 · AMD Radeon GPU | 37% | 1.4/8 GB | 58 °C | 19 W |');
  });

  it('drops temperature and power columns no Apple GPU fills', () => {
    const v = statusView(sampleOf(T0, [appleGpu(0, 10), appleGpu(1, 20, { uuid: 'apple-1-x' })]), 80);
    expect(v.text).toBe('$(pulse) GPU ×2 avg 15% · max 20%');
    expect(v.tooltip).toContain('| GPU | Util | GPU mem |\n|:--|--:|--:|\n| 0 · Apple M1 Pro | 10% | 2.1 GB |');
  });
});

const config = (over: Partial<AlertConfig> = {}): AlertConfig => ({
  enabled: true, hot: true, vramFull: true, idleHeld: true, jobFinished: true,
  tempWarn: 80, idleHeldMinutes: 15, jobFinishedMinutes: 10, intervalMs: 1000, ...over,
});

function feed(e: AlertEngine, minutes: number, g: (t: number) => GpuSample[], from = 0) {
  const out = [];
  for (let t = from; t < from + minutes * MIN; t += 1000) out.push(...e.update(sampleOf(T0 + t, g(t))));
  return out;
}

describe('alerts and insights on unified memory', () => {
  it('hot and power rules never fire without temperature / power readings', () => {
    const e = new AlertEngine(config({ tempWarn: 30 }));
    const fired = feed(e, 60, () => [appleGpu(0, 100)]);
    expect(fired.map((a) => a.rule)).not.toContain('hot');
    expect(e.active()).toEqual([]);
    const hour: WindowStats = {
      label: '1 hour', minutes: 60, coveredMinutes: 60, util: 95, utilPeak: 100, vram: 7, power: 0, tempPeak: 0,
      idleShare: 0, busyShare: 1, heldIdleShare: 0, cpu: 20, mem: 40,
    };
    const titles = buildInsights(hour, hour, appleGpu(0, 95), 30).map((i) => i.title);
    expect(titles).toEqual(['GPU kept busy']); // no "Running hot", "Peaked at", or power-draw notes
  });

  it('vramFull and idleHeld use the unified total as denominator', () => {
    const e = new AlertEngine(config());
    // 2.1 of 32 GB held while idle is 6.6%: far from the 50% idle-held and 95% full thresholds.
    expect(feed(e, 30, () => [appleGpu(0, 1)])).toEqual([]);
    const full = feed(new AlertEngine(config()), 1, () => [appleGpu(0, 50, { memUsedMiB: 31.5 * 1024 })]);
    expect(full.map((a) => a.title)).toEqual(['GPU 0 memory nearly full (31.5/32 GB unified)']);
    const held = feed(new AlertEngine(config()), 16, () => [appleGpu(0, 2, { memUsedMiB: 20 * 1024 })]);
    expect(held.map((a) => a.title)).toEqual(['GPU 0 idle for 15 min while holding 20.0 GB of unified memory']);
  });

  it('history records GPU-allocated share of unified memory', () => {
    const h = new HistoryStore(7);
    for (let t = 0; t < 2 * MIN; t += 1000) h.add(sampleOf(T0 + t, [appleGpu(0, 2, { memUsedMiB: 16_384 })]));
    const w = h.stats('5 min', 5, 'apple-0-apple-m1-pro', T0 + 2 * MIN);
    expect(w.vram).toBeCloseTo(50, 5);
    expect(w.heldIdleShare).toBe(1);
    expect(w.tempPeak).toBe(0);
    expect(w.power).toBe(0);
    const hour = { ...w, coveredMinutes: 60 }; // the insight needs 5+ recorded minutes
    const ins = buildInsights(hour, hour, appleGpu(0, 2, { memUsedMiB: 16_384 }), 80).find((i) => i.title === 'Memory held while GPU idles');
    expect(ins?.detail).toContain('of unified memory was allocated by the GPU');
  });
});

describe('view payload on macOS', () => {
  it('carries the backend vendor and the per-GPU unified-memory flag the views label by', () => {
    const s = sampleOf(T0, [appleGpu(0, 37, { allocMiB: 4096.7 })]);
    const init = initPayload({
      history: new HistoryStore(7), sample: s, selected: undefined, vendor: 'apple', processes: [], specs: undefined,
      tempWarn: 80, intervalMs: 500, alerts: [],
    });
    expect(init.vendor).toBe('apple');
    expect(init.processes).toEqual([]);
    expect(init.sample?.gpus[0]).toMatchObject({ unifiedMemory: true, rendererUtil: 37, tilerUtil: 37, coreCount: 16, allocMiB: 4097, tempC: null, powerW: null });
    const tick = new TickComposer(new HistoryStore(7), () => 80).tick(s, undefined);
    expect(tick.sample.gpus[0].unifiedMemory).toBe(true);
    expect(tick.point[2]).toBeCloseTo((2150 / 32_768) * 100, 1); // chart "GPU mem" % of unified total
  });
});
