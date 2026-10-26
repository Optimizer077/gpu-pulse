// Extension-host CPU benchmark for the sampler + history + dashboard tick path, on real hardware.
//   npm run bench -- --mode=stream  [--seconds=20] [--interval=500]   current: nvidia-smi stream + TickComposer
//   npm run bench -- --mode=legacy  [--seconds=20] [--interval=500]   v0.1.0 path: spawn per tick + full analysis per tick
// Host CPU is process.cpuUsage(); nvidia-smi CPU is read from the OS for the stream process.
// Spawn-per-tick children exit too fast to read; their cost is reported as samples x measured per-spawn CPU if given
// (--spawnCpuMs=26.6, measured separately with Start-Process/TotalProcessorTime).
// The alert engine runs on every sample as in the extension; --alerts=off leaves it out for comparison.
// macOS:
//   npm run bench -- --mode=apple-fake [--execMs=40] [--fixture=ioreg-apple-m1pro.txt]   any OS: ioreg replaced by a
//     test/fixtures capture returned after --execMs, so host CPU = sampler + ioreg parsing + views; no child process.
//   On a Mac, the default mode polls the real ioreg (no stream exists). ioreg exits too fast to read its CPU here:
//   pass --spawnCpuMs=<user+sys ms per ioreg run, from scripts/mac-probe.sh> to add it as samples x that cost.
import * as fs from 'fs';
import * as path from 'path';
import { AlertEngine } from '../src/alerts';
import { IOREG } from '../src/collectors/apple';
import type { ExecFn } from '../src/collectors/exec';
import { HistoryStore, toPoint } from '../src/history';
import { analysis, TickComposer } from '../src/payload';
import { Sampler, type SamplerOptions } from '../src/sampler';
import type { Sample } from '../src/types';
import { childCpuMs } from './childCpu';

const arg = (k: string, d: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.split('=')[1] ?? d;
const mode = arg('mode', 'stream');
const seconds = Number(arg('seconds', '20'));
const interval = Number(arg('interval', '500'));
const spawnCpuMs = Number(arg('spawnCpuMs', '0'));
const withAlerts = arg('alerts', 'on') !== 'off'; // --alerts=off measures the path without the alert engine
const execMs = Number(arg('execMs', '0'));

/** ioreg stand-in: a fixture capture, answered after `execMs` like a process that takes that long. */
function fakeIoreg(): ExecFn {
  const text = fs.readFileSync(path.join(__dirname, '..', 'test', 'fixtures', arg('fixture', 'ioreg-apple-m1pro.txt')), 'utf8');
  return async (cmd) => {
    if (cmd !== IOREG) throw new Error(`fake exec: ${cmd} not available`);
    if (execMs > 0) await new Promise((r) => setTimeout(r, execMs));
    return text;
  };
}

function samplerOptions(): SamplerOptions {
  if (mode === 'apple-fake') return { platform: 'darwin', exec: fakeIoreg() };
  return { stream: mode !== 'legacy' };
}

/** 24 h of minute buckets for 4 GPUs so window stats do realistic work. */
function preload(h: HistoryStore, ids: string[]): void {
  const start = Date.now() - 24 * 3600e3;
  const data = { version: 1, minutes: [] as unknown[] };
  for (let i = 0; i < 1440; i++) {
    const gpus: Record<string, unknown> = {};
    for (const id of ids) gpus[id] = { n: 60, util: 50, utilMax: 90, vram: 40, power: 100, tempMax: 60, idle: 10, busy: 20, heldIdle: 0 };
    data.minutes.push({ t: Math.floor(start / 60e3) * 60e3 + i * 60e3, n: 60, cpu: 20, mem: 40, gpus });
  }
  h.load(data);
}

async function main(): Promise<void> {
  const sampler = new Sampler('', interval, (m) => console.log('[log]', m), samplerOptions());
  sampler.setProcessPolling(false); // identical in both modes; the process list is a separate 5 s query
  const history = new HistoryStore(7);
  const composer = new TickComposer(history, () => 80);
  const alerts = new AlertEngine({
    enabled: true, hot: true, vramFull: true, idleHeld: true, jobFinished: true,
    tempWarn: 80, idleHeldMinutes: 15, jobFinishedMinutes: 10, intervalMs: interval,
  });
  let alertRev = -1;
  let n = 0;
  let bytes = 0;
  sampler.onSample((s: Sample) => {
    n++;
    history.add(s);
    if (withAlerts) {
      alerts.update(s); // same per-sample work as the extension host
      if (alerts.revision !== alertRev) {
        alertRev = alerts.revision;
        alerts.active();
      }
    }
    const id = s.gpus[0]?.uuid;
    const msg =
      mode === 'legacy'
        ? { type: 'tick', sample: s, point: toPoint(s, id), selected: id, ...analysis(history, s, id, 80) }
        : composer.tick(s, id);
    bytes += JSON.stringify(msg).length; // stand-in for postMessage serialization
  });
  await sampler.start();
  preload(history, [sampler.latest?.gpus[0]?.uuid ?? 'GPU-0', 'FAKE-1', 'FAKE-2', 'FAKE-3']);
  await new Promise((r) => setTimeout(r, 1500)); // let the stream settle
  const pid = sampler.streamPid;
  const child0 = childCpuMs(pid);
  const c0 = process.cpuUsage();
  const t0 = performance.now();
  const n0 = n;
  const b0 = bytes;
  await new Promise((r) => setTimeout(r, seconds * 1000));
  const wall = performance.now() - t0;
  const cpu = process.cpuUsage(c0);
  const child1 = childCpuMs(pid);
  sampler.stop();
  const samples = n - n0;
  const hostMs = (cpu.user + cpu.system) / 1000;
  const childMs = child0 !== null && child1 !== null ? child1 - child0 : spawnCpuMs ? samples * spawnCpuMs : null;
  const pct = (ms: number) => +((ms / wall) * 100).toFixed(2);
  const child = sampler.vendor === 'apple' ? 'ioreg' : 'nvidiaSmi'; // the per-sample child process
  console.log(JSON.stringify({
    mode: `${mode} (${sampler.vendor}, ${sampler.mode})`, alerts: withAlerts, intervalMs: interval, wallS: +(wall / 1000).toFixed(1), samples,
    samplesPerSec: +(samples / (wall / 1000)).toFixed(2),
    hostCpuMs: +hostMs.toFixed(0), hostPctOfCore: pct(hostMs), hostMsPerSample: +(hostMs / samples).toFixed(2),
    [`${child}CpuMs`]: childMs === null ? null : +childMs.toFixed(0), [`${child}PctOfCore`]: childMs === null ? null : pct(childMs),
    [`${child}Source`]: child0 !== null ? 'measured (stream process)' : spawnCpuMs ? `estimated: samples x ${spawnCpuMs} ms/spawn`
      : mode === 'apple-fake' ? 'none (fixture, no process)' : 'n/a',
    totalPctOfCore: childMs === null ? null : pct(hostMs + childMs),
    avgTickBytes: Math.round((bytes - b0) / Math.max(1, samples)),
  }));
}

void main();
