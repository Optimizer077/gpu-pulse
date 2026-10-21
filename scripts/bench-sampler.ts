// Extension-host CPU benchmark for the sampler + history + dashboard tick path, on real hardware.
//   npm run bench -- --mode=stream  [--seconds=20] [--interval=500]   current: nvidia-smi stream + TickComposer
//   npm run bench -- --mode=legacy  [--seconds=20] [--interval=500]   v0.1.0 path: spawn per tick + full analysis per tick
// Host CPU is process.cpuUsage(); nvidia-smi CPU is read from the OS for the stream process.
// Spawn-per-tick children exit too fast to read; their cost is reported as samples x measured per-spawn CPU if given
// (--spawnCpuMs=26.6, measured separately with Start-Process/TotalProcessorTime).
import { HistoryStore, toPoint } from '../src/history';
import { analysis, TickComposer } from '../src/payload';
import { Sampler } from '../src/sampler';
import type { Sample } from '../src/types';
import { childCpuMs } from './childCpu';

const arg = (k: string, d: string) => process.argv.find((a) => a.startsWith(`--${k}=`))?.split('=')[1] ?? d;
const mode = arg('mode', 'stream');
const seconds = Number(arg('seconds', '20'));
const interval = Number(arg('interval', '500'));
const spawnCpuMs = Number(arg('spawnCpuMs', '0'));

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
  const sampler = new Sampler('', interval, (m) => console.log('[log]', m), { stream: mode !== 'legacy' });
  sampler.setProcessPolling(false); // identical in both modes; the process list is a separate 5 s query
  const history = new HistoryStore(7);
  const composer = new TickComposer(history, () => 80);
  let n = 0;
  let bytes = 0;
  sampler.onSample((s: Sample) => {
    n++;
    history.add(s);
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
  console.log(JSON.stringify({
    mode: `${mode} (${sampler.mode})`, intervalMs: interval, wallS: +(wall / 1000).toFixed(1), samples,
    samplesPerSec: +(samples / (wall / 1000)).toFixed(2),
    hostCpuMs: +hostMs.toFixed(0), hostPctOfCore: pct(hostMs), hostMsPerSample: +(hostMs / samples).toFixed(2),
    nvidiaSmiCpuMs: childMs === null ? null : +childMs.toFixed(0), nvidiaSmiPctOfCore: childMs === null ? null : pct(childMs),
    nvidiaSmiSource: child0 !== null ? 'measured (stream process)' : spawnCpuMs ? `estimated: samples x ${spawnCpuMs} ms/spawn` : 'n/a',
    totalPctOfCore: childMs === null ? null : pct(hostMs + childMs),
    avgTickBytes: Math.round((bytes - b0) / Math.max(1, samples)),
  }));
}

void main();
