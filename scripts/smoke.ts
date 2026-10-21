// Runs the real collectors outside VS Code: `npm run smoke`.
// Streams for ~5 s at 500 ms and reports samples/sec and CPU cost, then writes the preview fixture.
import * as fs from 'fs';
import * as path from 'path';
import { collectSpecs } from '../src/collectors/specs';
import { HistoryStore } from '../src/history';
import { initPayload, TickComposer } from '../src/payload';
import { Sampler } from '../src/sampler';
import { statusView } from '../src/statusText';
import { childCpuMs } from './childCpu';

const STREAM_MS = 5000;
const INTERVAL_MS = 500;

async function main(): Promise<void> {
  const sampler = new Sampler('', INTERVAL_MS, (m) => console.log('[log]', m));
  const history = new HistoryStore(7);
  const composer = new TickComposer(history, () => 80);
  let n = 0;
  sampler.onSample((s) => {
    history.add(s);
    composer.tick(s, undefined);
    if (++n === 1) console.log('first sample', JSON.stringify(s.gpus, null, 1), 'cpu', s.cpu.util.toFixed(1));
  });

  const t0 = Date.now();
  await sampler.start();
  console.log(`started in ${Date.now() - t0} ms: vendor=${sampler.vendor} mode=${sampler.mode}`);
  const pid = sampler.streamPid;
  const child0 = childCpuMs(pid);
  const cpu0 = process.cpuUsage();
  const n0 = n;
  const w0 = performance.now();
  await new Promise((r) => setTimeout(r, STREAM_MS));
  const wall = performance.now() - w0;
  const cpu = process.cpuUsage(cpu0);
  const child1 = childCpuMs(pid);
  const samples = n - n0;
  const hostMs = (cpu.user + cpu.system) / 1000;
  console.log(
    `stream: ${samples} samples in ${(wall / 1000).toFixed(1)} s = ${(samples / (wall / 1000)).toFixed(2)} samples/s ` +
      `(interval ${INTERVAL_MS} ms) · host CPU ${hostMs.toFixed(0)} ms (${((hostMs / wall) * 100).toFixed(2)}% of a core)` +
      (child0 !== null && child1 !== null
        ? ` · nvidia-smi CPU ${(child1 - child0).toFixed(0)} ms (${(((child1 - child0) / wall) * 100).toFixed(2)}% of a core)`
        : ''),
  );
  if (sampler.latest) console.log('status bar:', statusView(sampler.latest, 80).text);
  sampler.stop();

  const id = sampler.latest?.gpus[0]?.uuid;
  console.log('1h stats', history.stats('1 hour', 60, id));
  console.log('processes', (await sampler.nvidia.processes()).slice(0, 5));
  const t1 = Date.now();
  const specs = await collectSpecs(await sampler.nvidia.staticInfo(), undefined);
  console.log(`specs in ${Date.now() - t1} ms`, JSON.stringify(specs, null, 1));

  // Real-data fixture for scripts/preview.html?real
  const init = initPayload({
    history, sample: sampler.latest, selected: undefined, vendor: sampler.vendor,
    processes: sampler.processes, specs, tempWarn: 80, intervalMs: INTERVAL_MS,
  });
  fs.writeFileSync(path.join(__dirname, 'preview-data.js'), `window.PREVIEW_INIT = ${JSON.stringify(init)};\n`);
  console.log('wrote dist/preview-data.js');
  if (sampler.vendor === 'nvidia' && samples < (STREAM_MS / INTERVAL_MS) * 0.7) {
    console.error(`FAIL: expected ~${STREAM_MS / INTERVAL_MS} samples, got ${samples}`);
    process.exitCode = 1;
  }
}

void main();
