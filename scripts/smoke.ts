// Runs the real collectors outside VS Code: `npm run smoke`.
import * as fs from 'fs';
import * as path from 'path';
import { collectSpecs } from '../src/collectors/specs';
import { HistoryStore } from '../src/history';
import { buildInsights } from '../src/insights';
import { initPayload } from '../src/payload';
import { Sampler } from '../src/sampler';

async function main(): Promise<void> {
  const sampler = new Sampler('', 500, (m) => console.log('[log]', m));
  const history = new HistoryStore(7);
  let n = 0;
  const done = new Promise<void>((resolve) =>
    sampler.onSample((s) => {
      history.add(s);
      if (++n === 1) console.log('first sample', JSON.stringify(s.gpus, null, 1), 'cpu', s.cpu.util.toFixed(1));
      if (n >= 20) resolve();
    }),
  );
  const t0 = Date.now();
  await sampler.start();
  await done;
  sampler.stop();
  console.log(`${n} samples in ${Date.now() - t0} ms, vendor=${sampler.vendor}`);
  const id = sampler.latest?.gpus[0]?.uuid;
  const hour = history.stats('1 hour', 60, id);
  console.log('1h stats', hour);
  console.log('insights', buildInsights(hour, history.stats('30 min', 30, id), sampler.latest?.gpus[0], 80));
  console.log('processes', (await sampler.nvidia.processes()).slice(0, 5));
  const t1 = Date.now();
  const specs = await collectSpecs(await sampler.nvidia.staticInfo(), undefined);
  console.log(`specs in ${Date.now() - t1} ms`, JSON.stringify(specs, null, 1));

  // Real-data fixture for scripts/preview.html?real
  const init = initPayload({
    history, sample: sampler.latest, selected: undefined, vendor: sampler.vendor,
    processes: sampler.processes, specs, tempWarn: 80, intervalMs: 500,
  });
  fs.writeFileSync(path.join(__dirname, 'preview-data.js'), `window.PREVIEW_INIT = ${JSON.stringify(init)};\n`);
  console.log('wrote dist/preview-data.js');
}

void main();
