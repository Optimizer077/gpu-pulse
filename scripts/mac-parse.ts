// Bundled and run by scripts/mac-probe.sh on a Mac (not shipped): shows what the extension's parsers make
// of the captured ioreg / system_profiler output, the status bar text, and live AppleCollector timings.
//   node mac-parse.js <gpu-ioreg.txt> <gpu-ioreg-vram.txt> <gpu-sp.json>
import * as fs from 'fs';
import { AppleCollector, parseIoregAccelerators, parseIoregVram, parseSystemProfilerDisplays } from '../src/collectors/apple';
import { statusView } from '../src/statusText';
import type { GpuSample, Sample } from '../src/types';

const read = (f: string | undefined) => (f && fs.existsSync(f) ? fs.readFileSync(f, 'utf8') : '');
const show = (label: string, v: unknown) => console.log(`${label}:\n${JSON.stringify(v, null, 2)}\n`);
const sampleOf = (gpus: GpuSample[]): Sample => ({ t: Date.now(), gpus, cpu: { util: 0, perCore: [] }, mem: { usedBytes: 0, totalBytes: 1 } });

async function main(): Promise<void> {
  const [ioregFile, vramFile, spFile] = process.argv.slice(2);
  const vram = parseIoregVram(read(vramFile));
  const gpus = parseIoregAccelerators(read(ioregFile), { vram });
  show('parseIoregVram (captured)', vram);
  show('parseIoregAccelerators (captured)', gpus);
  show('parseSystemProfilerDisplays (captured)', parseSystemProfilerDisplays(read(spFile)));
  console.log(`status bar text: ${statusView(sampleOf(gpus), 80).text.replace('$(pulse) ', '')}\n`);

  // Live, as the extension runs it: detect once, then one ioreg per sample.
  const c = new AppleCollector();
  const t0 = performance.now();
  const ok = await c.detect();
  console.log(`AppleCollector.detect(): ${ok} (${c.gpuCount} GPU(s)) in ${(performance.now() - t0).toFixed(0)} ms`);
  if (!ok) return;
  const wall: number[] = [];
  let last: GpuSample[] = [];
  const cpu0 = process.cpuUsage();
  for (let i = 0; i < 20; i++) {
    const t = performance.now();
    last = await c.sample();
    wall.push(performance.now() - t);
  }
  const cpu = process.cpuUsage(cpu0);
  wall.sort((a, b) => a - b);
  console.log(
    `AppleCollector.sample() x20: wall median ${wall[10].toFixed(1)} ms (min ${wall[0].toFixed(1)}, max ${wall[19].toFixed(1)}); ` +
      `extension-host CPU ${((cpu.user + cpu.system) / 1000 / 20).toFixed(2)} ms per sample (spawn + parse, excludes ioreg's own CPU)\n`,
  );
  show('live sample', last);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
