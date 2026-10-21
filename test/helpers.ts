import { EventEmitter } from 'events';
import type { GpuSample, Sample } from '../src/types';

/** Stand-in for the nvidia-smi child process used by the stream. */
export class FakeChild extends EventEmitter {
  pid = 4242;
  killed = false;
  stdout = Object.assign(new EventEmitter(), { setEncoding: () => undefined });
  stderr = new EventEmitter();
  kill(): boolean {
    this.killed = true;
    return true;
  }
  write(s: string): void {
    this.stdout.emit('data', s);
  }
}

/** One `--query-gpu` CSV row in GPU_FIELDS order. */
export const csvRow = (index: number, util: number | string = 50, power = '100.5') =>
  `${index}, GPU-${index}, Tesla T4, ${util}, 3, 100, 15360, 40, ${power}, 70.00, 585, 1590, 5000, [N/A], P0, 3, 16`;

export function gpu(index: number, util: number | null, extra: Partial<GpuSample> = {}): GpuSample {
  return {
    index, uuid: `GPU-${index}`, name: 'NVIDIA GeForce RTX 4090', vendor: 'nvidia', util, memUtil: 0, memUsedMiB: 12_288,
    memTotalMiB: 24_576, tempC: 60, powerW: 300, powerLimitW: 450, clockMHz: null, clockMaxMHz: null, memClockMHz: null,
    fanPct: null, pstate: null, pcieGen: null, pcieWidth: null, ...extra,
  };
}

export function sampleOf(t: number, gpus: GpuSample[], cpu = 20): Sample {
  return { t, gpus, cpu: { util: cpu, perCore: [cpu] }, mem: { usedBytes: 50, totalBytes: 100 } };
}
