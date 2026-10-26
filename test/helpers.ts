import { EventEmitter } from 'events';
import { readFileSync } from 'fs';
import { join } from 'path';
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

/** An Apple Silicon reading as the ioreg collector produces it: unified memory, no temperature / power / clocks. */
export function appleGpu(index: number, util: number | null, extra: Partial<GpuSample> = {}): GpuSample {
  return {
    index, uuid: `apple-${index}-apple-m1-pro`, name: 'Apple M1 Pro', vendor: 'apple', unifiedMemory: true,
    rendererUtil: util, tilerUtil: util, coreCount: 16, allocMiB: 4096, util, memUtil: null, memUsedMiB: 2150,
    memTotalMiB: 32_768, tempC: null, powerW: null, powerLimitW: null, clockMHz: null, clockMaxMHz: null, memClockMHz: null,
    fanPct: null, pstate: null, pcieGen: null, pcieWidth: null, ...extra,
  };
}

/** Text of a file in test/fixtures. */
export const fixture = (name: string): string => readFileSync(join(__dirname, 'fixtures', name), 'utf8');

export function sampleOf(t: number, gpus: GpuSample[], cpu = 20): Sample {
  return { t, gpus, cpu: { util: cpu, perCore: [cpu] }, mem: { usedBytes: 50, totalBytes: 100 } };
}
