import type { GpuSample } from '../types';
import { run } from './exec';

/** IORegistry counters are best effort and vary between macOS/driver versions. */
export function parseAppleGpus(stdout: string): GpuSample[] {
  const out: GpuSample[] = [];
  for (const entry of stdout.split(/(?=^\+-o )/m)) {
    if (!/^\+-o .*<class AGX/m.test(entry)) continue;
    const model = entry.match(/^\s*"model"\s*=\s*"([^"\r\n]+)"/m)?.[1];
    const stats = entry.match(/^\s*"PerformanceStatistics"\s*=\s*\{([^\r\n]*)\}/m)?.[1] ?? '';
    const values = new Map<string, number>();
    for (const m of stats.matchAll(/"([^"\r\n]+)"\s*=\s*([^,}]+)/g)) {
      const v = Number(m[2].trim());
      if (Number.isFinite(v) && v >= 0) values.set(m[1], v);
    }
    const percent = (key: string) => {
      const v = values.get(key);
      return v !== undefined && v <= 100 ? v : null;
    };
    const allocated = values.get('Alloc system memory');
    out.push({
      index: out.length, uuid: `apple-${out.length}`, name: model ?? 'Apple GPU (Metal)', vendor: 'apple',
      util: percent('Device Utilization %') ?? percent('Renderer Utilization %'),
      memUsedMiB: allocated === undefined ? null : allocated / 1048576,
      // Unified RAM is shared with the CPU; physical RAM is not a dedicated VRAM limit.
      memTotalMiB: null, memUtil: null, tempC: null, powerW: null, powerLimitW: null,
      clockMHz: null, clockMaxMHz: null, memClockMHz: null, fanPct: null, pstate: null,
      pcieGen: null, pcieWidth: null,
    });
  }
  return out;
}

/** Apple Silicon GPUs serving Metal, sampled without sudo or a compiled helper. */
export class AppleMetalCollector {
  private read(): Promise<string> {
    return run('/usr/sbin/ioreg', ['-r', '-c', 'IOAccelerator', '-d', '1', '-l', '-w', '0']);
  }

  async detect(): Promise<boolean> {
    if (process.platform !== 'darwin') return false;
    try {
      return parseAppleGpus(await this.read()).length > 0;
    } catch {
      return false;
    }
  }

  async sample(): Promise<GpuSample[]> {
    return parseAppleGpus(await this.read());
  }
}
