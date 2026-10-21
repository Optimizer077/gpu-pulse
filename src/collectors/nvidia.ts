import * as fs from 'fs';
import * as path from 'path';
import { run } from './exec';
import type { GpuProcess, GpuSample } from '../types';

export const GPU_FIELDS = [
  'index',
  'uuid',
  'name',
  'utilization.gpu',
  'utilization.memory',
  'memory.used',
  'memory.total',
  'temperature.gpu',
  'power.draw',
  'power.limit',
  'clocks.gr',
  'clocks.max.gr',
  'clocks.mem',
  'fan.speed',
  'pstate',
  'pcie.link.gen.current',
  'pcie.link.width.current',
] as const;

/** nvidia-smi prints "[N/A]", "N/A" or "[Not Supported]" for unavailable fields. */
function num(raw: string | undefined): number | null {
  if (raw === undefined) return null;
  const v = Number.parseFloat(raw.trim());
  return Number.isFinite(v) ? v : null;
}

function str(raw: string | undefined): string | null {
  const s = raw?.trim();
  if (!s || s.startsWith('[') || s === 'N/A') return null;
  return s;
}

/** Parse `--query-gpu=<GPU_FIELDS> --format=csv,noheader,nounits` output. */
export function parseGpuCsv(stdout: string): GpuSample[] {
  const out: GpuSample[] = [];
  for (const line of stdout.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const c = line.split(',').map((s) => s.trim());
    if (c.length < GPU_FIELDS.length) continue;
    out.push({
      index: num(c[0]) ?? out.length,
      uuid: c[1],
      name: c[2],
      vendor: 'nvidia',
      util: num(c[3]),
      memUtil: num(c[4]),
      memUsedMiB: num(c[5]),
      memTotalMiB: num(c[6]),
      tempC: num(c[7]),
      powerW: num(c[8]),
      powerLimitW: num(c[9]),
      clockMHz: num(c[10]),
      clockMaxMHz: num(c[11]),
      memClockMHz: num(c[12]),
      fanPct: num(c[13]),
      pstate: str(c[14]),
      pcieGen: num(c[15]),
      pcieWidth: num(c[16]),
    });
  }
  return out;
}

/** Parse `--query-compute-apps=pid,process_name,used_memory,gpu_uuid` output. */
export function parseProcessCsv(stdout: string): GpuProcess[] {
  const out: GpuProcess[] = [];
  for (const line of stdout.split(/\r?\n/)) {
    if (!line.trim()) continue;
    // process_name may itself contain commas on Windows paths; take the ends.
    const c = line.split(',').map((s) => s.trim());
    if (c.length < 4) continue;
    const pid = num(c[0]);
    if (pid === null) continue;
    const gpuUuid = c[c.length - 1];
    const usedMiB = num(c[c.length - 2]);
    const full = c.slice(1, c.length - 2).join(',');
    const name = full.startsWith('[') ? 'protected process' : path.basename(full.replace(/\\/g, '/'));
    out.push({ pid, name, usedMiB, gpuUuid });
  }
  return out;
}

/** Extract "CUDA Version: 13.0" from the plain nvidia-smi banner. */
export function parseCudaVersion(banner: string): string | null {
  return /CUDA Version:\s*([\d.]+)/.exec(banner)?.[1] ?? null;
}

/** Count physical GPUs in `nvidia-smi -L` output (MIG devices are indented sub-lines). */
export function countGpus(listing: string): number {
  return listing.split(/\r?\n/).filter((l) => /^GPU \d+:/.test(l)).length;
}

export class NvidiaCollector {
  private binPath: string | null = null;
  /** GPUs listed by `nvidia-smi -L` at detection (0 if unknown). */
  gpuCount = 0;

  constructor(private readonly customPath: string) {}

  /** Locate nvidia-smi. Returns false when no NVIDIA driver is present. */
  async detect(): Promise<boolean> {
    const candidates = [this.customPath, 'nvidia-smi'];
    if (process.platform === 'win32') {
      candidates.push(
        'C:\\Windows\\System32\\nvidia-smi.exe',
        'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe',
      );
    } else {
      candidates.push('/usr/bin/nvidia-smi', '/usr/local/bin/nvidia-smi', '/usr/lib/wsl/lib/nvidia-smi');
    }
    for (const c of candidates) {
      if (!c) continue;
      if (path.isAbsolute(c) && !fs.existsSync(c)) continue;
      try {
        this.gpuCount = countGpus(await run(c, ['-L'], 5000));
        this.binPath = c;
        return true;
      } catch {
        // try next candidate
      }
    }
    return false;
  }

  get available(): boolean {
    return this.binPath !== null;
  }

  /** Resolved nvidia-smi path after detect(), else null. */
  get bin(): string | null {
    return this.binPath;
  }

  async sample(): Promise<GpuSample[]> {
    if (!this.bin) return [];
    const stdout = await run(this.bin, [`--query-gpu=${GPU_FIELDS.join(',')}`, '--format=csv,noheader,nounits']);
    return parseGpuCsv(stdout);
  }

  async processes(): Promise<GpuProcess[]> {
    if (!this.bin) return [];
    const stdout = await run(this.bin, [
      '--query-compute-apps=pid,process_name,used_memory,gpu_uuid',
      '--format=csv,noheader,nounits',
    ]);
    return parseProcessCsv(stdout);
  }

  async staticInfo(): Promise<{ driver: string | null; cuda: string | null; vbios: string[] }> {
    if (!this.bin) return { driver: null, cuda: null, vbios: [] };
    const [banner, q] = await Promise.all([
      run(this.bin, [], 8000).catch(() => ''),
      run(this.bin, ['--query-gpu=driver_version,vbios_version', '--format=csv,noheader']).catch(() => ''),
    ]);
    const rows = q.split(/\r?\n/).filter((l) => l.trim());
    return {
      driver: rows[0]?.split(',')[0]?.trim() ?? null,
      cuda: parseCudaVersion(banner),
      vbios: rows.map((r) => r.split(',')[1]?.trim() ?? ''),
    };
  }
}
