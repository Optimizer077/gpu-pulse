import * as os from 'os';
import { run, type ExecFn } from './exec';
import type { GpuSample, SpecsGpu } from '../types';

/**
 * macOS GPUs through IOKit's IOAccelerator objects, read with `ioreg` (no admin rights, no
 * native addon). One accelerator per GPU: Apple Silicon (AGXAccelerator…), and on Intel Macs
 * the Intel iGPU (IntelAccelerator) and AMD dGPU (AMDRadeonX…GraphicsAccelerator).
 * There is no streaming interface, so the sampler spawns ioreg once per sample.
 */

export const IOREG = '/usr/sbin/ioreg';
export const SYSTEM_PROFILER = '/usr/sbin/system_profiler';
/** Every IOAccelerator with its properties; -d 1: no child objects; -w 0: no line truncation. */
export const IOREG_ARGS = ['-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'];
/** PCI devices with a VRAM size (discrete GPUs of Intel Macs). Read once at detection, only if a discrete GPU exists. */
export const IOREG_VRAM_ARGS = ['-r', '-d', '1', '-w', '0', '-k', 'VRAM,totalMB'];
export const IOREG_TIMEOUT_MS = 2000;
const MIB = 1048576;

/**
 * Property names per field; the first one present wins. Looked up in the PerformanceStatistics
 * dictionary first, then in the object's own properties. Reconstructed, NOT verified on real
 * hardware: when a real capture uses other names, extend this table rather than the parser.
 */
export const IOREG_KEYS = {
  util: ['Device Utilization %', 'GPU Activity(%)'],
  renderer: ['Renderer Utilization %'],
  tiler: ['Tiler Utilization %'],
  /** Bytes of system memory the GPU driver has in use (unified / shared memory). */
  sysMemUsed: ['In use system memory'],
  sysMemAlloc: ['Alloc system memory'],
  /** Discrete GPUs, bytes. */
  vramUsed: ['vramUsedBytes'],
  vramFree: ['vramFreeBytes'],
  /** MiB; on the accelerator or (usually) on its PCI parent, see IOREG_VRAM_ARGS. */
  vramTotalMB: ['VRAM,totalMB'],
  model: ['model'],
  cores: ['gpu-core-count'],
  // AMD drivers on Intel Macs are reported to publish these; Apple Silicon does not (stay null).
  tempC: ['Temperature(C)'],
  powerW: ['Total Power(W)'],
  fanPct: ['Fan Speed(%)'],
  clockMHz: ['Core Clock(MHz)'],
  memClockMHz: ['Memory Clock(MHz)'],
} as const;

// ---- ioreg text format ----

export type IoValue = string | number | boolean | IoValue[] | { [key: string]: IoValue };
export interface IoregEntry {
  name: string;
  cls: string;
  props: Record<string, IoValue>;
}

const HEADER = /^[\s|]*\+-o\s+(.+?)\s+<class\s+([^,>\s]+)/;
const PROP = /^[\s|]*"([^"]*)"\s*=\s*(.*)$/;
const MAX_DEPTH = 16;
const isSpace = (c: string | undefined) => c === ' ' || c === '\t';

/**
 * One ioreg property value: "text", 42, 0x1f, Yes/No, {"k"=v,...}, (a,b), <hex> or <"text">.
 * Returns the value (undefined if none) and the index after it. Never throws; every loop
 * either advances or stops, so garbage ends the value instead of hanging.
 */
function parseValue(s: string, i: number, depth: number): [IoValue | undefined, number] {
  while (isSpace(s[i])) i++;
  const c = s[i];
  if (c === undefined || depth > MAX_DEPTH) return [undefined, s.length];
  if (c === '"') {
    const end = s.indexOf('"', i + 1);
    return end < 0 ? [s.slice(i + 1), s.length] : [s.slice(i + 1, end), end + 1];
  }
  if (c === '<') {
    const end = s.indexOf('>', i + 1);
    const raw = s.slice(i + 1, end < 0 ? s.length : end);
    // <"text"> is printable data (e.g. a PCI model name); <0a1b…> stays a hex string.
    return [raw.startsWith('"') ? raw.replace(/^"|"$/g, '') : raw, end < 0 ? s.length : end + 1];
  }
  if (c === '{' || c === '(') {
    const dict = c === '{';
    const close = dict ? '}' : ')';
    const obj: Record<string, IoValue> = {};
    const arr: IoValue[] = [];
    i++;
    while (i < s.length) {
      while (isSpace(s[i]) || s[i] === ',') i++;
      if (s[i] === close) return [dict ? obj : arr, i + 1];
      if (dict) {
        if (s[i] !== '"') break;
        const end = s.indexOf('"', i + 1);
        if (end < 0) break;
        const key = s.slice(i + 1, end);
        i = end + 1;
        while (isSpace(s[i])) i++;
        if (s[i] !== '=') break;
        const [v, j] = parseValue(s, i + 1, depth + 1);
        i = j;
        if (v !== undefined) obj[key] = v;
      } else {
        const [v, j] = parseValue(s, i, depth + 1);
        if (j <= i) break;
        i = j;
        if (v !== undefined) arr.push(v);
      }
    }
    return [dict ? obj : arr, s.length]; // unterminated: keep what was read
  }
  let j = i;
  while (j < s.length && !',})'.includes(s[j]) && !isSpace(s[j])) j++;
  const tok = s.slice(i, j);
  if (!tok) return [undefined, j];
  if (tok === 'Yes') return [true, j];
  if (tok === 'No') return [false, j];
  const n = Number(tok);
  return [Number.isFinite(n) ? n : tok, j];
}

/** Split `ioreg` text output into objects (`+-o Name <class C, …>`) with their top-level properties. */
export function parseIoregEntries(text: string): IoregEntry[] {
  const out: IoregEntry[] = [];
  let cur: IoregEntry | null = null;
  for (const line of String(text ?? '').split(/\r?\n/)) {
    const h = HEADER.exec(line);
    if (h) {
      cur = { name: h[1], cls: h[2], props: {} };
      out.push(cur);
      continue;
    }
    const p = cur && PROP.exec(line);
    if (!cur || !p) continue;
    const [v] = parseValue(p[2].trimEnd(), 0, 0);
    if (v !== undefined) cur.props[p[1]] = v;
  }
  return out;
}

// ---- accelerators -> samples ----

type Family = 'apple' | 'intel' | 'amd' | 'nvidia' | 'other';
const FAMILIES: [RegExp, Family][] = [
  [/^AGX|\bApple\b/i, 'apple'],
  [/Intel/i, 'intel'],
  [/AMD|Radeon|\bATI\b/i, 'amd'],
  [/NVDA|NVIDIA|GeForce|^nv/i, 'nvidia'],
];
/** PCI vendor-id property bytes as ioreg prints them (little-endian data). */
const VENDOR_IDS: Record<string, Family> = { '6b10': 'apple', '8680': 'intel', '0210': 'amd', de10: 'nvidia' };
const FALLBACK_NAME: Record<Family, string> = { apple: 'Apple GPU', intel: 'Intel Graphics', amd: 'AMD Radeon GPU', nvidia: 'NVIDIA GPU', other: 'GPU' };

const family = (text: string): Family => FAMILIES.find(([re]) => re.test(text))?.[1] ?? 'other';
const isDict = (v: IoValue | undefined): v is Record<string, IoValue> => !!v && typeof v === 'object' && !Array.isArray(v);
const num = (v: IoValue | undefined): number | null => {
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
};
const str = (v: IoValue | undefined): string | null => (typeof v === 'string' && v.trim() ? v.trim() : null);
const pct = (v: number | null) => (v === null ? null : Math.max(0, Math.min(100, v)));
const mib = (bytes: number | null) => (bytes === null ? null : bytes / MIB);
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'gpu';

/** An accelerator object, not a user client or other child that a deeper capture would include. */
const isAccelerator = (e: IoregEntry) =>
  isDict(e.props.PerformanceStatistics) || (/Accelerator/i.test(e.cls) && !/UserClient/i.test(e.cls));

/** Name and VRAM size of a discrete GPU, from its PCI device (IOREG_VRAM_ARGS output). */
export interface VramInfo {
  family: Family;
  name: string | null;
  totalMiB: number | null;
}

export function parseIoregVram(text: string): VramInfo[] {
  return parseIoregEntries(text)
    .map((e) => {
      const name = str(e.props.model);
      const vid = typeof e.props['vendor-id'] === 'string' ? (e.props['vendor-id'] as string).slice(0, 4).toLowerCase() : '';
      const fam = family(name ?? '');
      return { family: fam === 'other' ? VENDOR_IDS[vid] ?? 'other' : fam, name, totalMiB: num(e.props['VRAM,totalMB']) };
    })
    .filter((v) => v.name !== null || v.totalMiB !== null);
}

export interface IoregOptions {
  /** Size of unified memory (bytes); os.totalmem() by default. */
  totalMemBytes?: number;
  /** PCI-parent info for discrete GPUs (parseIoregVram), matched to them by GPU family, in order. */
  vram?: VramInfo[];
}

function toSample(e: IoregEntry, index: number, unifiedMiB: number | null, pool: VramInfo[]): GpuSample {
  const perf = isDict(e.props.PerformanceStatistics) ? e.props.PerformanceStatistics : {};
  const get = (keys: readonly string[]): IoValue | undefined => {
    for (const k of keys) {
      if (k in perf) return perf[k];
      if (k in e.props) return e.props[k];
    }
    return undefined;
  };
  const n = (keys: readonly string[]) => num(get(keys));
  const model = str(get(IOREG_KEYS.model));
  const fam = family(`${e.cls} ${e.name} ${model ?? ''}`);
  const vramUsed = n(IOREG_KEYS.vramUsed);
  const vramFree = n(IOREG_KEYS.vramFree);
  const sysUsed = n(IOREG_KEYS.sysMemUsed);
  const hasVram = vramUsed !== null || vramFree !== null;
  // Unified: Apple Silicon, and Intel iGPUs (system RAM); everything with VRAM counters is discrete.
  const unified = !hasVram && (fam === 'apple' || fam === 'intel' || (fam === 'other' && sysUsed !== null));
  let name = model;
  let usedMiB: number | null;
  let totalMiB: number | null;
  if (unified) {
    usedMiB = mib(sysUsed);
    totalMiB = unifiedMiB;
  } else {
    const k = pool.findIndex((v) => v.family === fam || v.family === 'other');
    const info = k >= 0 ? pool.splice(k, 1)[0] : undefined;
    name ??= info?.name ?? null;
    usedMiB = mib(vramUsed);
    totalMiB = n(IOREG_KEYS.vramTotalMB) ?? info?.totalMiB ?? (vramUsed !== null && vramFree !== null ? mib(vramUsed + vramFree) : null);
  }
  name ??= FALLBACK_NAME[fam];
  return {
    index,
    uuid: `apple-${index}-${slug(name)}`,
    name,
    vendor: 'apple',
    unifiedMemory: unified,
    rendererUtil: pct(n(IOREG_KEYS.renderer)),
    tilerUtil: pct(n(IOREG_KEYS.tiler)),
    coreCount: n(IOREG_KEYS.cores),
    allocMiB: mib(n(IOREG_KEYS.sysMemAlloc)),
    util: pct(n(IOREG_KEYS.util)),
    memUtil: null,
    memUsedMiB: usedMiB,
    memTotalMiB: totalMiB,
    tempC: n(IOREG_KEYS.tempC),
    powerW: n(IOREG_KEYS.powerW),
    powerLimitW: null,
    clockMHz: n(IOREG_KEYS.clockMHz),
    clockMaxMHz: null,
    memClockMHz: n(IOREG_KEYS.memClockMHz),
    fanPct: pct(n(IOREG_KEYS.fanPct)),
    pstate: null,
    pcieGen: null,
    pcieWidth: null,
  };
}

/**
 * Parse `ioreg -r -d 1 -w 0 -c IOAccelerator` output into one sample per GPU. Pure and
 * defensive: missing keys give null, garbage gives fewer properties or no GPUs, never a throw.
 * uuid = `apple-<index>-<name slug>` (ioreg lists accelerators in a stable order).
 */
export function parseIoregAccelerators(text: string, opts: IoregOptions = {}): GpuSample[] {
  try {
    const total = opts.totalMemBytes ?? os.totalmem();
    const unifiedMiB = Number.isFinite(total) && total > 0 ? total / MIB : null;
    const pool = [...(opts.vram ?? [])];
    return parseIoregEntries(text).filter(isAccelerator).map((e, i) => toSample(e, i, unifiedMiB, pool));
  } catch {
    return [];
  }
}

export class AppleCollector {
  private vram: VramInfo[] = [];
  /** Accelerators found by detect() (0: none, e.g. a virtual machine). */
  gpuCount = 0;

  constructor(
    private readonly exec: ExecFn = run,
    private readonly totalMemBytes: () => number = () => os.totalmem(),
  ) {}

  /** True when ioreg lists at least one GPU accelerator. */
  async detect(): Promise<boolean> {
    let gpus: GpuSample[];
    try {
      gpus = parseIoregAccelerators(await this.exec(IOREG, IOREG_ARGS, IOREG_TIMEOUT_MS), { totalMemBytes: this.totalMemBytes() });
    } catch {
      return false;
    }
    this.gpuCount = gpus.length;
    if (gpus.some((g) => !g.unifiedMemory)) {
      this.vram = await this.exec(IOREG, IOREG_VRAM_ARGS, IOREG_TIMEOUT_MS).then(parseIoregVram, () => []);
    }
    return gpus.length > 0;
  }

  async sample(): Promise<GpuSample[]> {
    const text = await this.exec(IOREG, IOREG_ARGS, IOREG_TIMEOUT_MS);
    return parseIoregAccelerators(text, { totalMemBytes: this.totalMemBytes(), vram: this.vram });
  }
}

// ---- system_profiler (specs, read once) ----

const metalLabel = (v: unknown): string | null => {
  const m = typeof v === 'string' ? /metal(\d)(\d*)$/i.exec(v.replace(/^spdisplays_/, '')) : null;
  return m ? `Metal ${m[1]}${m[2] ? `.${m[2]}` : ''}` : null;
};
const VENDORS: Record<string, string> = { apple: 'Apple', amd: 'AMD', ati: 'AMD', intel: 'Intel', nvidia: 'NVIDIA' };
const BUSES: Record<string, string> = { builtin: 'built-in', pcie_device: 'PCIe', pcie: 'PCIe' };
/** "sppci_vendor_Apple" / "AMD (0x1002)" -> "Apple" / "AMD". */
const vendorLabel = (v: unknown): string | null => {
  if (typeof v !== 'string' || !v.trim()) return null;
  const s = v.replace(/^sppci_vendor_/, '').replace(/\s*\(0x[0-9a-f]+\)$/i, '').trim();
  return VENDORS[s.toLowerCase()] ?? (s || null);
};
/** "8 GB" / "1536 MB" -> MiB. */
const sizeMiB = (v: unknown): number | null => {
  const m = typeof v === 'string' ? /([\d.]+)\s*(GB|MB)/i.exec(v) : null;
  return m ? Number(m[1]) * (m[2].toUpperCase() === 'GB' ? 1024 : 1) : null;
};

/**
 * GPUs from `system_profiler SPDisplaysDataType -json`: model, core count, Metal family,
 * vendor, VRAM (discrete) or unified memory. Missing or malformed JSON gives [].
 */
export function parseSystemProfilerDisplays(json: string): SpecsGpu[] {
  let data: unknown;
  try {
    data = JSON.parse(json);
  } catch {
    return [];
  }
  const list = (data as { SPDisplaysDataType?: unknown })?.SPDisplaysDataType;
  if (!Array.isArray(list)) return [];
  return list
    .filter((d): d is Record<string, unknown> => !!d && typeof d === 'object')
    .map((d) => {
      const name = String(d.sppci_model ?? d._name ?? '').trim();
      const vendor = vendorLabel(d.sppci_vendor ?? d.spdisplays_vendor);
      const vram = sizeMiB(d.spdisplays_vram ?? d.spdisplays_vram_shared);
      const unified = /^apple\b/i.test(`${vendor ?? ''} ${name}`.trim()) || (d.spdisplays_vram === undefined && d.spdisplays_vram_shared !== undefined);
      const cores = Number.parseInt(String(d.sppci_cores ?? ''), 10);
      const busKey = typeof d.sppci_bus === 'string' ? d.sppci_bus.replace(/^spdisplays_/, '') : '';
      const bus = busKey ? BUSES[busKey] ?? busKey.replace(/_/g, ' ') : null;
      return {
        name,
        vramMiB: unified ? null : vram,
        driver: null,
        vbios: typeof d['spdisplays_rom-revision'] === 'string' ? d['spdisplays_rom-revision'] : null,
        extra: [vendor, bus].filter((x): x is string => !!x),
        cores: Number.isFinite(cores) && cores > 0 ? cores : null,
        metal: metalLabel(d.spdisplays_mtlgpufamilysupport ?? d.spdisplays_metal ?? d.spdisplays_metalfamily),
        unifiedMemory: unified,
      };
    })
    .filter((g) => g.name);
}

/** Specs GPUs on macOS (one system_profiler run, ~1 s); [] when it fails. */
export async function macSpecsGpus(exec: ExecFn = run): Promise<SpecsGpu[]> {
  try {
    return parseSystemProfilerDisplays(await exec(SYSTEM_PROFILER, ['SPDisplaysDataType', '-json'], 15_000));
  } catch {
    return [];
  }
}
