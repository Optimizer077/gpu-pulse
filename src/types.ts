/**
 * Where a GPU reading came from: nvidia-smi, Linux amdgpu sysfs, or macOS IOAccelerator
 * (`ioreg`; Apple Silicon, and the Intel/AMD GPUs of Intel Macs).
 */
export type GpuVendor = 'nvidia' | 'amd' | 'apple';

/** One GPU reading at a single instant. Missing values are `null`. */
export interface GpuSample {
  index: number;
  uuid: string;
  name: string;
  vendor: GpuVendor;
  /**
   * Memory is system RAM shared with the CPU (Apple Silicon, Intel integrated graphics):
   * memUsedMiB is what the GPU driver holds of it and memTotalMiB is all of system RAM, so
   * every "VRAM %" (gauges, history, alerts, insights) means GPU-allocated share of unified memory.
   */
  unifiedMemory?: boolean;
  /** Apple Silicon only (ioreg PerformanceStatistics): renderer / tiler utilization %, GPU cores, allocated MiB. */
  rendererUtil?: number | null;
  tilerUtil?: number | null;
  coreCount?: number | null;
  allocMiB?: number | null;
  util: number | null; // %
  memUtil: number | null; // % memory-controller busy
  memUsedMiB: number | null;
  memTotalMiB: number | null;
  tempC: number | null;
  powerW: number | null;
  powerLimitW: number | null;
  clockMHz: number | null;
  clockMaxMHz: number | null;
  memClockMHz: number | null;
  fanPct: number | null;
  pstate: string | null;
  pcieGen: number | null;
  pcieWidth: number | null;
}

export interface CpuSample {
  util: number; // % across all cores
  perCore: number[]; // % per logical core
}

export interface MemSample {
  usedBytes: number;
  totalBytes: number;
}

export interface Sample {
  t: number; // epoch ms
  gpus: GpuSample[];
  cpu: CpuSample;
  mem: MemSample;
}

export interface GpuProcess {
  pid: number;
  name: string;
  usedMiB: number | null;
  gpuUuid: string;
}

/** Aggregate of all samples that fell in one wall-clock minute. */
export interface MinuteBucket {
  t: number; // minute start, epoch ms
  n: number; // number of samples
  cpu: number; // mean CPU %
  mem: number; // mean RAM %
  gpus: Record<string, GpuMinute>; // keyed by GPU uuid
}

export interface GpuMinute {
  n: number;
  util: number; // mean %
  utilMax: number;
  vram: number; // mean VRAM %
  power: number; // mean W (0 if unknown)
  tempMax: number;
  idle: number; // samples with util < 10%
  busy: number; // samples with util >= 70%
  heldIdle: number; // samples with VRAM >= 30% while util < 10%
}

export interface WindowStats {
  label: string;
  minutes: number;
  coveredMinutes: number;
  util: number | null;
  utilPeak: number | null;
  vram: number | null;
  power: number | null;
  tempPeak: number | null;
  idleShare: number | null; // 0..1
  busyShare: number | null;
  heldIdleShare: number | null;
  cpu: number | null;
  mem: number | null;
}

export type InsightLevel = 'good' | 'warning' | 'serious' | 'critical' | 'info';

export interface Insight {
  level: InsightLevel;
  title: string;
  detail: string;
}

export interface SpecsGpu {
  name: string;
  vramMiB: number | null;
  driver: string | null;
  vbios: string | null;
  extra: string[];
  /** macOS (system_profiler): GPU core count, Metal family ("Metal 3"), memory shared with the CPU. */
  cores?: number | null;
  metal?: string | null;
  unifiedMemory?: boolean;
}

export interface Specs {
  hostname: string;
  os: string;
  kernel: string;
  arch: string;
  uptimeSec: number;
  cpu: { model: string; cores: number; threads: number; baseGHz: number | null; maxGHz: number | null };
  memory: { totalBytes: number; modules: string[] };
  board: string | null;
  system: string | null;
  disks: { name: string; sizeBytes: number; type: string }[];
  gpus: SpecsGpu[];
  cudaVersion: string | null;
  remote: string | null;
}
