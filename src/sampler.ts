import { AmdLinuxCollector } from './collectors/amdLinux';
import { CpuMemCollector } from './collectors/cpuMem';
import { NvidiaCollector } from './collectors/nvidia';
import type { GpuProcess, GpuSample, Sample } from './types';

const PROCESS_EVERY_MS = 5000;

type Listener<T> = (v: T) => void;

/**
 * Polls all collectors on a self-rescheduling timer. At most one tick runs at a
 * time: `now()` during an in-flight tick joins it instead of starting a second loop.
 * Independent of the vscode API so it can be smoke-tested from plain Node.
 */
export class Sampler {
  readonly nvidia: NvidiaCollector;
  private readonly amd = new AmdLinuxCollector();
  private readonly cpuMem = new CpuMemCollector();
  private timer: NodeJS.Timeout | undefined;
  private running: Promise<void> | undefined;
  private intervalMs: number;
  private lastProcAt = 0;
  private started = false;
  private stopped = false;
  private paused = false;
  private sampleListeners: Listener<Sample>[] = [];
  private procListeners: Listener<GpuProcess[]>[] = [];
  private lastError: string | null = null;
  latest: Sample | undefined;
  processes: GpuProcess[] = [];
  vendor: 'nvidia' | 'amd' | 'none' = 'none';

  constructor(nvidiaPath: string, intervalMs: number, private readonly log: (msg: string) => void = () => {}) {
    this.nvidia = new NvidiaCollector(nvidiaPath);
    this.intervalMs = intervalMs;
  }

  onSample(l: Listener<Sample>): () => void {
    this.sampleListeners.push(l);
    return () => (this.sampleListeners = this.sampleListeners.filter((x) => x !== l));
  }

  onProcesses(l: Listener<GpuProcess[]>): () => void {
    this.procListeners.push(l);
    return () => (this.procListeners = this.procListeners.filter((x) => x !== l));
  }

  async start(): Promise<void> {
    if (await this.nvidia.detect()) this.vendor = 'nvidia';
    else if (this.amd.detect()) this.vendor = 'amd';
    this.log(`GPU backend: ${this.vendor}`);
    this.started = true;
    await this.tick();
  }

  setInterval(ms: number): void {
    if (ms === this.intervalMs) return;
    this.intervalMs = ms;
    if (this.timer && !this.running) this.schedule();
  }

  /** Stop the timer (e.g. background recording disabled and dashboard hidden). `now()` resumes. */
  pause(): void {
    this.paused = true;
    clearTimeout(this.timer);
    this.timer = undefined;
  }

  /** Take a sample immediately (e.g. when the dashboard opens), resuming if paused. */
  now(): Promise<void> {
    this.paused = false;
    if (!this.started) return Promise.resolve();
    return this.running ?? this.tick();
  }

  stop(): void {
    this.stopped = true;
    clearTimeout(this.timer);
  }

  private schedule(): void {
    clearTimeout(this.timer);
    this.timer = undefined;
    if (this.stopped || this.paused) return;
    this.timer = setTimeout(() => void this.tick(), this.intervalMs);
  }

  private tick(): Promise<void> {
    clearTimeout(this.timer);
    this.timer = undefined;
    this.running ??= this.doTick().finally(() => {
      this.running = undefined;
      this.schedule();
    });
    return this.running;
  }

  /** A failed read yields no GPUs for this sample, so stale values never enter history. */
  private async readGpus(): Promise<GpuSample[]> {
    try {
      const gpus = this.vendor === 'nvidia' ? await this.nvidia.sample() : this.vendor === 'amd' ? this.amd.sample() : [];
      this.lastError = null;
      return gpus;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg !== this.lastError) this.log(`GPU sample failed: ${msg}`);
      this.lastError = msg;
      return [];
    }
  }

  private emit<T>(listeners: Listener<T>[], v: T): void {
    for (const l of listeners) {
      try {
        l(v);
      } catch (e) {
        this.log(`Listener failed: ${e instanceof Error ? e.stack ?? e.message : e}`);
      }
    }
  }

  private async doTick(): Promise<void> {
    const gpus = await this.readGpus();
    const sample: Sample = { t: Date.now(), gpus, cpu: this.cpuMem.sampleCpu(), mem: this.cpuMem.sampleMem() };
    this.latest = sample;
    this.emit(this.sampleListeners, sample);

    if (this.vendor === 'nvidia' && sample.t - this.lastProcAt >= PROCESS_EVERY_MS) {
      this.lastProcAt = sample.t;
      this.nvidia
        .processes()
        .then((p) => {
          this.processes = p;
          this.emit(this.procListeners, p);
        })
        .catch((e) => this.log(`Process query failed: ${e instanceof Error ? e.message : e}`));
    }
  }
}
