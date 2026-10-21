import { AmdLinuxCollector } from './collectors/amdLinux';
import { CpuMemCollector } from './collectors/cpuMem';
import { NvidiaCollector } from './collectors/nvidia';
import { NvidiaStream, type SpawnFn } from './collectors/nvidiaStream';
import type { GpuProcess, GpuSample, Sample } from './types';

const PROCESS_EVERY_MS = 5000;
/** Consecutive stream failures before falling back to spawn-per-tick polling. */
export const MAX_STREAM_RESTARTS = 3;
/** A stream that ran this long counts as healthy; its failure resets the restart budget. */
const HEALTHY_RUN_MS = 60_000;
/** After falling back to polling (e.g. a driver reset hung nvidia-smi), try the stream again this often. */
export const STREAM_RETRY_MS = 10 * 60_000;
/** Rate changes (dashboard tab shown/hidden) settle this long before the stream restarts with the new -lms. */
export const RATE_DEBOUNCE_MS = 1000;

type Listener<T> = (v: T) => void;

export interface SamplerOptions {
  /** Replaces GPU detection and reads (tests, benchmark, GPU_PULSE_FAKE_GPUS). Polled on the timer. */
  gpuReader?: () => GpuSample[] | Promise<GpuSample[]>;
  /** Use one long-lived `nvidia-smi -lms` process (default). false = spawn nvidia-smi every tick. */
  stream?: boolean;
  /** Injected process spawner for the stream (tests). */
  spawn?: SpawnFn;
}

/**
 * Produces samples from either
 *  - a long-lived `nvidia-smi --query-gpu ... -lms <interval>` stream (NVIDIA): a sample is
 *    emitted when every GPU of an interval has reported; restarted with backoff when it dies,
 *    replaced by polling after MAX_STREAM_RESTARTS consecutive failures; or
 *  - a self-rescheduling poll timer (AMD sysfs, CPU-only, fallback, injected reader). At most
 *    one poll tick runs at a time: `now()` during an in-flight tick joins it.
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
  private procInFlight = false;
  private procPolling = true;
  private started = false;
  private stopped = false;
  private paused = false;
  private sampleListeners: Listener<Sample>[] = [];
  private procListeners: Listener<GpuProcess[]>[] = [];
  private lastError: string | null = null;
  // stream state
  private stream: NvidiaStream | undefined;
  private streamStartedAt = 0;
  private streamFailures = 0;
  private restartTimer: NodeJS.Timeout | undefined;
  private rateTimer: NodeJS.Timeout | undefined;
  private watchdog: NodeJS.Timeout | undefined;
  private pollSince = 0;
  latest: Sample | undefined;
  processes: GpuProcess[] = [];
  vendor: 'nvidia' | 'amd' | 'none' = 'none';
  /** 'stream' while the long-lived nvidia-smi process is in use. */
  mode: 'stream' | 'poll' = 'poll';

  constructor(
    nvidiaPath: string,
    intervalMs: number,
    private readonly log: (msg: string) => void = () => {},
    private readonly opts: SamplerOptions = {},
  ) {
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

  /** PID of the streaming nvidia-smi process, if one is running (diagnostics, benchmark). */
  get streamPid(): number | undefined {
    return this.stream?.pid;
  }

  async start(): Promise<void> {
    if (this.opts.gpuReader) this.vendor = 'nvidia';
    else if (await this.nvidia.detect()) this.vendor = 'nvidia';
    else if (this.amd.detect()) this.vendor = 'amd';
    this.mode = this.streamAllowed() ? 'stream' : 'poll';
    this.log(`GPU backend: ${this.opts.gpuReader ? 'synthetic' : this.vendor} (${this.mode === 'stream' ? 'nvidia-smi stream' : 'polling'})`);
    this.started = true;
    if (this.stopped) return;
    if (this.mode === 'stream') this.startStream();
    else await this.tick();
  }

  setInterval(ms: number): void {
    if (ms === this.intervalMs) return;
    this.intervalMs = ms;
    if (this.mode === 'stream') {
      // nvidia-smi can't change -lms in place; a restart costs one process start (~15 ms CPU).
      // Debounced, so flipping between the dashboard tab and an editor doesn't spawn a process each time.
      clearTimeout(this.rateTimer);
      this.rateTimer = setTimeout(() => {
        this.rateTimer = undefined;
        if (this.stream && this.stream.intervalMs !== this.intervalMs) this.startStream();
      }, RATE_DEBOUNCE_MS);
    } else if (this.timer && !this.running) this.schedule();
  }

  /** Query the process list (spawns nvidia-smi every 5 s) only while someone shows it. */
  setProcessPolling(on: boolean): void {
    if (on && !this.procPolling) this.lastProcAt = 0;
    this.procPolling = on;
  }

  /** Stop sampling (e.g. background recording disabled and dashboard hidden). `now()` resumes. */
  pause(): void {
    this.paused = true;
    clearTimeout(this.timer);
    this.timer = undefined;
    this.stopStream();
  }

  /**
   * Resume if paused and take a sample soon. Polling: samples immediately (joining an
   * in-flight tick). Streaming: a (re)started stream reports at once; a running one
   * keeps its cadence.
   */
  now(): Promise<void> {
    this.paused = false;
    if (!this.started || this.stopped) return Promise.resolve();
    // Resuming is a natural moment to give a fallen-back stream another chance.
    if (this.mode === 'poll' && this.pollSince && this.streamAllowed()) this.retryStream();
    if (this.mode === 'stream') {
      if (!this.stream && !this.restartTimer) this.startStream();
      return Promise.resolve();
    }
    return this.running ?? this.tick();
  }

  stop(): void {
    this.stopped = true;
    clearTimeout(this.timer);
    clearTimeout(this.rateTimer);
    this.stopStream();
  }

  // ---- streaming ----

  private streamAllowed(): boolean {
    return this.vendor === 'nvidia' && !this.opts.gpuReader && this.opts.stream !== false;
  }

  /** Leave fallback polling and give the stream a fresh restart budget. */
  private retryStream(): void {
    this.log('Retrying the nvidia-smi stream.');
    clearTimeout(this.timer);
    this.timer = undefined;
    this.streamFailures = 0;
    this.pollSince = 0;
    this.mode = 'stream';
    this.startStream();
  }

  private startStream(): void {
    this.stopStream();
    if (this.stopped || this.paused || !this.nvidia.bin) return;
    const s = new NvidiaStream(
      this.nvidia.bin,
      this.intervalMs,
      this.nvidia.gpuCount,
      {
        onTick: (gpus) => {
          if (this.stream !== s) return;
          this.watchdog?.refresh();
          if (this.streamFailures && Date.now() - this.streamStartedAt >= HEALTHY_RUN_MS) this.streamFailures = 0;
          this.lastError = null;
          this.emitSample(gpus);
        },
        onExit: (reason) => {
          if (this.stream === s) this.onStreamFailure(reason);
        },
      },
      this.opts.spawn,
    );
    this.stream = s;
    this.streamStartedAt = Date.now();
    // A hung nvidia-smi (driver reset, GPU fell off the bus) prints nothing; treat silence as a failure.
    this.watchdog = setTimeout(() => {
      if (this.stream !== s) return;
      s.stop();
      this.onStreamFailure(`no output for ${Math.round(this.staleAfterMs() / 1000)} s`);
    }, this.staleAfterMs());
    s.start();
  }

  private staleAfterMs(): number {
    return Math.max(10_000, this.intervalMs * 5);
  }

  private stopStream(): void {
    clearTimeout(this.restartTimer);
    clearTimeout(this.watchdog);
    this.restartTimer = undefined;
    this.watchdog = undefined;
    const s = this.stream;
    this.stream = undefined;
    s?.stop();
  }

  private onStreamFailure(reason: string): void {
    this.stopStream();
    if (Date.now() - this.streamStartedAt >= HEALTHY_RUN_MS) this.streamFailures = 0;
    this.streamFailures++;
    this.log(`nvidia-smi stream ${reason} (failure ${this.streamFailures}/${MAX_STREAM_RESTARTS})`);
    if (this.stopped || this.paused) return;
    if (this.streamFailures >= MAX_STREAM_RESTARTS) {
      this.log(
        `nvidia-smi stream keeps failing; falling back to running nvidia-smi once per sample ` +
          `(the stream is retried in ${STREAM_RETRY_MS / 60_000} min).`,
      );
      this.mode = 'poll';
      this.pollSince = Date.now();
      void this.tick();
      return;
    }
    const delay = Math.min(30_000, 1000 * 2 ** (this.streamFailures - 1));
    this.restartTimer = setTimeout(() => {
      this.restartTimer = undefined;
      this.startStream();
    }, delay);
  }

  // ---- polling ----

  private schedule(): void {
    clearTimeout(this.timer);
    this.timer = undefined;
    if (this.stopped || this.paused || this.mode !== 'poll') return;
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
      const gpus = this.opts.gpuReader
        ? await this.opts.gpuReader()
        : this.vendor === 'nvidia'
          ? await this.nvidia.sample()
          : this.vendor === 'amd'
            ? this.amd.sample()
            : [];
      this.lastError = null;
      return gpus;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg !== this.lastError) this.log(`GPU sample failed: ${msg}`);
      this.lastError = msg;
      return [];
    }
  }

  private async doTick(): Promise<void> {
    this.emitSample(await this.readGpus());
    if (this.mode === 'poll' && this.pollSince && Date.now() - this.pollSince >= STREAM_RETRY_MS && this.streamAllowed()) {
      this.retryStream();
    }
  }

  // ---- shared ----

  private emit<T>(listeners: Listener<T>[], v: T): void {
    for (const l of listeners) {
      try {
        l(v);
      } catch (e) {
        this.log(`Listener failed: ${e instanceof Error ? e.stack ?? e.message : e}`);
      }
    }
  }

  private emitSample(gpus: GpuSample[]): void {
    const sample: Sample = { t: Date.now(), gpus, cpu: this.cpuMem.sampleCpu(), mem: this.cpuMem.sampleMem() };
    this.latest = sample;
    this.emit(this.sampleListeners, sample);
    this.maybeQueryProcesses(sample.t);
  }

  private maybeQueryProcesses(t: number): void {
    if (!this.procPolling || this.procInFlight || !this.nvidia.available || t - this.lastProcAt < PROCESS_EVERY_MS) return;
    this.lastProcAt = t;
    this.procInFlight = true;
    this.nvidia
      .processes()
      .then((p) => {
        this.processes = p;
        this.emit(this.procListeners, p);
      })
      .catch((e) => this.log(`Process query failed: ${e instanceof Error ? e.message : e}`))
      .finally(() => (this.procInFlight = false));
  }
}
