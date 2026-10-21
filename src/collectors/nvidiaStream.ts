import { spawn } from 'child_process';
import { GPU_FIELDS, parseGpuCsv } from './nvidia';
import type { GpuSample } from '../types';

/** Splits chunked stdout into complete lines; a partial trailing line waits for the next chunk. */
export class LineSplitter {
  private rest = '';

  push(chunk: string): string[] {
    const parts = (this.rest + chunk).split(/\r?\n/);
    this.rest = parts.pop() ?? '';
    // A runaway line without newlines is not CSV we can use; drop it rather than grow forever.
    if (this.rest.length > 65_536) this.rest = '';
    return parts.filter((l) => l.trim());
  }
}

/**
 * Groups per-GPU rows into ticks. `nvidia-smi -lms` prints one row per GPU per interval,
 * so a tick is complete when every expected GPU has reported, or when a GPU index
 * repeats (the next interval started). A caller-driven debounce flushes partial ticks.
 */
export class TickGrouper {
  private cur: GpuSample[] = [];
  private maxIndex = -1;

  /** @param expected GPU count from `nvidia-smi -L`; 0 = unknown (learned from the stream). */
  constructor(public expected = 0) {}

  push(g: GpuSample): GpuSample[][] {
    const out: GpuSample[][] = [];
    if (this.cur.some((x) => x.index === g.index)) out.push(this.take());
    this.cur.push(g);
    this.maxIndex = Math.max(this.maxIndex, g.index);
    // nvidia-smi prints GPUs in index order, so a tick is complete at the highest index seen.
    // Requiring both keeps a too-small `expected` from splitting every tick; it self-corrects in one tick.
    if (this.expected > 0 && this.cur.length >= this.expected && g.index >= this.maxIndex) out.push(this.take());
    return out;
  }

  get pending(): boolean {
    return this.cur.length > 0;
  }

  /** Emit a partial tick (debounce expired). */
  flush(): GpuSample[] | null {
    return this.cur.length ? this.take() : null;
  }

  private take(): GpuSample[] {
    const t = this.cur.sort((a, b) => a.index - b.index);
    this.cur = [];
    // Never under-count: a too-small `expected` would split every tick in two.
    this.expected = Math.max(this.expected, t.length);
    return t;
  }
}

/** The subset of ChildProcess the stream uses (lets tests inject a fake). */
export interface StreamChild {
  readonly pid?: number;
  stdout: { setEncoding(e: BufferEncoding): unknown; on(ev: 'data', fn: (chunk: string) => void): unknown } | null;
  stderr: { on(ev: 'data', fn: (chunk: Buffer | string) => void): unknown } | null;
  on(ev: 'exit', fn: (code: number | null, signal: string | null) => void): unknown;
  on(ev: 'error', fn: (e: Error) => void): unknown;
  kill(): boolean;
}

export type SpawnFn = (cmd: string, args: string[]) => StreamChild;

const defaultSpawn: SpawnFn = (cmd, args) =>
  spawn(cmd, args, { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] }) as unknown as StreamChild;

export function streamArgs(intervalMs: number): string[] {
  return [`--query-gpu=${GPU_FIELDS.join(',')}`, '--format=csv,noheader,nounits', '-lms', String(Math.round(intervalMs))];
}

export interface StreamHandlers {
  onTick(gpus: GpuSample[]): void;
  /** Unexpected end of the process (not called after stop()). */
  onExit(reason: string): void;
}

/** One long-lived `nvidia-smi --query-gpu ... -lms <interval>` process. Restarts are the caller's job. */
export class NvidiaStream {
  private child: StreamChild | undefined;
  private readonly lines = new LineSplitter();
  private readonly grouper: TickGrouper;
  private debounce: NodeJS.Timeout | undefined;
  private stderr = '';
  private ended = false;

  constructor(
    private readonly bin: string,
    readonly intervalMs: number,
    expectedGpus: number,
    private readonly handlers: StreamHandlers,
    private readonly spawnFn: SpawnFn = defaultSpawn,
  ) {
    this.grouper = new TickGrouper(expectedGpus);
  }

  get pid(): number | undefined {
    return this.child?.pid;
  }

  start(): void {
    let child: StreamChild;
    try {
      child = this.spawnFn(this.bin, streamArgs(this.intervalMs));
    } catch (e) {
      this.end(`spawn failed: ${e instanceof Error ? e.message : e}`);
      return;
    }
    this.child = child;
    child.stdout?.setEncoding('utf8');
    child.stdout?.on('data', (chunk) => this.onData(chunk));
    child.stderr?.on('data', (c) => {
      if (this.stderr.length < 500) this.stderr += String(c);
    });
    child.on('error', (e) => this.end(e.message));
    child.on('exit', (code, signal) => {
      const err = this.stderr.trim().split(/\r?\n/)[0];
      this.end(`exited (${signal ?? `code ${code}`})${err ? `: ${err}` : ''}`);
    });
  }

  stop(): void {
    this.ended = true;
    clearTimeout(this.debounce);
    try {
      this.child?.kill();
    } catch {
      // already gone
    }
    this.child = undefined;
  }

  /** Exposed for tests: feed raw stdout. */
  onData(chunk: string): void {
    if (this.ended) return;
    for (const line of this.lines.push(chunk)) {
      for (const g of parseGpuCsv(line)) {
        for (const tick of this.grouper.push(g)) this.handlers.onTick(tick);
      }
    }
    clearTimeout(this.debounce);
    this.debounce = undefined;
    if (this.grouper.pending) {
      // Rows of one tick usually arrive within milliseconds, but one slow GPU (e.g. waking from a
      // low-power state) can lag. Flushing early would split the tick into two partial samples
      // (alternating GPU sets in the UI), so wait most of the interval; the next tick's first row
      // completes the group sooner via the repeated-index rule.
      const wait = Math.max(50, this.intervalMs * 0.9);
      this.debounce = setTimeout(() => {
        const t = this.grouper.flush();
        if (t && !this.ended) this.handlers.onTick(t);
      }, wait);
    }
  }

  private end(reason: string): void {
    if (this.ended) return;
    this.ended = true;
    clearTimeout(this.debounce);
    this.child = undefined;
    this.handlers.onExit(reason);
  }
}
