import { afterEach, describe, expect, it, vi } from 'vitest';
import { countGpus, parseGpuCsv } from '../src/collectors/nvidia';
import { LineSplitter, NvidiaStream, streamArgs, TickGrouper, type StreamChild } from '../src/collectors/nvidiaStream';
import type { GpuSample } from '../src/types';
import { csvRow as row, FakeChild } from './helpers';

afterEach(() => vi.useRealTimers());

describe('LineSplitter', () => {
  it('carries partial lines (including a split CRLF) to the next chunk', () => {
    const sp = new LineSplitter();
    expect(sp.push('0, a, b\r')).toEqual([]);
    expect(sp.push('\n1, c')).toEqual(['0, a, b']);
    expect(sp.push(', d\n\n2, e\n')).toEqual(['1, c, d', '2, e']);
  });

  it('drops a runaway line without newlines', () => {
    const sp = new LineSplitter();
    sp.push('x'.repeat(70_000));
    expect(sp.push('0, ok\n')).toEqual(['0, ok']);
  });
});

describe('TickGrouper', () => {
  const g = (i: number) => parseGpuCsv(row(i))[0];

  it('emits a tick once every expected GPU reported', () => {
    const t = new TickGrouper(3);
    expect(t.push(g(0))).toEqual([]);
    expect(t.push(g(1))).toEqual([]);
    const [tick] = t.push(g(2));
    expect(tick.map((x) => x.index)).toEqual([0, 1, 2]);
    expect(t.pending).toBe(false);
  });

  it('starts a new tick when an index repeats and learns the GPU count', () => {
    const t = new TickGrouper(0); // unknown count
    t.push(g(0));
    t.push(g(1));
    const [first] = t.push(g(0)); // next interval began
    expect(first.map((x) => x.index)).toEqual([0, 1]);
    expect(t.expected).toBe(2);
    const [second] = t.push(g(1)); // now complete on count alone
    expect(second.map((x) => x.index)).toEqual([0, 1]);
  });

  it('recovers within one tick when told too few GPUs', () => {
    const t = new TickGrouper(1);
    const out: GpuSample[][] = [];
    for (let k = 0; k < 3; k++) for (const i of [0, 1, 2]) out.push(...t.push(g(i)));
    // The first interval is split while the GPU count is learned; after that every tick has all 3 GPUs.
    expect(out.map((x) => x.length)).toEqual([1, 1, 1, 3, 3]);
    expect(t.expected).toBe(3);
    expect(t.pending).toBe(false);
  });

  it('sorts by index and flushes partial ticks on demand', () => {
    const t = new TickGrouper(4);
    t.push(g(2));
    t.push(g(0));
    expect(t.flush()!.map((x) => x.index)).toEqual([0, 2]);
    expect(t.flush()).toBeNull();
  });
});

describe('NvidiaStream', () => {
  function make(expected: number, interval = 500) {
    const child = new FakeChild();
    const ticks: GpuSample[][] = [];
    const exits: string[] = [];
    const spawn = vi.fn((_cmd: string, _args: string[]) => child as unknown as StreamChild);
    const s = new NvidiaStream('nvidia-smi', interval, expected, { onTick: (t) => ticks.push(t), onExit: (r) => exits.push(r) }, spawn);
    s.start();
    return { child, ticks, exits, spawn, s };
  }

  it('passes the interval to nvidia-smi -lms', () => {
    const { spawn } = make(1, 250);
    expect(spawn.mock.calls[0][1]).toEqual(streamArgs(250));
    expect(streamArgs(250).slice(-2)).toEqual(['-lms', '250']);
  });

  it('groups N GPUs per interval across arbitrary chunk boundaries', () => {
    const { child, ticks } = make(4);
    const text = [0, 1, 2, 3, 0, 1, 2, 3].map((i) => row(i, i * 10)).join('\r\n') + '\r\n';
    for (let k = 0; k < text.length; k += 37) child.write(text.slice(k, k + 37)); // ragged chunks
    expect(ticks).toHaveLength(2);
    expect(ticks[1].map((g) => g.util)).toEqual([0, 10, 20, 30]);
  });

  it('maps [N/A] fields to null and skips error lines', () => {
    const { child, ticks } = make(2);
    child.write(`Unable to determine the device handle for GPU 0000:3B:00.0: Unknown Error\n${row(0, '[N/A]', '[N/A]')}\n${row(1)}\n`);
    expect(ticks).toHaveLength(1);
    expect(ticks[0][0].util).toBeNull();
    expect(ticks[0][0].powerW).toBeNull();
    expect(ticks[0][0].fanPct).toBeNull();
    expect(ticks[0][1].powerW).toBe(100.5);
  });

  it('flushes a partial tick only after most of the interval (a GPU stopped reporting)', () => {
    vi.useFakeTimers();
    const { child, ticks } = make(2, 500);
    child.write(`${row(0)}\n`);
    vi.advanceTimersByTime(250);
    expect(ticks).toHaveLength(0); // a slow sibling may still report: don't split the tick early
    vi.advanceTimersByTime(250);
    expect(ticks).toHaveLength(1);
    expect(ticks[0]).toHaveLength(1);
  });

  it('keeps a slow GPU row in the same tick instead of splitting it', () => {
    vi.useFakeTimers();
    const { child, ticks } = make(2, 250);
    child.write(`${row(0)}\n`);
    vi.advanceTimersByTime(150); // longer than the old 100 ms debounce
    child.write(`${row(1)}\n`);
    expect(ticks).toHaveLength(1);
    expect(ticks[0].map((g) => g.index)).toEqual([0, 1]);
  });

  it('reports an unexpected exit with stderr, but not after stop()', () => {
    const a = make(1);
    a.child.stderr.emit('data', 'NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.\n');
    a.child.emit('exit', 9, null);
    expect(a.exits).toEqual(['exited (code 9): NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.']);

    const b = make(1);
    b.s.stop();
    expect(b.child.killed).toBe(true);
    b.child.emit('exit', null, 'SIGTERM');
    b.child.write(`${row(0)}\n`);
    expect(b.exits).toEqual([]);
    expect(b.ticks).toEqual([]);
  });

  it('turns a spawn error into an exit reason', () => {
    const { child, exits } = make(1);
    child.emit('error', new Error('spawn nvidia-smi ENOENT'));
    child.emit('exit', -2, null);
    expect(exits).toEqual(['spawn nvidia-smi ENOENT']);
  });
});

describe('countGpus', () => {
  it('counts physical GPUs, not MIG instances', () => {
    const listing = [
      'GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-a)',
      '  MIG 3g.40gb     Device  0: (UUID: MIG-x)',
      'GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-b)',
      '',
    ].join('\n');
    expect(countGpus(listing)).toBe(2);
  });
});
