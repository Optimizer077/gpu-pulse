import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AppleMetalCollector } from '../src/collectors/appleMetal';
import type { StreamChild } from '../src/collectors/nvidiaStream';
import { MAX_STREAM_RESTARTS, RATE_DEBOUNCE_MS, Sampler, STREAM_RETRY_MS } from '../src/sampler';
import type { Sample } from '../src/types';
import { csvRow, FakeChild, gpu } from './helpers';

beforeEach(() => vi.spyOn(AppleMetalCollector.prototype, 'detect').mockResolvedValue(false));
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

/** A Sampler in stream mode with a fake nvidia-smi; `children` collects every spawned process. */
async function streamingSampler(gpuCount = 2, interval = 500) {
  const children: FakeChild[] = [];
  const args: string[][] = [];
  const logs: string[] = [];
  const s = new Sampler('', interval, (m) => logs.push(m), {
    spawn: (_cmd, a) => {
      args.push(a);
      const c = new FakeChild();
      children.push(c);
      return c as unknown as StreamChild;
    },
  });
  vi.spyOn(s.nvidia, 'detect').mockResolvedValue(true);
  Object.defineProperty(s.nvidia, 'bin', { get: () => 'nvidia-smi' });
  s.nvidia.gpuCount = gpuCount;
  s.setProcessPolling(false);
  const samples: Sample[] = [];
  s.onSample((x) => samples.push(x));
  await s.start();
  return { s, children, args, logs, samples, last: () => children[children.length - 1] };
}

const burst = (n: number, util = 50) => Array.from({ length: n }, (_, i) => csvRow(i, util)).join('\n') + '\n';

describe('Sampler (nvidia-smi stream)', () => {
  it('emits one sample per interval with every GPU, from one long-lived process', async () => {
    vi.useFakeTimers();
    const { s, children, samples, last } = await streamingSampler(2);
    expect(s.mode).toBe('stream');
    last().write(burst(2, 10));
    last().write(burst(2, 20).slice(0, 30)); // partial line
    last().write(burst(2, 20).slice(30));
    expect(samples.map((x) => x.gpus.map((g) => g.util))).toEqual([[10, 10], [20, 20]]);
    expect(children).toHaveLength(1);
    expect(s.streamPid).toBe(4242);
    s.stop();
    expect(children[0].killed).toBe(true);
  });

  it('restarts with the new -lms when the interval changes', async () => {
    vi.useFakeTimers();
    const { s, children, args } = await streamingSampler(1, 5000);
    s.setInterval(250);
    expect(children[0].killed).toBe(false); // debounced
    await vi.advanceTimersByTimeAsync(RATE_DEBOUNCE_MS);
    expect(children[0].killed).toBe(true);
    expect(args.map((a) => a[a.length - 1])).toEqual(['5000', '250']);
    s.setInterval(250); // unchanged: no restart
    await vi.advanceTimersByTimeAsync(RATE_DEBOUNCE_MS);
    expect(children).toHaveLength(2);
    s.stop();
  });

  it('rapid rate flips (tab switching) cause at most one restart', async () => {
    vi.useFakeTimers();
    const { s, children } = await streamingSampler(1, 5000);
    for (let i = 0; i < 6; i++) s.setInterval(i % 2 ? 5000 : 500);
    await vi.advanceTimersByTimeAsync(RATE_DEBOUNCE_MS);
    expect(children).toHaveLength(1); // ended back on 5000: the running stream already matches
    s.setInterval(500);
    s.setInterval(5000);
    s.setInterval(500);
    await vi.advanceTimersByTimeAsync(RATE_DEBOUNCE_MS);
    expect(children).toHaveLength(2);
    s.stop();
  });

  it('retries the stream after the fallback period instead of polling forever', async () => {
    vi.useFakeTimers();
    const { s, children } = await streamingSampler(1);
    vi.spyOn(s.nvidia, 'sample').mockResolvedValue([gpu(0, 1)]);
    for (let i = 1; i < MAX_STREAM_RESTARTS; i++) {
      children[i - 1].emit('exit', 1, null);
      await vi.advanceTimersByTimeAsync(1000 * 2 ** (i - 1));
    }
    children[children.length - 1].emit('exit', 1, null);
    expect(s.mode).toBe('poll');
    await vi.advanceTimersByTimeAsync(STREAM_RETRY_MS + 1000);
    expect(s.mode).toBe('stream');
    expect(children).toHaveLength(MAX_STREAM_RESTARTS + 1);
    s.stop();
  });

  it('restarts a dead stream with backoff, then falls back to polling', async () => {
    vi.useFakeTimers();
    const { s, children, logs } = await streamingSampler(1);
    const poll = vi.spyOn(s.nvidia, 'sample').mockResolvedValue([gpu(0, 77)]);
    for (let i = 1; i < MAX_STREAM_RESTARTS; i++) {
      children[i - 1].emit('exit', 1, null);
      expect(children).toHaveLength(i); // waits for the backoff
      await vi.advanceTimersByTimeAsync(1000 * 2 ** (i - 1));
      expect(children).toHaveLength(i + 1);
    }
    children[children.length - 1].emit('exit', 1, null);
    expect(s.mode).toBe('poll');
    expect(logs.some((l) => l.includes('falling back'))).toBe(true);
    await vi.advanceTimersByTimeAsync(1600);
    expect(poll).toHaveBeenCalled();
    expect(children).toHaveLength(MAX_STREAM_RESTARTS);
    s.stop();
  });

  it('treats a silent stream as dead (watchdog)', async () => {
    vi.useFakeTimers();
    const { s, children, logs } = await streamingSampler(1);
    children[0].write(burst(1));
    await vi.advanceTimersByTimeAsync(9000);
    expect(children).toHaveLength(1); // output keeps it alive
    await vi.advanceTimersByTimeAsync(2000);
    expect(children[0].killed).toBe(true);
    expect(logs.some((l) => l.includes('no output'))).toBe(true);
    s.stop();
  });

  it('pause() kills the stream, now() starts a fresh one, stop() prevents restarts', async () => {
    vi.useFakeTimers();
    const { s, children } = await streamingSampler(1);
    s.pause();
    expect(children[0].killed).toBe(true);
    children[0].emit('exit', null, 'SIGTERM'); // our own kill is not a failure
    await vi.advanceTimersByTimeAsync(5000);
    expect(children).toHaveLength(1);
    await s.now();
    expect(children).toHaveLength(2);
    s.stop();
    children[1].emit('exit', 1, null);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(children).toHaveLength(2);
  });

  it('uses spawn-per-tick polling when streaming is disabled', async () => {
    vi.useFakeTimers();
    const s = new Sampler('', 1000, () => {}, { stream: false });
    vi.spyOn(s.nvidia, 'detect').mockResolvedValue(true);
    const poll = vi.spyOn(s.nvidia, 'sample').mockResolvedValue([gpu(0, 5)]);
    s.setProcessPolling(false);
    await s.start();
    expect(s.mode).toBe('poll');
    await vi.advanceTimersByTimeAsync(3000);
    expect(poll.mock.calls.length).toBeGreaterThanOrEqual(3);
    s.stop();
  });

  it('accepts an injected multi-GPU reader (tests, preview, GPU_PULSE_FAKE_GPUS)', async () => {
    vi.useFakeTimers();
    const s = new Sampler('', 1000, () => {}, { gpuReader: () => [gpu(0, 90), gpu(1, 10), gpu(2, 80), gpu(3, 85)] });
    let got: Sample | undefined;
    s.onSample((x) => (got = x));
    await s.start();
    expect(s.vendor).toBe('nvidia');
    expect(got?.gpus.map((g) => g.util)).toEqual([90, 10, 80, 85]);
    s.stop();
  });
});

describe('Sampler', () => {
  it('polls Apple Metal, drops failed reads and recovers without stale GPU values', async () => {
    vi.useFakeTimers();
    const s = new Sampler('', 1000);
    vi.spyOn(s.nvidia, 'detect').mockResolvedValue(false);
    vi.spyOn(s.apple, 'detect').mockResolvedValue(true);
    const apple = { ...gpu(0, 42), vendor: 'apple' as const, uuid: 'apple-0', memTotalMiB: null };
    const read = vi.spyOn(s.apple, 'sample').mockResolvedValue([apple]);
    try {
      await s.start();
      expect(s.vendor).toBe('apple');
      expect(s.mode).toBe('poll');
      expect(s.latest?.gpus).toEqual([apple]);
      read.mockRejectedValueOnce(new Error('ioreg timed out'));
      await s.now();
      expect(s.latest?.gpus).toEqual([]);
      await s.now();
      expect(s.latest?.gpus).toEqual([apple]);
      s.pause();
      const count = read.mock.calls.length;
      await vi.advanceTimersByTimeAsync(5000);
      expect(read).toHaveBeenCalledTimes(count);
      await s.now();
      expect(read).toHaveBeenCalledTimes(count + 1);
    } finally { s.stop(); }
  });

  it('never runs more than one polling loop, however often now() is called', async () => {
    vi.useFakeTimers();
    const s = new Sampler('/nonexistent/nvidia-smi', 1000);
    // Skip hardware detection: vendor stays 'none', so ticks only read CPU/RAM.
    vi.spyOn(s.nvidia, 'detect').mockResolvedValue(false);
    let n = 0;
    s.onSample(() => n++);
    await s.start();
    for (let i = 0; i < 5; i++) void s.now(); // e.g. rapid panel focus changes
    await vi.advanceTimersByTimeAsync(10_000);
    s.stop();
    // 1 start tick + 1 joined/extra now() tick + ~10 timer ticks; 5 parallel loops would give ~50.
    expect(n).toBeGreaterThanOrEqual(10);
    expect(n).toBeLessThanOrEqual(13);
  });

  it('pause() stops sampling until now()', async () => {
    vi.useFakeTimers();
    const s = new Sampler('', 1000);
    vi.spyOn(s.nvidia, 'detect').mockResolvedValue(false);
    let n = 0;
    s.onSample(() => n++);
    await s.start();
    s.pause();
    const before = n;
    await vi.advanceTimersByTimeAsync(5000);
    expect(n).toBe(before);
    await s.now();
    await vi.advanceTimersByTimeAsync(2000);
    expect(n).toBeGreaterThan(before + 1);
    s.stop();
  });

  it('a throwing listener does not stop the loop', async () => {
    vi.useFakeTimers();
    const s = new Sampler('', 1000, () => {});
    vi.spyOn(s.nvidia, 'detect').mockResolvedValue(false);
    let n = 0;
    s.onSample(() => {
      throw new Error('boom');
    });
    s.onSample(() => n++);
    await s.start();
    await vi.advanceTimersByTimeAsync(3000);
    s.stop();
    expect(n).toBeGreaterThanOrEqual(3);
  });
});
