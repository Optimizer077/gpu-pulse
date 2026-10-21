import { afterEach, describe, expect, it, vi } from 'vitest';
import { Sampler } from '../src/sampler';

afterEach(() => vi.useRealTimers());

describe('Sampler', () => {
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
