import type { GpuSample } from '../types';

/**
 * Deterministic synthetic GPUs for tests, the benchmark and UI work without hardware
 * (set GPU_PULSE_FAKE_GPUS=4 before launching VS Code). GPU 0..n-2 follow a
 * training-like pattern; the last GPU lags to exercise the imbalance insight.
 */
export function fakeGpus(n: number, t = Date.now()): GpuSample[] {
  return Array.from({ length: n }, (_, i) => {
    const phase = t / 9000 + i * 0.7;
    const lag = n > 1 && i === n - 1;
    const util = Math.round(Math.max(0, Math.min(100, lag ? 18 + 10 * Math.sin(phase) : 80 + 15 * Math.sin(phase))));
    const total = 24_576;
    return {
      index: i,
      uuid: `GPU-fake-${i}`,
      name: 'NVIDIA Synthetic GPU',
      vendor: 'nvidia',
      util,
      memUtil: Math.round(util * 0.45),
      memUsedMiB: Math.round(total * (lag ? 0.35 : i === 0 ? 0.88 : 0.71)),
      memTotalMiB: total,
      tempC: Math.round(48 + util * 0.28 + i),
      powerW: Math.round((60 + util * 2.4) * 10) / 10,
      powerLimitW: 350,
      clockMHz: Math.round(1400 + util * 4),
      clockMaxMHz: 1980,
      memClockMHz: 9501,
      fanPct: Math.round(30 + util * 0.4),
      pstate: util > 10 ? 'P2' : 'P8',
      pcieGen: 4,
      pcieWidth: 16,
    };
  });
}
