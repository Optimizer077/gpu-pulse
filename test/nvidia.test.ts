import { describe, expect, it } from 'vitest';
import { parseCudaVersion, parseGpuCsv, parseProcessCsv } from '../src/collectors/nvidia';

describe('parseGpuCsv', () => {
  it('parses a real RTX 5060 Ti row', () => {
    const [g] = parseGpuCsv(
      '0, GPU-18b2830b, NVIDIA GeForce RTX 5060 Ti, 1, 0, 5186, 16311, 29, 18.35, 180.00, 2595, 3090, 14001, 51, P1, 4, 8\r\n',
    );
    expect(g).toMatchObject({ index: 0, uuid: 'GPU-18b2830b', util: 1, memUsedMiB: 5186, memTotalMiB: 16311, powerW: 18.35, pstate: 'P1', pcieWidth: 8 });
  });

  it('maps [N/A] and [Not Supported] to null', () => {
    const [g] = parseGpuCsv('1, GPU-x, Tesla T4, 55, 3, 100, 15360, 40, [N/A], [Not Supported], 585, 1590, 5000, [N/A], [N/A], 3, 16');
    expect(g.powerW).toBeNull();
    expect(g.powerLimitW).toBeNull();
    expect(g.fanPct).toBeNull();
    expect(g.pstate).toBeNull();
    expect(g.util).toBe(55);
  });

  it('handles multiple GPUs and skips short lines', () => {
    const out = parseGpuCsv(
      ['0, A, G0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, P0, 4, 16', 'garbage', '1, B, G1, 20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, P0, 4, 16', ''].join('\n'),
    );
    expect(out.map((g) => g.uuid)).toEqual(['A', 'B']);
  });
});

describe('parseProcessCsv', () => {
  it('keeps the basename and tolerates commas in paths', () => {
    const out = parseProcessCsv(
      [
        '4821, /usr/bin/python3, 9842, GPU-1',
        '14316, C:\\Program Files\\Foo, Inc\\app.exe, [N/A], GPU-1',
        '2240, [Insufficient Permissions], [N/A], GPU-1',
      ].join('\n'),
    );
    expect(out[0]).toEqual({ pid: 4821, name: 'python3', usedMiB: 9842, gpuUuid: 'GPU-1' });
    expect(out[1]).toMatchObject({ name: 'app.exe', usedMiB: null });
    expect(out[2].name).toBe('protected process');
  });
});

describe('parseCudaVersion', () => {
  it('reads the banner', () => {
    expect(parseCudaVersion('| NVIDIA-SMI 591.74   Driver Version: 591.74   CUDA Version: 13.1 |')).toBe('13.1');
    expect(parseCudaVersion('no banner')).toBeNull();
  });
});
