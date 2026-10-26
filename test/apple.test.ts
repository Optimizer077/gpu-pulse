import { describe, expect, it } from 'vitest';
import {
  AppleCollector, IOREG, IOREG_ARGS, IOREG_TIMEOUT_MS, IOREG_VRAM_ARGS, macSpecsGpus, parseIoregAccelerators, parseIoregEntries,
  parseIoregVram, parseSystemProfilerDisplays, SYSTEM_PROFILER,
} from '../src/collectors/apple';
import type { ExecFn } from '../src/collectors/exec';
import { fixture } from './helpers';

const GIB = 2 ** 30;
const RAM = 32 * GIB;
const opts = { totalMemBytes: RAM };

/** One accelerator entry in ioreg's text format with the given property lines. */
const entry = (cls: string, ...props: string[]) =>
  `+-o ${cls}  <class ${cls}, id 0x100000a4b, registered, matched, active, busy 0 (0 ms), retain 56>\n    {\n` +
  props.map((p) => `      ${p}\n`).join('') + '    }\n    \n';

describe('parseIoregAccelerators: Apple Silicon fixture', () => {
  const [g, ...rest] = parseIoregAccelerators(fixture('ioreg-apple-m1pro.txt'), opts);

  it('reads one unified-memory GPU with utilization and memory in MiB', () => {
    expect(rest).toEqual([]);
    expect(g).toMatchObject({
      index: 0, uuid: 'apple-0-apple-m1-pro', name: 'Apple M1 Pro', vendor: 'apple', unifiedMemory: true,
      util: 6, rendererUtil: 4, tilerUtil: 5, coreCount: 16, memUsedMiB: 432, allocMiB: 1787.5, memTotalMiB: 32_768,
    });
  });

  it('reports what macOS does not expose as null', () => {
    for (const k of ['memUtil', 'tempC', 'powerW', 'powerLimitW', 'clockMHz', 'clockMaxMHz', 'memClockMHz', 'fanPct', 'pstate', 'pcieGen', 'pcieWidth'] as const) {
      expect(g[k], k).toBeNull();
    }
  });

  it('parses nested dictionaries, arrays, data and booleans without losing the top-level keys', () => {
    const [e] = parseIoregEntries(fixture('ioreg-apple-m1pro.txt'));
    expect(e.cls).toBe('AGXAcceleratorG13X');
    expect(e.props.IOMatchedAtBoot).toBe(true);
    expect(e.props.AGXEnableTraceBuffer).toBe('00000000');
    expect(e.props.GPUConfigurationVariable).toMatchObject({ num_cores: 16, core_mask_list: [255, 255] });
    expect((e.props.IOReportLegend as unknown[])[0]).toMatchObject({ IOReportGroupName: 'GPU Stats' });
    expect(e.props.MetalPluginName).toBe('AGXMetalG13X'); // a key after the long nested values
  });

  it('keeps uuid stable across samples and uses os.totalmem() by default', () => {
    const again = parseIoregAccelerators(fixture('ioreg-apple-m1pro.txt'));
    expect(again[0].uuid).toBe(g.uuid);
    expect(again[0].memTotalMiB).toBeGreaterThan(0);
  });
});

describe('parseIoregAccelerators: Intel Mac with AMD GPU', () => {
  it('reads the iGPU as shared memory and the dGPU from its VRAM counters', () => {
    const [intel, amd] = parseIoregAccelerators(fixture('ioreg-intel-amd.txt'), opts);
    expect(intel).toMatchObject({ index: 0, name: 'Intel Graphics', unifiedMemory: true, util: 2, memUsedMiB: 256, memTotalMiB: 32_768, tempC: null });
    expect(amd).toMatchObject({
      index: 1, uuid: 'apple-1-amd-radeon-gpu', name: 'AMD Radeon GPU', unifiedMemory: false, util: 37,
      memUsedMiB: 1472, memTotalMiB: 8192, // used + free
      tempC: 58, fanPct: 31, clockMHz: 1300, memClockMHz: 1500, powerW: 19, powerLimitW: null,
    });
  });

  it('takes name and VRAM size from the PCI parent, matched by GPU family', () => {
    const vram = parseIoregVram(fixture('ioreg-intel-amd-vram.txt'));
    expect(vram).toEqual([{ family: 'amd', name: 'AMD Radeon Pro 5500M', totalMiB: 8192 }]);
    const [intel, amd] = parseIoregAccelerators(fixture('ioreg-intel-amd.txt'), { ...opts, vram });
    expect(intel.name).toBe('Intel Graphics'); // the AMD entry is not consumed by the iGPU
    expect(amd).toMatchObject({ name: 'AMD Radeon Pro 5500M', uuid: 'apple-1-amd-radeon-pro-5500m', memTotalMiB: 8192 });
  });

  it('identifies a PCI parent without a model by its vendor-id', () => {
    const vram = parseIoregVram(entry('IOPCIDevice', '"vendor-id" = <02100000>', '"VRAM,totalMB" = 4096'));
    expect(vram).toEqual([{ family: 'amd', name: null, totalMiB: 4096 }]);
  });
});

describe('parseIoregAccelerators: missing keys, garbage, empty', () => {
  it('returns no GPUs for empty output (VM / no accelerator)', () => {
    for (const t of [fixture('ioreg-empty.txt'), '', '\n', '   \r\n']) expect(parseIoregAccelerators(t, opts)).toEqual([]);
    expect(parseIoregAccelerators(undefined as unknown as string, opts)).toEqual([]);
  });

  it('gives null fields, not a throw, when PerformanceStatistics or model are missing', () => {
    const [g] = parseIoregAccelerators(entry('AGXAcceleratorG14X', '"IOClass" = "AGXAcceleratorG14X"'), opts);
    expect(g).toMatchObject({ name: 'Apple GPU', uuid: 'apple-0-apple-gpu', unifiedMemory: true, util: null, memUsedMiB: null, memTotalMiB: 32_768, coreCount: null });
  });

  it('falls back through the key aliases (GPU Activity(%))', () => {
    const [g] = parseIoregAccelerators(entry('AGXAcceleratorG15X', '"PerformanceStatistics" = {"GPU Activity(%)"=41,"In use system memory"=1073741824}'), opts);
    expect(g).toMatchObject({ util: 41, memUsedMiB: 1024 });
  });

  it('accepts numeric strings and clamps percentages', () => {
    const [g] = parseIoregAccelerators(entry('AGXAcceleratorG13G', '"PerformanceStatistics" = {"Device Utilization %"="7","Renderer Utilization %"=250,"Tiler Utilization %"=-3}'), opts);
    expect(g).toMatchObject({ util: 7, rendererUtil: 100, tilerUtil: 0 });
  });

  it('survives truncated and malformed values', () => {
    const cases = [
      entry('AGXAcceleratorG13X', '"PerformanceStatistics" = {"Device Utilization %"=12,"In use sys'),
      entry('AGXAcceleratorG13X', '"PerformanceStatistics" = {"Device Utilization %"=12,garbage=1,"In use system memory"=5}'),
      entry('AGXAcceleratorG13X', '"PerformanceStatistics" = ' + '('.repeat(5000), '"model" = "Apple M2"'),
      entry('AGXAcceleratorG13X', '"PerformanceStatistics" = {"Device Utilization %"=}', '"model" = <"Apple M3'),
      '+-o\n+-o <class>\n"key" = 1\n\u0000ÿ{{{))) "a" = = =\n' + entry('AGXAcceleratorG13X', '"model" = "Apple M4 Max"'),
    ];
    const out = cases.map((t) => parseIoregAccelerators(t, opts));
    expect(out[0][0].util).toBe(12);
    expect(out[1][0].util).toBe(12);
    expect(out[2][0].name).toBe('Apple M2');
    expect(out[3][0]).toMatchObject({ util: null, name: 'Apple M3' });
    expect(out[4].map((g) => g.name)).toEqual(['Apple M4 Max']);
  });

  it('ignores child objects of a deeper capture (user clients)', () => {
    const text = fixture('ioreg-apple-m1pro.txt') + '  +-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x1000c1a2f, !registered, !matched, active, busy 0, retain 6>\n    {\n      "IOUserClientCreator" = "pid 412, WindowServer"\n    }\n';
    expect(parseIoregAccelerators(text, opts)).toHaveLength(1);
  });

  it('leaves the unified total null when the RAM size is unknown', () => {
    expect(parseIoregAccelerators(fixture('ioreg-apple-m1pro.txt'), { totalMemBytes: 0 })[0].memTotalMiB).toBeNull();
  });
});

describe('parseSystemProfilerDisplays', () => {
  it('reads model, core count, Metal family and unified memory (Apple Silicon)', () => {
    expect(parseSystemProfilerDisplays(fixture('system_profiler-displays-m1pro.json'))).toEqual([
      { name: 'Apple M1 Pro', vramMiB: null, driver: null, vbios: null, extra: ['Apple', 'built-in'], cores: 16, metal: 'Metal 3', unifiedMemory: true },
    ]);
  });

  it('reads VRAM of a discrete GPU and shared memory of an iGPU (Intel Mac)', () => {
    const [intel, amd] = parseSystemProfilerDisplays(fixture('system_profiler-displays-intel-amd.json'));
    expect(intel).toMatchObject({ name: 'Intel UHD Graphics 630', unifiedMemory: true, vramMiB: null, metal: 'Metal 2', cores: null, extra: ['Intel', 'built-in'] });
    expect(amd).toMatchObject({ name: 'AMD Radeon Pro 5500M', unifiedMemory: false, vramMiB: 8192, metal: 'Metal 3', vbios: '113-D3220E-190', extra: ['AMD', 'PCIe'] });
  });

  it('returns [] for malformed or unexpected JSON', () => {
    for (const t of ['', '{', 'null', '[]', '{"SPDisplaysDataType": 5}', '{"SPDisplaysDataType": [null, 3, {}]}']) {
      expect(parseSystemProfilerDisplays(t)).toEqual([]);
    }
  });
});

/** Fake command runner: answers by the args, records every call. */
function fakeExec(answers: Record<string, string | Error>) {
  const calls: { cmd: string; args: string[]; timeoutMs?: number }[] = [];
  const exec: ExecFn = async (cmd, args, timeoutMs) => {
    calls.push({ cmd, args, timeoutMs });
    const a = answers[args.join(' ')];
    if (a === undefined || a instanceof Error) throw a ?? new Error(`unexpected ${cmd} ${args.join(' ')}`);
    return a;
  };
  return { exec, calls };
}

describe('AppleCollector', () => {
  it('detects Apple Silicon with one ioreg run (2 s timeout) and samples with the same command', async () => {
    const { exec, calls } = fakeExec({ [IOREG_ARGS.join(' ')]: fixture('ioreg-apple-m1pro.txt') });
    const c = new AppleCollector(exec, () => RAM);
    expect(await c.detect()).toBe(true);
    expect(calls).toEqual([{ cmd: IOREG, args: IOREG_ARGS, timeoutMs: IOREG_TIMEOUT_MS }]);
    expect((await c.sample())[0]).toMatchObject({ name: 'Apple M1 Pro', util: 6, memTotalMiB: 32_768 });
    expect(calls).toHaveLength(2);
  });

  it('reads the PCI parents once at detection when a discrete GPU exists', async () => {
    const { exec, calls } = fakeExec({
      [IOREG_ARGS.join(' ')]: fixture('ioreg-intel-amd.txt'),
      [IOREG_VRAM_ARGS.join(' ')]: fixture('ioreg-intel-amd-vram.txt'),
    });
    const c = new AppleCollector(exec, () => RAM);
    expect(await c.detect()).toBe(true);
    expect(c.gpuCount).toBe(2);
    for (let i = 0; i < 3; i++) expect((await c.sample())[1].name).toBe('AMD Radeon Pro 5500M');
    expect(calls.filter((x) => x.args === IOREG_VRAM_ARGS)).toHaveLength(1);
  });

  it('a failing PCI lookup still detects the GPUs', async () => {
    const { exec } = fakeExec({ [IOREG_ARGS.join(' ')]: fixture('ioreg-intel-amd.txt'), [IOREG_VRAM_ARGS.join(' ')]: new Error('boom') });
    const c = new AppleCollector(exec, () => RAM);
    expect(await c.detect()).toBe(true);
    expect((await c.sample())[1]).toMatchObject({ name: 'AMD Radeon GPU', memTotalMiB: 8192 });
  });

  it('does not detect without accelerators or when ioreg fails', async () => {
    expect(await new AppleCollector(fakeExec({ [IOREG_ARGS.join(' ')]: '' }).exec).detect()).toBe(false);
    expect(await new AppleCollector(fakeExec({}).exec).detect()).toBe(false);
  });

  it('macSpecsGpus runs system_profiler once and never rejects', async () => {
    const { exec, calls } = fakeExec({ 'SPDisplaysDataType -json': fixture('system_profiler-displays-m1pro.json') });
    expect((await macSpecsGpus(exec)).map((g) => g.name)).toEqual(['Apple M1 Pro']);
    expect(calls[0].cmd).toBe(SYSTEM_PROFILER);
    expect(await macSpecsGpus(fakeExec({}).exec)).toEqual([]);
  });
});
