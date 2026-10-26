import { afterEach, describe, expect, it, vi } from 'vitest';
import { AppleMetalCollector, parseAppleGpus } from '../src/collectors/appleMetal';
import { run } from '../src/collectors/exec';
import { HistoryStore } from '../src/history';
import { statusView } from '../src/statusText';
import { sampleOf } from './helpers';

vi.mock('../src/collectors/exec', () => ({ run: vi.fn() }));
const platform = process.platform;
afterEach(() => { Object.defineProperty(process, 'platform', { value: platform }); vi.resetAllMocks(); });

const entry = (stats: string, name = 'Apple M2 Pro', cls = 'AGXAcceleratorG14X') =>
  `+-o ${cls}  <class ${cls}, id 0x100000515, registered>\n  {\n    "PerformanceStatistics" = {${stats}}\n    "model" = "${name}"\n  }\n`;
const live = entry('"Device Utilization %"=42,"Renderer Utilization %"=12,"Alloc system memory"=2147483648');

describe('Apple Metal collector', () => {
  it('reads utilization and allocated bytes without inventing a VRAM capacity or sensors', () => {
    const [g] = parseAppleGpus(live);
    expect(g).toMatchObject({ uuid: 'apple-0', name: 'Apple M2 Pro', vendor: 'apple', util: 42,
      memUsedMiB: 2048, memTotalMiB: null, tempC: null, powerW: null, clockMHz: null });
    expect(statusView(sampleOf(Date.now(), [g]), 80).text).toContain('2.0 GB shared');
  });

  it('keeps zero, rejects invalid values, falls back to renderer and ignores non-Apple GPUs', () => {
    expect(parseAppleGpus(entry('"Device Utilization %"=0,"Renderer Utilization %"=80'))[0].util).toBe(0);
    const [g] = parseAppleGpus(entry('"Device Utilization %"=101,"Renderer Utilization %"=33,"Alloc system memory"=-1'));
    expect(g.util).toBe(33);
    expect(g.memUsedMiB).toBeNull();
    expect(parseAppleGpus(entry('"Device Utilization %"=NaN'))[0].util).toBeNull();
    expect(parseAppleGpus(entry('', 'Intel', 'IntelAccelerator'))).toEqual([]);
    expect(parseAppleGpus('')).toEqual([]);
  });

  it('uses stable identities across driver restarts and separates accelerators', () => {
    expect(parseAppleGpus(live.replace('0x100000515', '0x200000999'))[0].uuid).toBe('apple-0');
    expect(parseAppleGpus(live + entry('', 'Apple M3')).map((g) => g.uuid)).toEqual(['apple-0', 'apple-1']);
  });

  it('detects only on macOS and handles unavailable commands', async () => {
    const c = new AppleMetalCollector();
    Object.defineProperty(process, 'platform', { value: 'linux' });
    expect(await c.detect()).toBe(false);
    expect(run).not.toHaveBeenCalled();
    Object.defineProperty(process, 'platform', { value: 'darwin' });
    vi.mocked(run).mockResolvedValueOnce(live);
    expect(await c.detect()).toBe(true);
    vi.mocked(run).mockRejectedValueOnce(new Error('timeout'));
    expect(await c.detect()).toBe(false);
    vi.mocked(run).mockRejectedValueOnce(new Error('timeout'));
    await expect(c.sample()).rejects.toThrow('timeout');
  });

  it('keeps unavailable metrics blank through history save, reload, merging and CSV', () => {
    const now = Math.floor(Date.now() / 60000) * 60000 + 1000;
    const s = sampleOf(now, parseAppleGpus(live));
    const h = new HistoryStore(7);
    h.add(s);
    const copy = new HistoryStore(7);
    copy.load(h.snapshotForSave());
    copy.add({ ...s, t: now + 1 });
    const stats = copy.stats('5 min', 5, 'apple-0', now + 1);
    expect(stats).toMatchObject({ util: 42, vram: null, power: null, tempPeak: null, heldIdleShare: null });
    expect(copy.toCsv().trim().split('\n')[1].split(',').slice(5, 8)).toEqual(['', '', '']);
    // An unavailable read must not dilute a later valid measurement in the same minute.
    copy.add({ ...s, t: now + 2, gpus: [{ ...s.gpus[0], powerW: 12, tempC: 45, memTotalMiB: 4096 }] });
    expect(copy.stats('5 min', 5, 'apple-0', now + 2)).toMatchObject({ vram: 50, power: 12, tempPeak: 45 });
  });
});
