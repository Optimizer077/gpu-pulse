import * as fs from 'fs';
import * as path from 'path';
import type { GpuSample } from '../types';

const DRM = '/sys/class/drm';

function readNum(file: string): number | null {
  try {
    const v = Number.parseFloat(fs.readFileSync(file, 'utf8').trim());
    return Number.isFinite(v) ? v : null;
  } catch {
    return null;
  }
}

function readText(file: string): string | null {
  try {
    return fs.readFileSync(file, 'utf8').trim();
  } catch {
    return null;
  }
}

const EMPTY: GpuSample = {
  index: 0, uuid: '', name: '', vendor: 'amd', util: null, memUtil: null, memUsedMiB: null, memTotalMiB: null,
  tempC: null, powerW: null, powerLimitW: null, clockMHz: null, clockMaxMHz: null, memClockMHz: null,
  fanPct: null, pstate: null, pcieGen: null, pcieWidth: null,
};

function hwmonDir(device: string): string | null {
  try {
    const root = path.join(device, 'hwmon');
    const first = fs.readdirSync(root)[0];
    return first ? path.join(root, first) : null;
  } catch {
    return null;
  }
}

/**
 * Best-effort AMD GPU reader for the Linux amdgpu driver, using sysfs only
 * (no rocm-smi dependency). Only cards exposing gpu_busy_percent are reported.
 */
export class AmdLinuxCollector {
  private cards: string[] = [];

  detect(): boolean {
    if (process.platform !== 'linux') return false;
    try {
      this.cards = fs
        .readdirSync(DRM)
        .filter((d) => /^card\d+$/.test(d))
        .map((d) => path.join(DRM, d, 'device'))
        .filter((dev) => fs.existsSync(path.join(dev, 'gpu_busy_percent')));
    } catch {
      this.cards = [];
    }
    return this.cards.length > 0;
  }

  sample(): GpuSample[] {
    return this.cards.map((dev, i) => {
      // Reading sensors would wake a runtime-suspended GPU (laptops); a sleeping GPU is idle.
      if (readText(path.join(dev, 'power', 'runtime_status')) === 'suspended') {
        return { ...EMPTY, index: i, uuid: `amd-${path.basename(path.dirname(dev))}`, name: `AMD GPU (${path.basename(path.dirname(dev))})`, util: 0 };
      }
      const hw = hwmonDir(dev);
      const used = readNum(path.join(dev, 'mem_info_vram_used'));
      const total = readNum(path.join(dev, 'mem_info_vram_total'));
      const power = hw ? readNum(path.join(hw, 'power1_average')) ?? readNum(path.join(hw, 'power1_input')) : null;
      const cap = hw ? readNum(path.join(hw, 'power1_cap')) : null;
      const temp = hw ? readNum(path.join(hw, 'temp1_input')) : null;
      return {
        index: i,
        uuid: `amd-${path.basename(path.dirname(dev))}`,
        name: `AMD GPU (${path.basename(path.dirname(dev))})`,
        vendor: 'amd',
        util: readNum(path.join(dev, 'gpu_busy_percent')),
        memUtil: readNum(path.join(dev, 'mem_busy_percent')),
        memUsedMiB: used === null ? null : used / 1048576,
        memTotalMiB: total === null ? null : total / 1048576,
        tempC: temp === null ? null : temp / 1000,
        powerW: power === null ? null : power / 1e6,
        powerLimitW: cap === null ? null : cap / 1e6,
        clockMHz: null,
        clockMaxMHz: null,
        memClockMHz: null,
        fanPct: null,
        pstate: null,
        pcieGen: null,
        pcieWidth: null,
      };
    });
  }
}
