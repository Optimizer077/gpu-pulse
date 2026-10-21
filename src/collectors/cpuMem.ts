import * as os from 'os';
import type { CpuSample, MemSample } from '../types';

interface CoreTimes {
  idle: number;
  total: number;
}

function snapshot(): CoreTimes[] {
  return os.cpus().map((c) => {
    const t = c.times;
    return { idle: t.idle, total: t.user + t.nice + t.sys + t.idle + t.irq };
  });
}

/** CPU utilization from the delta of os.cpus() tick counters between calls. */
export class CpuMemCollector {
  private prev = snapshot();

  sampleCpu(): CpuSample {
    const cur = snapshot();
    const perCore = cur.map((c, i) => {
      const p = this.prev[i];
      if (!p) return 0;
      const total = c.total - p.total;
      const idle = c.idle - p.idle;
      return total > 0 ? Math.max(0, Math.min(100, (1 - idle / total) * 100)) : 0;
    });
    this.prev = cur;
    const util = perCore.length ? perCore.reduce((a, b) => a + b, 0) / perCore.length : 0;
    return { util, perCore };
  }

  sampleMem(): MemSample {
    const total = os.totalmem();
    return { usedBytes: total - os.freemem(), totalBytes: total };
  }
}
