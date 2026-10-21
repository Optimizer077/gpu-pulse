// CPU time (ms) consumed so far by another process, for the smoke test and benchmark. null if unknown.
import { execFileSync } from 'child_process';
import * as fs from 'fs';

export function childCpuMs(pid: number | undefined): number | null {
  if (!pid) return null;
  try {
    if (process.platform === 'win32') {
      const out = execFileSync('powershell.exe', ['-NoProfile', '-Command', `(Get-Process -Id ${pid}).TotalProcessorTime.TotalMilliseconds`], {
        encoding: 'utf8',
        windowsHide: true,
      });
      const v = Number.parseFloat(out.trim());
      return Number.isFinite(v) ? v : null;
    }
    // /proc/<pid>/stat fields 14/15: utime/stime in clock ticks (100 Hz on Linux).
    const f = fs.readFileSync(`/proc/${pid}/stat`, 'utf8').split(') ')[1].split(' ');
    return ((Number(f[11]) + Number(f[12])) / 100) * 1000;
  } catch {
    return null;
  }
}
