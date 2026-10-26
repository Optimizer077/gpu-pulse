import { execFile } from 'child_process';

/** Runs a command and resolves with stdout. Injectable so collectors can be tested with fixtures. */
export type ExecFn = (cmd: string, args: string[], timeoutMs?: number) => Promise<string>;

/** Run a command and resolve with stdout; reject on non-zero exit or timeout. */
export function run(cmd: string, args: string[], timeoutMs = 4000): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { timeout: timeoutMs, windowsHide: true, maxBuffer: 4 * 1024 * 1024 }, (err, stdout) => {
      if (err) reject(err);
      else resolve(stdout);
    });
  });
}
