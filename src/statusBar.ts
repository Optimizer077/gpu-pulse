import * as vscode from 'vscode';
import type { Sample } from './types';

export class GpuStatusBar implements vscode.Disposable {
  private readonly item = vscode.window.createStatusBarItem('gpuPulse.status', vscode.StatusBarAlignment.Right, 100);

  constructor() {
    this.item.name = 'GPU Pulse';
    this.item.command = 'gpuPulse.openDashboard';
  }

  setVisible(v: boolean): void {
    if (v) this.item.show();
    else this.item.hide();
  }

  update(s: Sample, tempWarn: number): void {
    const g = s.gpus[0];
    if (!g) {
      this.item.text = `$(pulse) CPU ${Math.round(s.cpu.util)}%`;
      this.item.tooltip = 'GPU Pulse — no GPU detected. Click to open the dashboard.';
      this.item.backgroundColor = undefined;
      return;
    }
    const vram =
      g.memUsedMiB !== null && g.memTotalMiB ? ` · ${(g.memUsedMiB / 1024).toFixed(1)}/${Math.round(g.memTotalMiB / 1024)} GB` : '';
    const temp = g.tempC !== null ? ` · ${Math.round(g.tempC)}°C` : '';
    const more = s.gpus.length > 1 ? ` +${s.gpus.length - 1}` : '';
    this.item.text = `$(pulse) GPU ${g.util ?? '–'}%${vram}${temp}${more}`;
    const md = new vscode.MarkdownString(undefined, true);
    md.appendMarkdown('**GPU Pulse**\n\n');
    for (const x of s.gpus) {
      md.appendMarkdown(
        `$(circuit-board) ${x.name} — ${x.util ?? '–'}% util, ` +
          `${x.memUsedMiB !== null ? Math.round(x.memUsedMiB) : '–'}/${x.memTotalMiB ?? '–'} MiB, ` +
          `${x.tempC ?? '–'} °C, ${x.powerW !== null ? x.powerW.toFixed(0) : '–'} W\n\n`,
      );
    }
    md.appendMarkdown(`$(dashboard) CPU ${Math.round(s.cpu.util)}% · RAM ${Math.round((s.mem.usedBytes / s.mem.totalBytes) * 100)}%\n\n`);
    md.appendMarkdown('_Click to open the dashboard_');
    this.item.tooltip = md;
    this.item.backgroundColor =
      g.tempC !== null && g.tempC >= tempWarn ? new vscode.ThemeColor('statusBarItem.warningBackground') : undefined;
  }

  dispose(): void {
    this.item.dispose();
  }
}
