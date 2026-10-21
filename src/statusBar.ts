import * as vscode from 'vscode';
import { statusView, type AlertSummary } from './statusText';
import type { Sample } from './types';

/** Readable text changes slower than the dashboard samples; this caps status bar IPC. */
const MIN_UPDATE_MS = 1000;

export class GpuStatusBar implements vscode.Disposable {
  private readonly item = vscode.window.createStatusBarItem('gpuPulse.status', vscode.StatusBarAlignment.Right, 100);
  private lastAt = 0;
  private text = '';
  private tooltip = '';
  private warn = false;

  constructor() {
    this.item.name = 'GPU Pulse';
    this.item.command = 'gpuPulse.openDashboard';
  }

  setVisible(v: boolean): void {
    if (v) this.item.show();
    else this.item.hide();
  }

  update(s: Sample, tempWarn: number, alerts?: AlertSummary): void {
    if (s.t - this.lastAt < MIN_UPDATE_MS && s.t >= this.lastAt) return;
    this.lastAt = s.t;
    const v = statusView(s, tempWarn, alerts);
    // Each property write is a message to the renderer; only send what changed.
    if (v.text !== this.text) this.item.text = this.text = v.text;
    if (v.tooltip !== this.tooltip) {
      this.tooltip = v.tooltip;
      const md = new vscode.MarkdownString(v.tooltip, true);
      this.item.tooltip = md;
    }
    if (v.warn !== this.warn) {
      this.warn = v.warn;
      this.item.backgroundColor = v.warn ? new vscode.ThemeColor('statusBarItem.warningBackground') : undefined;
    }
  }

  dispose(): void {
    this.item.dispose();
  }
}
