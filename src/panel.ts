import * as vscode from 'vscode';
import { HistoryStore, minuteStart } from './history';
import { analysis, HEATMAP_HOURS, initPayload, resolveGpu } from './payload';
import type { Sampler } from './sampler';
import type { Sample, Specs } from './types';

export const VIEW_TYPE = 'gpuPulse.dashboard';

export interface DashboardSource {
  sampler: Sampler;
  history: HistoryStore;
  specs(): Specs | undefined;
  tempWarn(): number;
  intervalMs(): number;
}

function nonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let s = '';
  for (let i = 0; i < 32; i++) s += chars[Math.floor(Math.random() * chars.length)];
  return s;
}

export class DashboardPanel implements vscode.Disposable {
  static current: DashboardPanel | undefined;
  private readonly disposables: vscode.Disposable[] = [];
  private selected: string | undefined;
  private lastHeatmapMinute = 0;
  private onVisibility: (visible: boolean) => void = () => {};

  static show(ctx: vscode.ExtensionContext, src: DashboardSource): DashboardPanel {
    if (DashboardPanel.current) {
      DashboardPanel.current.panel.reveal();
      return DashboardPanel.current;
    }
    const panel = vscode.window.createWebviewPanel(VIEW_TYPE, 'GPU Pulse', vscode.ViewColumn.Active, {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(ctx.extensionUri, 'media')],
    });
    return (DashboardPanel.current = new DashboardPanel(panel, ctx, src));
  }

  static revive(panel: vscode.WebviewPanel, ctx: vscode.ExtensionContext, src: DashboardSource): DashboardPanel {
    panel.webview.options = { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(ctx.extensionUri, 'media')] };
    return (DashboardPanel.current = new DashboardPanel(panel, ctx, src));
  }

  private constructor(
    readonly panel: vscode.WebviewPanel,
    private readonly ctx: vscode.ExtensionContext,
    private readonly src: DashboardSource,
  ) {
    panel.iconPath = vscode.Uri.joinPath(ctx.extensionUri, 'media', 'icon.png');
    panel.webview.html = this.html();
    panel.onDidDispose(() => this.dispose(), null, this.disposables);
    panel.onDidChangeViewState((e) => this.onVisibility(e.webviewPanel.visible), null, this.disposables);
    panel.webview.onDidReceiveMessage((m) => this.receive(m), null, this.disposables);
  }

  setVisibilityHandler(fn: (visible: boolean) => void): void {
    this.onVisibility = fn;
    fn(this.panel.visible);
  }

  get visible(): boolean {
    return this.panel.visible;
  }

  private receive(m: { type: string; uuid?: string; command?: string }): void {
    if (m.type === 'ready') this.sendInit();
    else if (m.type === 'selectGpu') {
      this.selected = m.uuid;
      this.sendInit();
    } else if (m.type === 'command' && m.command?.startsWith('gpuPulse.')) {
      void vscode.commands.executeCommand(m.command);
    }
  }

  private gpuId(s: Sample | undefined): string | undefined {
    return resolveGpu(s, this.selected);
  }

  sendInit(): void {
    this.lastHeatmapMinute = minuteStart(Date.now());
    void this.panel.webview.postMessage(
      initPayload({
        history: this.src.history,
        sample: this.src.sampler.latest,
        selected: this.selected,
        vendor: this.src.sampler.vendor,
        processes: this.src.sampler.processes,
        specs: this.src.specs(),
        tempWarn: this.src.tempWarn(),
        intervalMs: this.src.intervalMs(),
      }),
    );
  }

  pushSample(s: Sample): void {
    if (!this.panel.visible) return;
    const id = this.gpuId(s);
    const point = this.src.history.recentPoints(id, s.t)[0];
    void this.panel.webview.postMessage({
      type: 'tick',
      sample: s,
      point,
      selected: id,
      ...analysis(this.src.history, s, id, this.src.tempWarn()),
    });
    const m = minuteStart(s.t);
    if (m !== this.lastHeatmapMinute) {
      this.lastHeatmapMinute = m;
      void this.panel.webview.postMessage({ type: 'heatmap', heatmap: this.src.history.heatmap(id, HEATMAP_HOURS) });
    }
  }

  pushProcesses(): void {
    if (this.panel.visible) void this.panel.webview.postMessage({ type: 'processes', processes: this.src.sampler.processes });
  }

  pushSpecs(): void {
    void this.panel.webview.postMessage({ type: 'specs', specs: this.src.specs() });
  }

  private html(): string {
    const w = this.panel.webview;
    const media = (f: string) => w.asWebviewUri(vscode.Uri.joinPath(this.ctx.extensionUri, 'media', f)).toString();
    const n = nonce();
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${w.cspSource} data:; font-src ${w.cspSource}; style-src ${w.cspSource} 'unsafe-inline'; script-src 'nonce-${n}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="${media('dashboard.css')}">
<title>GPU Pulse</title>
</head>
<body>
<div id="app"></div>
<script nonce="${n}" src="${media('charts.js')}"></script>
<script nonce="${n}" src="${media('dashboard.js')}"></script>
</body>
</html>`;
  }

  private disposed = false;

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    DashboardPanel.current = undefined;
    this.onVisibility(false);
    this.panel.dispose();
    for (const d of this.disposables) d.dispose();
  }
}
