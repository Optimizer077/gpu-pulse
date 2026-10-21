import * as vscode from 'vscode';
import { HistoryStore, minuteStart } from './history';
import { HEATMAP_HOURS, initPayload, resolveGpu, type AlertView, type TickMessage } from './payload';
import type { Sampler } from './sampler';
import type { Sample, Specs } from './types';
import { webviewHtml, webviewOptions } from './webviewHtml';

export const VIEW_TYPE = 'gpuPulse.dashboard';
/** The only commands the dashboard webview may trigger (exact match; Clear asks for confirmation itself). */
const PANEL_COMMANDS: ReadonlySet<string> = new Set(['gpuPulse.exportHistory', 'gpuPulse.clearHistory']);

/** State shared by the dashboard panel and the side bar view. */
export interface DashboardSource {
  sampler: Sampler;
  history: HistoryStore;
  specs(): Specs | undefined;
  tempWarn(): number;
  intervalMs(): number;
  /** GPU shown in detail; shared so the panel and side bar agree. */
  selected(): string | undefined;
  select(uuid: string | undefined): void;
  /** A view (re)loaded: its next tick must carry analysis. */
  viewLoaded(): void;
  /** Active alerts (badge in the dashboard header and the side bar). */
  alerts(): AlertView[];
}

export class DashboardPanel implements vscode.Disposable {
  static current: DashboardPanel | undefined;
  private readonly disposables: vscode.Disposable[] = [];
  private lastHeatmapMinute = 0;
  private onVisibility: (visible: boolean) => void = () => {};

  static show(ctx: vscode.ExtensionContext, src: DashboardSource): DashboardPanel {
    if (DashboardPanel.current) {
      DashboardPanel.current.panel.reveal();
      return DashboardPanel.current;
    }
    const panel = vscode.window.createWebviewPanel(VIEW_TYPE, 'GPU Pulse', vscode.ViewColumn.Active, webviewOptions(ctx.extensionUri));
    return (DashboardPanel.current = new DashboardPanel(panel, ctx, src));
  }

  static revive(panel: vscode.WebviewPanel, ctx: vscode.ExtensionContext, src: DashboardSource): DashboardPanel {
    panel.webview.options = webviewOptions(ctx.extensionUri);
    return (DashboardPanel.current = new DashboardPanel(panel, ctx, src));
  }

  private constructor(
    readonly panel: vscode.WebviewPanel,
    ctx: vscode.ExtensionContext,
    private readonly src: DashboardSource,
  ) {
    panel.iconPath = vscode.Uri.joinPath(ctx.extensionUri, 'media', 'icon.png');
    panel.webview.html = webviewHtml(panel.webview, ctx.extensionUri, 'GPU Pulse', ['dashboard.css'], ['charts.js', 'dashboard.js']);
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
    else if (m.type === 'selectGpu') this.src.select(m.uuid);
    else if (m.type === 'command' && m.command && PANEL_COMMANDS.has(m.command)) {
      void vscode.commands.executeCommand(m.command);
    }
  }

  sendInit(): void {
    this.lastHeatmapMinute = minuteStart(Date.now());
    this.src.viewLoaded();
    void this.panel.webview.postMessage(
      initPayload({
        history: this.src.history,
        sample: this.src.sampler.latest,
        selected: this.src.selected(),
        vendor: this.src.sampler.vendor,
        processes: this.src.sampler.processes,
        specs: this.src.specs(),
        tempWarn: this.src.tempWarn(),
        intervalMs: this.src.intervalMs(),
        alerts: this.src.alerts(),
      }),
    );
  }

  /** Called with the shared tick message; posts nothing while hidden (the webview re-inits when shown). */
  pushTick(msg: TickMessage, s: Sample): void {
    if (!this.panel.visible) return;
    void this.panel.webview.postMessage(msg);
    const m = minuteStart(s.t);
    if (m !== this.lastHeatmapMinute) {
      this.lastHeatmapMinute = m;
      const id = resolveGpu(s, this.src.selected());
      void this.panel.webview.postMessage({ type: 'heatmap', heatmap: this.src.history.heatmap(id, HEATMAP_HOURS) });
    }
  }

  pushProcesses(): void {
    if (this.panel.visible) void this.panel.webview.postMessage({ type: 'processes', processes: this.src.sampler.processes });
  }

  pushSpecs(): void {
    void this.panel.webview.postMessage({ type: 'specs', specs: this.src.specs() });
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
