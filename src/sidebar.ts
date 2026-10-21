import * as vscode from 'vscode';
import type { DashboardSource } from './panel';
import { initPayload, type TickMessage } from './payload';
import { webviewHtml, webviewOptions } from './webviewHtml';

/** The only commands the side bar webview may trigger (exact match, not a prefix). */
const SIDEBAR_COMMANDS: ReadonlySet<string> = new Set(['gpuPulse.openDashboard']);

export const SIDEBAR_VIEW = 'gpuPulse.sidebar';

/**
 * Compact live view for the (primary or secondary) side bar. Shares the sampler, the tick
 * composer and the GPU selection with the dashboard panel; posts only while visible.
 */
export class SidebarProvider implements vscode.WebviewViewProvider, vscode.Disposable {
  view: vscode.WebviewView | undefined;
  private readonly disposables: vscode.Disposable[] = [];

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly src: DashboardSource,
    private readonly onVisibility: () => void,
  ) {}

  get visible(): boolean {
    return !!this.view?.visible;
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = webviewOptions(this.extensionUri);
    view.webview.html = webviewHtml(view.webview, this.extensionUri, 'GPU Pulse', ['dashboard.css', 'sidebar.css'], ['charts.js', 'sidebar.js']);
    this.disposables.push(
      view.webview.onDidReceiveMessage((m: { type: string; uuid?: string; command?: string }) => {
        if (m.type === 'ready') this.sendInit();
        else if (m.type === 'selectGpu') this.src.select(m.uuid);
        else if (m.type === 'command' && m.command && SIDEBAR_COMMANDS.has(m.command)) {
          void vscode.commands.executeCommand(m.command);
        }
      }),
      view.onDidChangeVisibility(() => this.onVisibility()),
      // The view is disposed when moved between containers; resolveWebviewView runs again for the new one.
      view.onDidDispose(() => {
        if (this.view === view) this.view = undefined;
        this.dispose();
        this.onVisibility();
      }),
    );
    this.onVisibility();
  }

  sendInit(): void {
    if (!this.view) return;
    this.src.viewLoaded();
    void this.view.webview.postMessage(
      initPayload({
        history: this.src.history,
        sample: this.src.sampler.latest,
        selected: this.src.selected(),
        vendor: this.src.sampler.vendor,
        processes: [],
        specs: undefined,
        tempWarn: this.src.tempWarn(),
        intervalMs: this.src.intervalMs(),
        alerts: this.src.alerts(),
        compact: true,
      }),
    );
  }

  pushTick(msg: TickMessage): void {
    if (this.view?.visible) void this.view.webview.postMessage(msg);
  }

  dispose(): void {
    for (const d of this.disposables.splice(0)) d.dispose();
  }
}
