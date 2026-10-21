import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as vscode from 'vscode';
import { fakeGpus } from './collectors/fake';
import { collectSpecs } from './collectors/specs';
import { HistoryStore } from './history';
import { DashboardPanel, VIEW_TYPE, type DashboardSource } from './panel';
import { TickComposer } from './payload';
import { samplingPlan } from './rate';
import { Sampler } from './sampler';
import { SIDEBAR_VIEW, SidebarProvider } from './sidebar';
import { GpuStatusBar } from './statusBar';
import type { Specs } from './types';

const SAVE_EVERY_MS = 60_000;

let saveHistory: (() => void) | undefined;

const clamp = (v: number, lo: number, hi: number, dflt: number) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : dflt);

/** settings.json bypasses the schema's min/max, so clamp here. */
function cfg() {
  const c = vscode.workspace.getConfiguration('gpuPulse');
  return {
    fastMs: clamp(c.get<number>('sampleIntervalMs', 500), 250, 10_000, 500),
    slowMs: clamp(c.get<number>('backgroundIntervalMs', 5000), 1000, 60_000, 5000),
    days: clamp(c.get<number>('historyDays', 7), 1, 30, 7),
    background: c.get<boolean>('recordInBackground', true),
    statusBar: c.get<boolean>('statusBar.enabled', true),
    nvidiaPath: c.get<string>('nvidiaSmiPath', ''),
    tempWarn: clamp(c.get<number>('temperatureWarning', 80), 30, 120, 80),
  };
}

function readJson(file: string, log: vscode.OutputChannel): unknown {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (e) {
    const code = (e as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return undefined;
    // Keep the unreadable file for inspection instead of overwriting it on the next save.
    try {
      fs.copyFileSync(file, `${file}.corrupt-${Date.now()}`);
    } catch {
      // ignore
    }
    log.appendLine(`Could not read history (${file}): ${e}`);
    return undefined;
  }
}

export function activate(ctx: vscode.ExtensionContext): GpuPulseApi {
  const log = vscode.window.createOutputChannel('GPU Pulse');
  let conf = cfg();
  const historyFile = path.join(ctx.globalStorageUri.fsPath, 'history.json');

  const history = new HistoryStore(conf.days);
  history.load(readJson(historyFile, log));

  // Other VS Code windows save to the same file: merge their minutes first, then write atomically.
  saveHistory = () => {
    try {
      history.mergeFrom(readJson(historyFile, log));
      fs.mkdirSync(path.dirname(historyFile), { recursive: true });
      const tmp = `${historyFile}.${process.pid}.tmp`;
      fs.writeFileSync(tmp, JSON.stringify(history.snapshotForSave()));
      fs.renameSync(tmp, historyFile);
    } catch (e) {
      log.appendLine(`Could not save history: ${e}`);
    }
  };
  const saveTimer = setInterval(saveHistory, SAVE_EVERY_MS);

  // Development aid: GPU_PULSE_FAKE_GPUS=4 renders synthetic GPUs instead of reading hardware.
  // Never in an installed (production) extension: a stray env var must not record fake data.
  const fake =
    ctx.extensionMode === vscode.ExtensionMode.Production
      ? 0
      : Math.min(16, Math.max(0, Math.floor(Number(process.env.GPU_PULSE_FAKE_GPUS) || 0)));
  const sampler = new Sampler(conf.nvidiaPath, conf.slowMs, (m) => log.appendLine(m), fake ? { gpuReader: () => fakeGpus(fake) } : {});
  sampler.setProcessPolling(false); // enabled by applyRate() while the dashboard is visible
  const statusBar = new GpuStatusBar();
  statusBar.setVisible(conf.statusBar);
  let specs: Specs | undefined;
  let selected: string | undefined;
  // One composer for every view: a sample is turned into one tick message, posted to each visible view.
  const composer = new TickComposer(history, () => conf.tempWarn);

  const plan = () =>
    samplingPlan({
      dashboardVisible: !!DashboardPanel.current?.visible,
      sidebarVisible: sidebar.visible,
      background: conf.background,
      statusBar: conf.statusBar,
      fastMs: conf.fastMs,
      slowMs: conf.slowMs,
    });

  const src: DashboardSource = {
    sampler,
    history,
    specs: () => specs,
    tempWarn: () => conf.tempWarn,
    intervalMs: () => plan().intervalMs,
    selected: () => selected,
    select: (uuid) => {
      if (!uuid) return;
      // Always answer with an init (even when nothing changed): the webview waits for it to
      // clear its pending selection, otherwise its chart stops appending points.
      selected = uuid;
      DashboardPanel.current?.sendInit();
      sidebar.sendInit();
    },
    viewLoaded: () => composer.invalidate(),
  };

  /** Fast while the dashboard or the side bar is visible (one sampler either way); slow or paused otherwise. */
  const applyRate = () => {
    const p = plan();
    sampler.setInterval(p.intervalMs);
    sampler.setProcessPolling(p.processes);
    if (p.run) void sampler.now();
    else sampler.pause();
  };

  const sidebar = new SidebarProvider(ctx.extensionUri, src, () => applyRate());
  const attach = (p: DashboardPanel) => p.setVisibilityHandler(() => applyRate());

  sampler.onSample((s) => {
    history.add(s);
    if (conf.statusBar) statusBar.update(s, conf.tempWarn);
    const panel = DashboardPanel.current?.visible ? DashboardPanel.current : undefined;
    if (!panel && !sidebar.visible) return;
    const msg = composer.tick(s, selected);
    panel?.pushTick(msg, s);
    sidebar.pushTick(msg);
  });
  sampler.onProcesses(() => DashboardPanel.current?.pushProcesses());

  void sampler.start().then(async () => {
    applyRate();
    try {
      specs = await collectSpecs(await sampler.nvidia.staticInfo(), vscode.env.remoteName);
      DashboardPanel.current?.pushSpecs();
    } catch (e) {
      log.appendLine(`Spec collection failed: ${e}`);
    }
  });

  ctx.subscriptions.push(
    log,
    statusBar,
    { dispose: () => clearInterval(saveTimer) },
    { dispose: () => sampler.stop() },
    vscode.commands.registerCommand('gpuPulse.openDashboard', () => attach(DashboardPanel.show(ctx, src))),
    vscode.commands.registerCommand('gpuPulse.clearHistory', async () => {
      const ok = await vscode.window.showWarningMessage(
        'Delete all recorded GPU Pulse usage history on this machine?',
        { modal: true },
        'Delete',
      );
      if (ok !== 'Delete') return;
      history.clear();
      saveHistory?.();
      DashboardPanel.current?.sendInit();
      sidebar.sendInit();
    }),
    sidebar,
    vscode.window.registerWebviewViewProvider(SIDEBAR_VIEW, sidebar, { webviewOptions: { retainContextWhenHidden: false } }),
    // `<view id>.focus` is generated by VS Code; it reveals the view wherever the user moved it (left or right side bar).
    vscode.commands.registerCommand('gpuPulse.focusSidebar', () => vscode.commands.executeCommand(`${SIDEBAR_VIEW}.focus`)),
    vscode.commands.registerCommand('gpuPulse.exportHistory', async () => {
      // Reuse globalStorageUri's scheme/authority so the home dir resolves on the remote host under Remote-SSH.
      const home = ctx.globalStorageUri.with({ path: vscode.Uri.file(os.homedir()).path });
      const base = vscode.workspace.workspaceFolders?.[0]?.uri ?? home;
      const uri = await vscode.window.showSaveDialog({
        defaultUri: vscode.Uri.joinPath(base, 'gpu-pulse-history.csv'),
        filters: { CSV: ['csv'] },
      });
      if (!uri) return;
      await vscode.workspace.fs.writeFile(uri, Buffer.from(history.toCsv(), 'utf8'));
      void vscode.window.showInformationMessage(`GPU Pulse history exported to ${uri.fsPath}`);
    }),
    vscode.window.registerWebviewPanelSerializer(VIEW_TYPE, {
      async deserializeWebviewPanel(panel) {
        attach(DashboardPanel.revive(panel, ctx, src));
      },
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (!e.affectsConfiguration('gpuPulse')) return;
      conf = cfg();
      history.setRetention(conf.days);
      statusBar.setVisible(conf.statusBar);
      applyRate();
    }),
  );
  return { sidebarResolved: () => !!sidebar.view, sampler };
}

/** Returned from activate(); used by the integration test. */
export interface GpuPulseApi {
  sidebarResolved(): boolean;
  sampler: Sampler;
}

export function deactivate(): void {
  saveHistory?.();
}
