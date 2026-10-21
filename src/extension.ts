import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as vscode from 'vscode';
import { collectSpecs } from './collectors/specs';
import { HistoryStore } from './history';
import { DashboardPanel, VIEW_TYPE, type DashboardSource } from './panel';
import { Sampler } from './sampler';
import { GpuStatusBar } from './statusBar';
import type { Specs } from './types';

const SAVE_EVERY_MS = 60_000;

let saveHistory: (() => void) | undefined;

const clamp = (v: number, lo: number, hi: number, dflt: number) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : dflt);

/** settings.json bypasses the schema's min/max, so clamp here. */
function cfg() {
  const c = vscode.workspace.getConfiguration('gpuPulse');
  return {
    fastMs: clamp(c.get<number>('sampleIntervalMs', 1000), 250, 10_000, 1000),
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

export function activate(ctx: vscode.ExtensionContext): void {
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

  const sampler = new Sampler(conf.nvidiaPath, conf.slowMs, (m) => log.appendLine(m));
  const statusBar = new GpuStatusBar();
  statusBar.setVisible(conf.statusBar);
  let specs: Specs | undefined;

  const src: DashboardSource = {
    sampler,
    history,
    specs: () => specs,
    tempWarn: () => conf.tempWarn,
    intervalMs: () => (DashboardPanel.current?.visible ? conf.fastMs : conf.slowMs),
  };

  /** Fast while the dashboard is visible; slow (or paused) otherwise. */
  const applyRate = () => {
    const visible = !!DashboardPanel.current?.visible;
    sampler.setInterval(visible ? conf.fastMs : conf.slowMs);
    if (visible || conf.background || conf.statusBar) void sampler.now();
    else sampler.pause();
  };

  const attach = (p: DashboardPanel) => p.setVisibilityHandler(() => applyRate());

  sampler.onSample((s) => {
    history.add(s);
    if (conf.statusBar) statusBar.update(s, conf.tempWarn);
    DashboardPanel.current?.pushSample(s);
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
    }),
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
}

export function deactivate(): void {
  saveHistory?.();
}
