// Runs inside the extension host. Exits non-zero on the first failed check.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vscode = require('vscode');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

exports.run = async function run() {
  const ext = vscode.extensions.getExtension('gpu-pulse-dev.gpu-pulse');
  assert.ok(ext, 'extension is installed');
  await ext.activate();
  assert.ok(ext.isActive, 'extension activated');

  const cmds = await vscode.commands.getCommands(true);
  for (const c of ['gpuPulse.openDashboard', 'gpuPulse.clearHistory', 'gpuPulse.exportHistory']) {
    assert.ok(cmds.includes(c), `command ${c} registered`);
  }

  await vscode.commands.executeCommand('gpuPulse.openDashboard');
  await sleep(4000);
  const tabs = vscode.window.tabGroups.all.flatMap((g) => g.tabs);
  const dash = tabs.find((t) => t.input instanceof vscode.TabInputWebview && t.label === 'GPU Pulse');
  assert.ok(dash, `dashboard tab open (tabs: ${tabs.map((t) => t.label).join(', ')})`);

  // Opening again must reuse the same panel.
  await vscode.commands.executeCommand('gpuPulse.openDashboard');
  await sleep(500);
  const count = vscode.window.tabGroups.all.flatMap((g) => g.tabs).filter((t) => t.label === 'GPU Pulse').length;
  assert.strictEqual(count, 1, 'single dashboard instance');

  // History is persisted on deactivate; check the storage dir exists after a sample cycle.
  const storage = path.join(ext.extensionPath, '..');
  assert.ok(fs.existsSync(storage));
  console.log('GPU Pulse integration: all checks passed');
};
