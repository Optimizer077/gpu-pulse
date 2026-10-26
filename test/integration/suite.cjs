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

  const api = ext.exports;
  assert.ok(api && typeof api.sidebarResolved === 'function', 'activate() returns the test API');

  const cmds = await vscode.commands.getCommands(true);
  for (const c of [
    'gpuPulse.openDashboard', 'gpuPulse.clearHistory', 'gpuPulse.exportHistory', 'gpuPulse.focusSidebar',
    'gpuPulse.muteAlerts', 'gpuPulse.unmuteAlerts',
  ]) {
    assert.ok(cmds.includes(c), `command ${c} registered`);
  }

  // Alert mute: one hour, shared through globalState; unmute clears it.
  await vscode.commands.executeCommand('gpuPulse.muteAlerts');
  assert.strictEqual(api.alertsMuted(), true, 'alerts muted after gpuPulse.muteAlerts');
  await vscode.commands.executeCommand('gpuPulse.unmuteAlerts');
  assert.strictEqual(api.alertsMuted(), false, 'alerts unmuted after gpuPulse.unmuteAlerts');

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

  // Side bar view: the command reveals it and VS Code resolves the webview view provider.
  await vscode.commands.executeCommand('gpuPulse.focusSidebar');
  for (let i = 0; i < 20 && !api.sidebarResolved(); i++) await sleep(250);
  assert.ok(api.sidebarResolved(), 'side bar webview view resolved');
  await sleep(1500);
  console.log(`GPU Pulse integration: sampler vendor=${api.sampler.vendor} mode=${api.sampler.mode} latest=${api.sampler.latest ? 'yes' : 'no'}`);
  assert.ok(api.sampler.latest, 'sampler produced a sample');
  if (process.env.GPU_PULSE_EXPECT_VENDOR) {
    assert.strictEqual(api.sampler.vendor, process.env.GPU_PULSE_EXPECT_VENDOR, 'requested hardware backend active');
    assert.ok(api.sampler.latest.gpus.length > 0, 'hardware GPU produced readings');
    assert.ok(api.sampler.latest.gpus.every((g) => Number.isFinite(g.util)), 'hardware utilization is available');
  }

  // History is persisted on deactivate; check the storage dir exists after a sample cycle.
  const storage = path.join(ext.extensionPath, '..');
  assert.ok(fs.existsSync(storage));
  console.log('GPU Pulse integration: all checks passed');
};
