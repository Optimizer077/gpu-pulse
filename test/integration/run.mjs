// Launches a throwaway VS Code with this extension and runs suite.cjs inside it.
import { downloadAndUnzipVSCode, runTests } from '@vscode/test-electron';
import { existsSync } from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// Set when launched from inside VS Code; it would make Code.exe behave as plain Node.
delete process.env.ELECTRON_RUN_AS_NODE;
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
try {
  let vscodeExecutablePath = await downloadAndUnzipVSCode();
  // Recent macOS bundles name their binary Code; older test-electron versions return Electron.
  if (process.platform === 'darwin' && !existsSync(vscodeExecutablePath)) {
    const code = path.join(path.dirname(vscodeExecutablePath), 'Code');
    if (existsSync(code)) vscodeExecutablePath = code;
  }
  await runTests({
    vscodeExecutablePath,
    extensionDevelopmentPath: root,
    extensionTestsPath: path.join(root, 'test/integration/suite.cjs'),
    launchArgs: ['--disable-extensions', '--skip-welcome', '--skip-release-notes'],
  });
} catch (e) {
  console.error('Integration tests failed:', e);
  process.exit(1);
}
