// Launches a throwaway VS Code with this extension and runs suite.cjs inside it.
import { runTests } from '@vscode/test-electron';
import * as path from 'path';
import { fileURLToPath } from 'url';

// Set when launched from inside VS Code; it would make Code.exe behave as plain Node.
delete process.env.ELECTRON_RUN_AS_NODE;
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
try {
  await runTests({
    extensionDevelopmentPath: root,
    extensionTestsPath: path.join(root, 'test/integration/suite.cjs'),
    launchArgs: ['--disable-extensions', '--skip-welcome', '--skip-release-notes'],
  });
} catch (e) {
  console.error('Integration tests failed:', e);
  process.exit(1);
}
