import * as vscode from 'vscode';

function nonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let s = '';
  for (let i = 0; i < 32; i++) s += chars[Math.floor(Math.random() * chars.length)];
  return s;
}

export function webviewOptions(extensionUri: vscode.Uri): vscode.WebviewOptions {
  return { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')] };
}

/** Page shell with a strict CSP: only files from media/, scripts by nonce, no network. */
export function webviewHtml(w: vscode.Webview, extensionUri: vscode.Uri, title: string, styles: string[], scripts: string[]): string {
  const media = (f: string) => w.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'media', f)).toString();
  const n = nonce();
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${w.cspSource} data:; font-src ${w.cspSource}; style-src ${w.cspSource} 'unsafe-inline'; script-src 'nonce-${n}';">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
${styles.map((s) => `<link rel="stylesheet" href="${media(s)}">`).join('\n')}
<title>${title}</title>
</head>
<body>
<div id="app"></div>
${scripts.map((s) => `<script nonce="${n}" src="${media(s)}"></script>`).join('\n')}
</body>
</html>`;
}
