import * as esbuild from 'esbuild';

const args = new Set(process.argv.slice(2));
const production = args.has('--production');

const common = {
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node18',
  sourcemap: !production,
  minify: production,
  logLevel: 'info',
};

const builds = [
  { ...common, entryPoints: ['src/extension.ts'], outfile: 'dist/extension.js', external: ['vscode'] },
];
if (args.has('--smoke')) {
  builds.push({ ...common, entryPoints: ['scripts/smoke.ts'], outfile: 'dist/smoke.js' });
}

if (args.has('--watch')) {
  for (const b of builds) await (await esbuild.context(b)).watch();
} else {
  await Promise.all(builds.map((b) => esbuild.build(b)));
}
