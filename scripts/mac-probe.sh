#!/usr/bin/env bash
# GPU Pulse macOS probe. Read-only: changes no system state, needs no sudo, sends nothing anywhere.
# Prints what GPU Pulse reads on this Mac and saves the raw captures, so the reconstructed test
# fixtures (test/fixtures/ioreg-*.txt, system_profiler-*.json) can be replaced with real ones:
#   <out>/gpu-ioreg.txt        ioreg -r -d 1 -w 0 -c IOAccelerator   (the per-sample read)
#   <out>/gpu-ioreg.plist      the same as XML (ioreg -a), for plist-based parsers
#   <out>/gpu-ioreg-vram.txt   ioreg -r -d 1 -w 0 -k VRAM,totalMB    (discrete GPUs of Intel Macs; empty on Apple Silicon)
#   <out>/gpu-sp.json          system_profiler SPDisplaysDataType -json, attached displays (spdisplays_ndrvs,
#                              which hold monitor serial numbers) removed
#   <out>/ioreg-timing.txt     time of 20 ioreg runs
# Usage, from the repository root:   bash scripts/mac-probe.sh [out-dir]      (default: ./gpu-pulse-probe)
# Run `npm ci` first to also see the extension's parser output (needs node and node_modules/.bin/esbuild).
set -u

if [ "$(uname -s)" != "Darwin" ]; then
  echo "mac-probe.sh: this probe is for macOS." >&2
  exit 1
fi
cd "$(dirname "$0")/.." || exit 1
OUT=${1:-gpu-pulse-probe}
mkdir -p "$OUT" || exit 1
IOREG=/usr/sbin/ioreg
SP=/usr/sbin/system_profiler
section() { printf '\n==== %s ====\n' "$1"; }

section "machine"
sw_vers
echo "arch: $(uname -m)"
echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
echo "hw.memsize: $(sysctl -n hw.memsize) bytes"

section "ioreg -r -d 1 -w 0 -c IOAccelerator"
"$IOREG" -r -d 1 -w 0 -c IOAccelerator > "$OUT/gpu-ioreg.txt"
cat "$OUT/gpu-ioreg.txt"
[ -s "$OUT/gpu-ioreg.txt" ] || echo "(empty: no GPU accelerator, e.g. a virtual machine)"
"$IOREG" -a -r -d 1 -w 0 -c IOAccelerator > "$OUT/gpu-ioreg.plist" 2>/dev/null

section "ioreg -r -d 1 -w 0 -k VRAM,totalMB"
"$IOREG" -r -d 1 -w 0 -k VRAM,totalMB > "$OUT/gpu-ioreg-vram.txt"
cat "$OUT/gpu-ioreg-vram.txt"
[ -s "$OUT/gpu-ioreg-vram.txt" ] || echo "(empty: expected on Apple Silicon)"

section "PerformanceStatistics, 5 readings 1 s apart (start a GPU workload to see them move)"
for i in 1 2 3 4 5; do
  "$IOREG" -r -d 1 -w 0 -c IOAccelerator | grep '"PerformanceStatistics"' | sed 's/^ *//'
  sleep 1
done

section "system_profiler SPDisplaysDataType -json (spdisplays_ndrvs removed)"
RAW="$OUT/.sp-raw.json"
"$SP" SPDisplaysDataType -json > "$RAW" 2>/dev/null
rm -f "$OUT/gpu-sp.json"
if command -v node > /dev/null 2>&1; then
  node -e 'const fs = require("fs"); const j = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
    for (const d of j.SPDisplaysDataType || []) delete d.spdisplays_ndrvs;
    process.stdout.write(JSON.stringify(j, null, 2) + "\n");' "$RAW" > "$OUT/gpu-sp.json" || rm -f "$OUT/gpu-sp.json"
fi
if [ ! -s "$OUT/gpu-sp.json" ]; then
  # No node: JavaScript for Automation ships with macOS.
  cat > "$OUT/.filter.js" <<'JXA'
function run(argv) {
  ObjC.import('Foundation');
  var j = JSON.parse($.NSString.stringWithContentsOfFileEncodingError(argv[0], $.NSUTF8StringEncoding, null).js);
  (j.SPDisplaysDataType || []).forEach(function (d) { delete d.spdisplays_ndrvs; });
  return JSON.stringify(j, null, 2);
}
JXA
  osascript -l JavaScript "$OUT/.filter.js" "$RAW" > "$OUT/gpu-sp.json" 2>/dev/null || rm -f "$OUT/gpu-sp.json"
  rm -f "$OUT/.filter.js"
fi
rm -f "$RAW" # the unfiltered output may contain monitor serial numbers; it is never printed or kept
if [ -s "$OUT/gpu-sp.json" ]; then cat "$OUT/gpu-sp.json"; else echo "(could not filter the JSON; nothing saved)"; fi

section "ioreg cost: 20 runs"
TIMEFORMAT='real %R s  user %U s  sys %S s'
{ time (for i in $(seq 20); do "$IOREG" -r -d 1 -w 0 -c IOAccelerator > /dev/null; done); } 2> "$OUT/ioreg-timing.txt"
cat "$OUT/ioreg-timing.txt"
awk '{ printf "per run: wall %.1f ms, CPU (user+sys) %.1f ms  ->  npm run bench -- --spawnCpuMs=%.1f\n", $2 * 50, ($5 + $8) * 50, ($5 + $8) * 50 }' "$OUT/ioreg-timing.txt"

section "extension parser (src/collectors/apple.ts)"
if command -v node > /dev/null 2>&1 && [ -x node_modules/.bin/esbuild ]; then
  node_modules/.bin/esbuild scripts/mac-parse.ts --bundle --platform=node --format=cjs --log-level=warning --outfile="$OUT/mac-parse.js" &&
    node "$OUT/mac-parse.js" "$OUT/gpu-ioreg.txt" "$OUT/gpu-ioreg-vram.txt" "$OUT/gpu-sp.json"
else
  echo "skipped: needs node and node_modules (run npm ci in the repository root, then re-run this probe)"
fi

section "done"
echo "Captures saved in $OUT/: gpu-ioreg.txt gpu-ioreg.plist gpu-ioreg-vram.txt gpu-sp.json ioreg-timing.txt"
