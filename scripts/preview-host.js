// Dev-only harness shared by preview.html and preview-sidebar.html: synthetic GPUs, an emulated
// extension host (ready -> init, selectGpu -> init, live ticks) and optional render counters. Not shipped.
(function () {
  const q = new URLSearchParams(location.search);
  const N = Math.max(1, Math.min(16, Number(q.get('gpus')) || 1));
  const INTERVAL = Math.max(100, Number(q.get('interval')) || 500);
  const legacy = q.has('legacy');

  // ---- instrumentation (installed before the dashboard scripts) ----
  const B = (window.__bench = { draws: 0, rafFrames: 0, rafMs: 0, mutations: 0, msgs: 0, handlerMs: 0, ticks: 0 });
  if (q.has('bench')) {
    const clr = CanvasRenderingContext2D.prototype.clearRect;
    CanvasRenderingContext2D.prototype.clearRect = function (...a) { B.draws++; return clr.apply(this, a); };
    const raf = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = (cb) => raf((t) => { const s = performance.now(); B.rafFrames++; cb(t); B.rafMs += performance.now() - s; });
    window.addEventListener('message', () => { B._t0 = performance.now(); });
  }

  // ---- synthetic data ----
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const lagging = (i) => N > 1 && i === N - 1;
  // Training-like pattern: busy plateaus with dataloader dips; the last of several GPUs lags.
  const steady = q.has('steady'); // &steady: an idle machine whose readings do not change (common case)
  const util = (t, i = 0) => {
    if (steady && t > Date.now() - 5000) return 1;
    const m = (t / 60000) % 240;
    if (m < 40) return 2 + rnd() * 4;
    const base = (lagging(i) ? 22 : 78) + 14 * Math.sin(t / 47000 + i * 0.9);
    return Math.max(0, Math.min(100, base - (rnd() < 0.12 ? 55 * rnd() : 0) + rnd() * 6));
  };
  const multi = N > 1;
  const NAME = multi ? 'NVIDIA A100-SXM4-80GB' : 'NVIDIA GeForce RTX 5060 Ti';
  const TOTAL = multi ? 81920 : 16311;
  const uuid = (i) => `GPU-${i + 1}`;
  function gpuAt(t, i) {
    const u = Math.round(util(t, i));
    const memFrac = lagging(i) ? 0.38 : i === 0 && multi ? 0.91 : 0.72;
    return { index: i, uuid: uuid(i), name: NAME, vendor: 'nvidia', util: u, memUtil: Math.round(u * 0.47),
      memUsedMiB: Math.round(TOTAL * memFrac + (steady ? 0 : rnd() * 200)), memTotalMiB: TOTAL, tempC: Math.round(52 + u * 0.18 + (i === 1 && multi ? 18 : 0)),
      powerW: Math.round((multi ? 90 + u * 2.9 : 22 + u * 1.5) * 10) / 10, powerLimitW: multi ? 400 : 180,
      clockMHz: Math.round(multi ? 900 + u * 5 : 1600 + u * 12), clockMaxMHz: multi ? 1410 : 3090, memClockMHz: multi ? 1593 : 14001,
      fanPct: multi ? null : Math.round(30 + u * 0.35), pstate: u > 10 ? 'P2' : 'P8', pcieGen: 4, pcieWidth: multi ? 16 : 8 };
  }
  const cores = () => Array.from({ length: 20 }, (_, i) => steady ? 3 : Math.round(15 + 70 * Math.abs(Math.sin(i * 1.7 + Date.now() / 3000)) * (0.8 + rnd() * 0.2)));
  function sampleAt(t) {
    const c = cores();
    return { t, gpus: Array.from({ length: N }, (_, i) => gpuAt(t, i)), cpu: { util: c.reduce((a, b) => a + b, 0) / c.length, perCore: c },
      mem: { usedBytes: (31.7 + (steady ? 0 : rnd() * 0.2)) * 2 ** 30, totalBytes: 63.65 * 2 ** 30 } };
  }
  const point = (s, id) => {
    const g = s.gpus.find((x) => x.uuid === id) || s.gpus[0];
    return [s.t, g.util, Math.round((g.memUsedMiB / g.memTotalMiB) * 1000) / 10, Math.round(s.cpu.util * 10) / 10, 49.8];
  };

  let selected = uuid(0);
  const selIndex = () => Number(selected.split('-')[1]) - 1;
  function analysisFor(t) {
    const k = selIndex(), lag = lagging(k);
    const W = (label, minutes, covered, u, idle, busy) => ({ label, minutes, coveredMinutes: covered, util: lag ? u * 0.3 : u + (t % 7) / 10,
      utilPeak: 100, vram: lag ? 38 : 64, power: 128, tempPeak: 71, idleShare: idle, busyShare: lag ? busy * 0.2 : busy, heldIdleShare: 0.04, cpu: 34, mem: 48 });
    const insights = [
      { level: 'good', title: 'GPU kept busy', detail: 'Above 70% utilization for 55% of the last hour (mean 61%).' },
      { level: 'warning', title: 'Peaked at 81 °C this hour', detail: 'Above your 80 °C threshold at least once in the last hour.' },
      { level: 'info', title: 'High utilization, low power draw', detail: 'Utilization counts time any kernel runs, not how many cores are busy.' },
    ];
    if (multi) {
      insights.splice(1, 0, { level: 'warning', title: `GPU ${N - 1} well below the others`,
        detail: `Last 10 min: GPU ${N - 1} 24% vs 81% mean on the other GPUs. If one job spans these GPUs (e.g. data-parallel training), this fits a straggler, an uneven data split, or a rank that is not using its GPU. If they run separate jobs, it is expected.` });
      insights.push({ level: 'info', title: 'Uneven memory across GPUs',
        detail: 'Last 10 min: GPU 0 averaged 91% VRAM, GPU ' + (N - 1) + ' 38%. In data-parallel jobs rank 0 often holds extra state (gathered outputs, logging); a large gap can also mean an unbalanced model split.' });
    }
    return {
      windows: [W('5 min', 5, 5, 84.2, 0.03, 0.82), W('30 min', 30, 30, 79.5, 0.07, 0.74), W('1 hour', 60, 60, 61.3, 0.27, 0.55), W('24 hours', 1440, 1080, 52.8, 0.31, 0.46)],
      grade: lag ? { label: 'Underused', level: 'low' } : { label: 'Well used', level: 'good' },
      insights,
      fleet: multi ? Array.from({ length: N }, (_, i) => ({ uuid: uuid(i), util: lagging(i) ? 24.3 : 79 + i * 1.7 })) : [],
    };
  }

  function initMsg() {
    const now = Date.now(), k = selIndex();
    const recent = [];
    for (let t = now - 3600e3; t <= now; t += 1000) {
      const u = util(t, k);
      recent.push([t, Math.round(u), u > 10 ? 71 + rnd() * 2 : 32, 18 + rnd() * 22 + u * 0.2, 46 + rnd()]);
    }
    const start = Math.floor(now / 60000) * 60000 - (25 * 60 - 1) * 60000;
    const cells = [];
    for (let i = 0; i < 25 * 60; i++) {
      const t = start + i * 60000, hr = new Date(t).getHours();
      cells.push(hr >= 2 && hr < 8 ? null : Math.round(util(t, k) * 10) / 10);
    }
    const a = analysisFor(now);
    const procs = Array.from({ length: N }, (_, i) => [
      { pid: 48211 + i, name: multi ? `python (rank ${i})` : 'python', usedMiB: Math.round(TOTAL * 0.6), gpuUuid: uuid(i) },
      ...(i === 0 ? [{ pid: 3102, name: 'jupyter-kernel', usedMiB: 1418, gpuUuid: uuid(0) }, { pid: 1931, name: 'Xorg', usedMiB: 312, gpuUuid: uuid(0) }] : []),
    ]).flat();
    return Object.assign(legacy ? a : {}, {
      type: 'init', selected, vendor: 'nvidia', platform: 'linux', config: { tempWarn: 80, intervalMs: INTERVAL },
      sample: sampleAt(now), recent, heatmap: { start, cells }, processes: procs, analysis: a,
      specs: { hostname: multi ? 'dgx-node-07' : 'gpu-box-01', os: 'Ubuntu 22.04.4 LTS', kernel: '6.8.0-45-generic', arch: 'x64', uptimeSec: 71306,
        cpu: { model: multi ? 'AMD EPYC 7742 64-Core' : 'Intel Core Ultra 7 265F', cores: multi ? 128 : 20, threads: multi ? 256 : 20, baseGHz: 2.4, maxGHz: 5.3 },
        memory: { totalBytes: 68344610816, modules: ['32 GB · DDR5 · 5600 MT/s · Micron Technology', '32 GB · DDR5 · 5600 MT/s · Micron Technology'] },
        board: 'HP 8D34', system: multi ? 'NVIDIA DGX A100' : 'HP OMEN 16L Gaming Desktop', disks: [{ name: 'SAMSUNG MZVL81T0HFLB', sizeBytes: 1024203640320, type: 'NVMe' }],
        gpus: Array.from({ length: N }, () => ({ name: NAME, vramMiB: TOTAL, driver: '591.74', vbios: '98.06.1f.00.f0', extra: ['NVIDIA', 'PCI'] })),
        cudaVersion: '13.1', remote: 'ssh-remote' },
    });
  }

  const post = (m) => { B.msgs++; window.postMessage(m, '*'); };
  let lastAnalysis = 0;
  function tick() {
    const s = sampleAt(Date.now());
    B.ticks++;
    const m = { type: 'tick', sample: s, point: point(s, selected), selected };
    if (legacy) Object.assign(m, analysisFor(s.t)); // v0.1.0 host: full analysis every tick
    else if (s.t - lastAnalysis >= 2000) { lastAnalysis = s.t; m.analysis = analysisFor(s.t); }
    post(m);
  }

  // ---- emulated extension host ----
  window.acquireVsCodeApi = () => ({
    postMessage(m) {
      if (m.type === 'ready') {
        if (q.has('real')) {
          const s = document.createElement('script');
          s.src = '../dist/preview-data.js';
          s.onload = () => post(window.PREVIEW_INIT);
          document.head.appendChild(s);
          return;
        }
        setTimeout(() => {
          post(initMsg());
          if (!q.has('static')) setInterval(tick, INTERVAL);
        }, 0);
      } else if (m.type === 'selectGpu') {
        selected = m.uuid;
        setTimeout(() => post(initMsg()), 30);
      } else console.log('[to host]', JSON.stringify(m));
    },
    getState() {}, setState() {},
  });

  if (q.has('bench')) {
    const secs = Number(q.get('bench')) || 10;
    // Warm up 1.5 s (initial render and card animations), then count.
    setTimeout(() => {
      for (const k of Object.keys(B)) B[k] = 0;
      new MutationObserver((recs) => (B.mutations += recs.length)).observe(document.body, { subtree: true, childList: true, attributes: true, characterData: true });
      const t0 = performance.now();
      setTimeout(() => {
        const s = (performance.now() - t0) / 1000, per = (v) => Math.round((v / s) * 10) / 10;
        const out = { gpus: N, intervalMs: INTERVAL, legacy, seconds: Math.round(s * 10) / 10, ticksPerSec: per(B.ticks), canvasDrawsPerSec: per(B.draws),
          rafFramesPerSec: per(B.rafFrames), domMutationsPerSec: per(B.mutations), jsMsPerSec: Math.round(((B.handlerMs + B.rafMs) / s) * 100) / 100 };
        const pre = document.createElement('pre');
        pre.id = 'bench-out';
        pre.textContent = JSON.stringify(out);
        document.body.appendChild(pre);
        document.title = 'BENCH ' + pre.textContent;
      }, secs * 1000);
    }, 1500);
  }
})();
