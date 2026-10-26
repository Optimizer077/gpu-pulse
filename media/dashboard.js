// GPU Pulse dashboard (webview). Receives messages from the extension host and renders them.
// Per tick only values that changed touch the DOM; canvases redraw only when their data changed.
(function () {
  'use strict';
  const P = window.Pulse;
  const vscode = typeof acquireVsCodeApi === 'function' ? acquireVsCodeApi() : { postMessage() {}, getState() {}, setState() {} };
  const saved = vscode.getState() || {};

  const state = {
    recent: [], heatmap: null, windows: [], insights: [], grade: null, fleet: [], sample: null,
    specs: null, processes: [], selected: null, config: { tempWarn: 80, intervalMs: 500 },
    vendor: 'none', platform: '', lastTick: 0, alerts: [], range: saved.range || 5 * 60e3, cpuRange: saved.cpuRange || 5 * 60e3,
  };

  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const fmt = (v, d = 0) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
  const gb = (bytes) => (bytes / 1073741824).toFixed(bytes >= 100 * 1073741824 ? 0 : 1);
  const mibGb = (m) => (m == null ? '–' : (m / 1024).toFixed(1));
  const shortName = (n) => String(n || '').replace(/^NVIDIA (GeForce )?/, '');
  const dur = (sec) => {
    const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60);
    return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
  };
  // Write-if-changed helpers: the cached value lives on the element, so unchanged ticks cost no DOM work.
  const txt = (el, v) => { if (el._t !== v) { el._t = v; el.textContent = v; } };
  const html = (el, v) => { if (el._h !== v) { el._h = v; el.innerHTML = v; return true; } return false; };
  const css = (el, prop, v) => { const k = '_s' + prop; if (el[k] !== v) { el[k] = v; el.style[prop] = v; } };

  const ICON = {
    pulse: '<svg viewBox="0 0 40 40" fill="none"><defs><linearGradient id="lg" x1="0" y1="0" x2="40" y2="40"><stop style="stop-color:var(--glow-gpu)"/><stop offset="1" style="stop-color:var(--s-cpu)"/></linearGradient></defs><rect x="2" y="2" width="36" height="36" rx="10" stroke="url(#lg)" stroke-width="2"/><path d="M8 22h6l3-9 5 16 4-11 2 4h4" stroke="url(#lg)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    cpu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx=".8"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>',
    gpu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="2" y="6" width="20" height="11" rx="2"/><circle cx="9" cy="11.5" r="3"/><circle cx="16.5" cy="11.5" r="2"/><path d="M5 17v3M2 9H1"/></svg>',
    mem: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="2" y="7" width="20" height="10" rx="1.5"/><path d="M6 10v4M10 10v4M14 10v4M18 10v4M5 17v2M19 17v2"/></svg>',
    disk: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1"/><path d="M17 18l-2.5-2.5"/></svg>',
    os: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="2" y="4" width="20" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></svg>',
    board: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 7h4v4H7zM14 7h3M14 10h3M7 15h10M7 18h6"/></svg>',
    bolt: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M13 2L4 14h7l-1 8 9-12h-7z"/></svg>',
    clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    remote: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 9l4 3-4 3M10 15h6"/><rect x="2" y="4" width="20" height="16" rx="2"/></svg>',
    good: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M5 12.5l4.5 4.5L19 7"/></svg>',
    warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.5"/></svg>',
    serious: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v6M12 16.5v.5"/></svg>',
    critical: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 2h8l6 6v8l-6 6H8l-6-6V8z"/><path d="M12 7v6M12 16.5v.5"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/></svg>',
  };
  const LEVEL_LABEL = { good: 'good', warning: 'warning', serious: 'serious', critical: 'critical', info: 'note' };
  const RANGES = [[60e3, '1m'], [5 * 60e3, '5m'], [15 * 60e3, '15m'], [60 * 60e3, '60m']];

  function seg(id, current) {
    return `<div class="seg" role="group" aria-label="Time range" id="${id}">${RANGES.map(([ms, l]) =>
      `<button data-ms="${ms}" aria-pressed="${ms === current}">${l}</button>`).join('')}</div>`;
  }

  function gaugeHtml(id, label, big) {
    return `<div class="gauge${big ? ' big' : ''}"><canvas id="g-${id}" aria-hidden="true"></canvas>
      <div class="read"><div class="val" id="v-${id}">–</div><div class="lbl" id="l-${id}">${label}</div><div class="extra" id="x-${id}"></div>${big ? '<div id="grade"></div>' : ''}</div></div>`;
  }

  function shell() {
    $('#app').innerHTML = `
    <header class="top">
      <div class="brand">${ICON.pulse}<div><h1>GPU PULSE</h1><small id="host">—</small></div></div>
      <div class="spacer"></div>
      <span class="chip" id="remote" hidden>${ICON.remote.replace('<svg', '<svg width="13" height="13"')}<span></span></span>
      <span class="chip alert-chip" id="alerts" role="status" hidden></span>
      <span class="chip"><span class="live-dot" id="live"></span><span id="rate">live</span></span>
      <button class="btn" data-cmd="gpuPulse.exportHistory" title="Export per-minute history as CSV">Export CSV</button>
      <button class="btn" data-cmd="gpuPulse.clearHistory" title="Delete recorded history">Clear</button>
    </header>
    <main class="grid">
      <section class="card fleet" id="fleet" hidden style="animation-delay:0s">
        <h2>All GPUs <span class="sub" id="fleet-sub"></span><span class="right"><span class="sub hint">click or use ← → to choose the detailed view</span></span></h2>
        <div class="fleet-row" id="fleet-row" role="radiogroup" aria-label="GPU shown in detail"></div>
      </section>
      <section class="card" style="animation-delay:.02s">
        <h2>Live <span class="sub" id="gpu-name">detecting GPU…</span><span class="right"><span class="sub" id="pstate"></span></span></h2>
        <div class="hero">
          ${gaugeHtml('util', 'GPU utilization', true)}
          ${gaugeHtml('vram', 'VRAM')}
          ${gaugeHtml('temp', 'Temperature')}
          ${gaugeHtml('power', 'Power')}
        </div>
        <div class="readouts" id="readouts"></div>
      </section>

      <section class="card span-8 timeline" style="animation-delay:.08s">
        <h2>GPU timeline <span class="right"><span class="legend" id="lg-gpu"></span>${seg('rng-gpu', state.range)}</span></h2>
        <div class="chart"><canvas id="c-gpu" role="img" aria-label="GPU utilization and VRAM over time"></canvas></div>
      </section>
      <section class="card span-4" style="animation-delay:.12s">
        <h2>Insights <span class="sub">last hour</span></h2>
        <div class="insights" id="insights" aria-live="polite"></div>
      </section>

      <section class="card" style="animation-delay:.16s">
        <h2>Usage over time <span class="sub">mean GPU utilization · share of time idle (&lt;10%) / active / busy (≥70%)</span></h2>
        <div class="windows" id="windows"></div>
      </section>

      <section class="card span-7" style="animation-delay:.2s">
        <h2>24-hour activity <span class="sub">one cell per minute</span><span class="right"><span class="scale">0%<i></i>100%</span></span></h2>
        <div class="heat"><canvas id="c-heat" role="img" aria-label="Per-minute GPU utilization over the last 24 hours"></canvas></div>
      </section>
      <section class="card span-5" style="animation-delay:.24s">
        <h2>CPU &amp; memory <span class="right"><span class="legend" id="lg-cpu"></span>${seg('rng-cpu', state.cpuRange)}</span></h2>
        <div class="chart short"><canvas id="c-cpu" role="img" aria-label="CPU and RAM usage over time"></canvas></div>
        <div class="cores" id="cores" aria-label="Per-core CPU load"></div>
        <div class="mem-row"><span>RAM</span><span id="mem-txt">–</span></div>
        <div class="mem-bar"><div id="mem-bar" style="width:0"></div></div>
      </section>

      <section class="card" style="animation-delay:.28s">
        <h2>Machine</h2>
        <div class="specs" id="specs">${'<div class="skeleton"></div>'.repeat(6)}</div>
      </section>
      <section class="card" style="animation-delay:.32s">
        <h2>GPU processes <span class="sub" id="proc-sub"></span></h2>
        <div class="scroll" id="procs"><div class="empty">Waiting for data…</div></div>
      </section>
    </main>
    <div class="foot" id="foot">History is recorded while VS Code is running and kept on this machine.</div>`;
  }

  shell();
  const E = {};
  for (const id of ['host', 'remote', 'alerts', 'live', 'rate', 'fleet', 'fleet-sub', 'fleet-row', 'gpu-name', 'pstate', 'readouts', 'grade',
    'v-util', 'x-util', 'v-vram', 'x-vram', 'l-vram', 'v-temp', 'x-temp', 'v-power', 'x-power', 'lg-gpu', 'lg-cpu', 'insights', 'windows',
    'cores', 'mem-txt', 'mem-bar', 'specs', 'proc-sub', 'procs']) E[id] = document.getElementById(id);

  const tip = P.makeTip();
  const gauges = {
    util: new P.Gauge($('#g-util'), { big: true, ticks: 50 }),
    vram: new P.Gauge($('#g-vram'), { color: 's-vram', glow: 'glow-vram', ticks: 30 }),
    temp: new P.Gauge($('#g-temp'), { color: 's-ram', glow: 's-ram', max: 100, ticks: 30 }),
    power: new P.Gauge($('#g-power'), { color: 's-cpu', glow: 's-cpu', max: 100, ticks: 30 }),
  };
  const gpuChart = new P.LineChart($('#c-gpu'), {
    tip, rangeMs: state.range,
    series: [
      { idx: 2, color: 's-vram', glow: 'glow-vram', label: 'VRAM', fill: false },
      { idx: 1, color: 's-gpu', glow: 'glow-gpu', label: 'GPU util', fill: true },
    ],
  });
  const cpuChart = new P.LineChart($('#c-cpu'), {
    tip, rangeMs: state.cpuRange,
    series: [
      { idx: 4, color: 's-ram', label: 'RAM', fill: false },
      { idx: 3, color: 's-cpu', label: 'CPU', fill: true },
    ],
  });
  const heat = new P.Heatmap($('#c-heat'), tip);

  // ---- selection ----
  function gpus() { return state.sample?.gpus || []; }
  function currentGpu() {
    const gs = gpus();
    return gs.find((g) => g.uuid === state.selected) || gs[0];
  }
  // Ticks already in flight still carry the old selection; ignore it until the host's init confirms.
  let pendingSel = null;
  /** Switch the detailed view at once from the latest sample; the host follows up with that GPU's history. */
  function select(uuid) {
    if (!uuid || uuid === currentGpu()?.uuid) return;
    state.selected = pendingSel = uuid;
    renderFleetSelection(); renderLive(); renderProcs();
    vscode.postMessage({ type: 'selectGpu', uuid });
  }

  // ---- events ----
  document.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.cmd) vscode.postMessage({ type: 'command', command: b.dataset.cmd });
    if (b.dataset.gpu) select(b.dataset.gpu);
    if (b.dataset.ms) {
      const group = b.parentElement;
      const ms = Number(b.dataset.ms);
      group.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      if (group.id === 'rng-gpu') { state.range = ms; gpuChart.setRange(ms); }
      else { state.cpuRange = ms; cpuChart.setRange(ms); }
      vscode.setState({ range: state.range, cpuRange: state.cpuRange });
    }
  });
  // Radio-group keyboard pattern: arrows move focus and selection, Home/End jump.
  E['fleet-row'].addEventListener('keydown', (e) => {
    const btns = [...E['fleet-row'].querySelectorAll('button[data-gpu]')];
    const i = btns.indexOf(document.activeElement);
    if (i < 0) return;
    const next = { ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: btns.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    const b = btns[(next + btns.length) % btns.length];
    b.focus();
    select(b.dataset.gpu);
  });
  // Core tooltips are filled on hover instead of rewriting 20+ title attributes every tick.
  E.cores.addEventListener('mouseover', (e) => {
    const c = e.target.closest('.core');
    if (c) c.title = `core ${[...E.cores.children].indexOf(c)}: ${c._v ?? 0}%`;
  });
  new MutationObserver(() => { P.readTokens(); redrawAll(); }).observe(document.body, { attributes: true, attributeFilter: ['class'] });

  function redrawAll() {
    Object.values(gauges).forEach((g) => g.draw());
    for (const c of fleetCards.values()) c.gauge.draw();
    gpuChart.draw(); cpuChart.draw(); heat.draw();
    renderWindows(true);
  }

  // ---- all-GPUs strip (only with 2+ GPUs) ----
  const fleetCards = new Map();
  let fleetKey = '';
  function buildFleet(gs) {
    E['fleet-row'].innerHTML = gs.map((g) => `<button class="gcard" role="radio" aria-checked="false" tabindex="-1" data-gpu="${esc(g.uuid)}">
        <span class="gc-head"><span class="gc-idx">GPU ${g.index}</span><span class="gc-name" title="${esc(g.name)}">${esc(shortName(g.name))}</span></span>
        <span class="gc-body">
          <span class="gc-gauge"><canvas aria-hidden="true"></canvas><span class="gc-val"><b data-f="util">–</b><small>%</small></span></span>
          <span class="gc-stats">
            <span class="gc-k"><span>VRAM</span><b data-f="vram">–</b></span>
            <span class="gc-bar"><span data-f="vbar"></span></span>
            <span class="gc-k"><span>Temp</span><b data-f="temp">–</b></span>
            <span class="gc-k"><span>Power</span><b data-f="power">–</b></span>
            <span class="gc-k"><span>10 min avg</span><b data-f="avg">–</b></span>
          </span>
        </span></button>`).join('');
    fleetCards.clear();
    for (const el of E['fleet-row'].children) {
      const f = {};
      el.querySelectorAll('[data-f]').forEach((x) => (f[x.dataset.f] = x));
      fleetCards.set(el.dataset.gpu, { el, f, gauge: new P.Gauge(el.querySelector('canvas'), { mini: true, ticks: 0 }) });
    }
  }

  function renderFleet() {
    const gs = gpus();
    const multi = gs.length > 1;
    E.fleet.hidden = !multi;
    if (!multi) {
      if (fleetKey) { E['fleet-row'].innerHTML = ''; fleetCards.clear(); fleetKey = ''; }
      return;
    }
    const key = gs.map((g) => `${g.uuid}|${g.index}|${g.name}`).join(',');
    if (key !== fleetKey) { fleetKey = key; buildFleet(gs); renderFleetSelection(); }
    const avg = new Map((state.fleet || []).map((f) => [f.uuid, f.util]));
    const warn = state.config.tempWarn;
    let sum = 0, n = 0, watts = 0;
    for (const g of gs) {
      const c = fleetCards.get(g.uuid);
      if (!c) continue;
      if (g.util != null) { sum += g.util; n++; }
      watts += g.powerW || 0;
      c.gauge.set(g.util);
      txt(c.f.util, fmt(g.util));
      const vp = g.memTotalMiB ? (g.memUsedMiB / g.memTotalMiB) * 100 : null;
      txt(c.f.vram, `${mibGb(g.memUsedMiB)} / ${mibGb(g.memTotalMiB)} GB`);
      css(c.f.vbar, 'width', `${vp == null ? 0 : Math.round(vp)}%`);
      txt(c.f.temp, g.tempC == null ? '–' : `${fmt(g.tempC)} °C`);
      txt(c.f.power, g.powerW == null ? '–' : `${fmt(g.powerW)} W`);
      const a = avg.get(g.uuid);
      txt(c.f.avg, a == null ? '–' : `${fmt(a)}%`);
      const hot = g.tempC != null && g.tempC >= warn;
      if (c.hot !== hot) { c.hot = hot; c.el.classList.toggle('hot', hot); }
    }
    txt(E['fleet-sub'], `${gs.length} GPUs · mean ${n ? fmt(sum / n) : '–'}% · ${fmt(watts)} W total`);
  }

  function renderFleetSelection() {
    const cur = currentGpu()?.uuid;
    for (const [uuid, c] of fleetCards) {
      const on = uuid === cur;
      c.el.setAttribute('aria-checked', String(on));
      c.el.tabIndex = on ? 0 : -1;
    }
  }

  // ---- renderers ----
  function renderLive() {
    const s = state.sample;
    if (!s) return;
    const g = currentGpu();
    const hour = state.windows[2];
    const apple = g?.vendor === 'apple' || (!g && state.vendor === 'apple');
    $('.hero').classList.toggle('metal', apple);
    $('#g-vram').parentElement.classList.toggle('absolute', apple);
    $('#g-temp').parentElement.hidden = apple;
    $('#g-power').parentElement.hidden = apple;
    E.readouts.hidden = apple;
    const chartSeries = gpuChart.opt.series;
    if (apple && chartSeries.length > 1) {
      gpuChart.opt.series = chartSeries.filter((s) => s.idx !== 2);
      gpuChart.requestDraw();
    } else if (!apple && chartSeries.length === 1) {
      gpuChart.opt.series = [{ idx: 2, color: 's-vram', glow: 'glow-vram', label: 'VRAM', fill: false }, ...chartSeries];
      gpuChart.requestDraw();
    }
    if (g) {
      txt(E['gpu-name'], `${g.name}${gpus().length > 1 ? ` · GPU ${g.index}` : ''}`);
      txt(E.pstate, [g.vendor === 'apple' ? 'Metal · unified memory' : null, g.pstate, g.pcieGen ? `PCIe ${g.pcieGen}.0 x${g.pcieWidth}` : null].filter(Boolean).join(' · '));
      gauges.util.set(g.util, hour?.util);
      html(E['v-util'], `${fmt(g.util)}<sup>%</sup>`);
      txt(E['x-util'], hour?.util != null ? `▲ 1h avg ${fmt(hour.util)}%` : '');
      const vp = g.memTotalMiB ? (g.memUsedMiB / g.memTotalMiB) * 100 : null;
      gauges.vram.set(vp);
      txt(E['l-vram'], g.vendor === 'apple' ? 'GPU memory' : 'VRAM');
      html(E['v-vram'], g.vendor === 'apple' ? `${mibGb(g.memUsedMiB)}<sup>GB</sup>` : `${fmt(vp)}<sup>%</sup>`);
      txt(E['x-vram'], g.vendor === 'apple' ? `${mibGb(g.memUsedMiB)} GB shared · GPU allocation` : `${mibGb(g.memUsedMiB)} / ${mibGb(g.memTotalMiB)} GB`);
      if (gauges.temp.opt.warn !== state.config.tempWarn) { gauges.temp.opt.warn = state.config.tempWarn; gauges.temp.draw(); }
      gauges.temp.set(g.tempC);
      html(E['v-temp'], `${fmt(g.tempC)}<sup>°C</sup>`);
      txt(E['x-temp'], hour?.tempPeak ? `1h peak ${fmt(hour.tempPeak)}°C` : '');
      gauges.power.setMax(g.powerLimitW || 100);
      gauges.power.set(g.powerW);
      html(E['v-power'], `${fmt(g.powerW)}<sup>W</sup>`);
      txt(E['x-power'], g.powerLimitW ? `limit ${fmt(g.powerLimitW)} W` : '');
      const ro = [
        ['Core clock', g.clockMHz != null ? `${fmt(g.clockMHz)} MHz` : '–', g.clockMaxMHz ? g.clockMHz / g.clockMaxMHz : null],
        ['Max clock', g.clockMaxMHz != null ? `${fmt(g.clockMaxMHz)} MHz` : '–', null],
        ['Mem clock', g.memClockMHz != null ? `${fmt(g.memClockMHz)} MHz` : '–', null],
        ['Mem bus busy', g.memUtil != null ? `${fmt(g.memUtil)}%` : '–', g.memUtil != null ? g.memUtil / 100 : null],
        ['Fan', g.fanPct != null ? `${fmt(g.fanPct)}%` : '–', g.fanPct != null ? g.fanPct / 100 : null],
        ['Power use', g.powerLimitW ? `${fmt((g.powerW / g.powerLimitW) * 100)}% of cap` : '–', g.powerLimitW ? g.powerW / g.powerLimitW : null],
      ];
      html(E.readouts, ro.map(([k, v, f]) =>
        `<div class="ro"><span>${k}</span><b>${esc(v)}</b>${f != null ? `<div class="bar"><div style="width:${Math.round(Math.max(0, Math.min(100, f * 100)))}%"></div></div>` : ''}</div>`).join(''));
      const gr = state.grade;
      html(E.grade, gr ? `<span class="grade ${gr.level}" title="Based on the last hour's mean utilization"><i></i>${esc(gr.label)}</span>` : '');
    } else {
      txt(E['gpu-name'], state.vendor === 'none' ? 'no supported GPU found' : 'reading…');
    }
    // CPU / RAM
    const cores = s.cpu.perCore;
    const host = E.cores;
    if (host.childElementCount !== cores.length) {
      host.style.gridTemplateColumns = `repeat(${cores.length}, minmax(0, 1fr))`;
      host.innerHTML = cores.map((_, i) => `<div class="core" title="core ${i}"><div></div></div>`).join('');
    }
    cores.forEach((v, i) => {
      const c = host.children[i], p = Math.round(v);
      if (c._v !== p) { c._v = p; c.firstChild.style.height = `${p}%`; }
    });
    const mp = (s.mem.usedBytes / s.mem.totalBytes) * 100;
    css(E['mem-bar'], 'width', `${mp.toFixed(1)}%`);
    txt(E['mem-txt'], `${gb(s.mem.usedBytes)} / ${gb(s.mem.totalBytes)} GB · ${mp.toFixed(0)}%`);

    const last = state.recent[state.recent.length - 1];
    html(E['lg-gpu'], legend(apple ? [['GPU util', 's-gpu', last?.[1]]]
      : [['GPU util', 's-gpu', last?.[1]], ['VRAM', 's-vram', last?.[2]]]));
    html(E['lg-cpu'], legend([['CPU', 's-cpu', last?.[3]], ['RAM', 's-ram', last?.[4]]]));
  }

  function legend(items) {
    return items.map(([l, c, v]) => `<span class="k"><i style="background:var(--${c})"></i>${l} <b>${v == null ? '–' : fmt(v)}%</b></span>`).join('');
  }

  /** Rebuilds the cards only when their text changed; sparklines follow the heatmap (per minute) or a theme change. */
  function renderWindows(redrawSparks) {
    if (!state.windows.length) return;
    const markup = state.windows.map((w, i) => {
      const apple = currentGpu()?.vendor === 'apple' || state.vendor === 'apple';
      const has = w.util != null;
      const idle = w.idleShare ?? 0, busy = w.busyShare ?? 0, mid = Math.max(0, 1 - idle - busy);
      return `<div class="win">
        <div class="head"><span>${esc(w.label)}</span><em>${w.coveredMinutes}/${w.minutes} min recorded</em></div>
        <div class="big">${has ? fmt(w.util) : '–'}<small>% avg</small></div>
        <div class="what">${has ? `peak ${fmt(w.utilPeak)}%${apple ? '' : ` · VRAM avg ${fmt(w.vram)}%`}` : 'Collecting GPU history…'}</div>
        <div class="stack" role="img" aria-label="idle ${Math.round(idle * 100)}%, active ${Math.round(mid * 100)}%, busy ${Math.round(busy * 100)}%">
          <div class="idle" style="flex-grow:${idle.toFixed(3)}"></div><div class="mid" style="flex-grow:${mid.toFixed(3)}"></div><div class="busy" style="flex-grow:${busy.toFixed(3)}"></div></div>
        <div class="stack-l"><span>idle ${has ? Math.round(idle * 100) : '–'}%</span><span>busy ${has ? Math.round(busy * 100) : '–'}%</span></div>
        <dl class="kv">
          ${apple ? '' : `<dt>Avg power</dt><dd>${w.power ? `${fmt(w.power)} W` : '–'}</dd>
          <dt>Peak temp</dt><dd>${w.tempPeak ? `${fmt(w.tempPeak)} °C` : '–'}</dd>
          <dt>VRAM held, GPU idle</dt><dd>${w.heldIdleShare != null ? `${Math.round(w.heldIdleShare * 100)}%` : '–'}</dd>`}
          <dt>CPU / RAM avg</dt><dd>${fmt(w.cpu)}% / ${fmt(w.mem)}%</dd>
        </dl>
        <canvas class="spark" data-i="${i}" aria-hidden="true"></canvas>
      </div>`;
    }).join('');
    if (html(E.windows, markup) || redrawSparks) drawSparks();
  }

  function drawSparks() {
    const cells = state.heatmap?.cells || [];
    E.windows.querySelectorAll('canvas.spark').forEach((cv) => {
      const w = state.windows[Number(cv.dataset.i)];
      if (!w) return;
      let vals = cells.slice(-w.minutes);
      if (vals.length > 120) { // downsample long windows to ~120 points
        const k = Math.ceil(vals.length / 120), out = [];
        for (let j = 0; j < vals.length; j += k) {
          const chunk = vals.slice(j, j + k).filter((v) => v != null);
          out.push(chunk.length ? chunk.reduce((a, b) => a + b, 0) / chunk.length : null);
        }
        vals = out;
      }
      P.spark(cv, vals, 's-gpu');
    });
  }

  function renderInsights() {
    // The list is aria-live: only touch the DOM when the content changes, or screen readers re-announce.
    html(E.insights, state.insights.length
      ? state.insights.map((x) => `<div class="ins ${x.level}"><div class="ic">${ICON[x.level] || ICON.info}</div>
          <div><b>${esc(x.title)}<span class="lv">${LEVEL_LABEL[x.level] || ''}</span></b><p>${esc(x.detail)}</p></div></div>`).join('')
      : `<div class="ins good"><div class="ic">${ICON.good}</div><div><b>All clear<span class="lv">good</span></b><p>No issues detected in the last hour.</p></div></div>`);
  }

  function renderProcs() {
    const apple = currentGpu()?.vendor === 'apple' || state.vendor === 'apple';
    E.procs.closest('.card').hidden = apple;
    if (apple) return;
    const g = currentGpu();
    const list = state.processes.filter((p) => !g || p.gpuUuid === g.uuid)
      .sort((a, b) => (b.usedMiB ?? -1) - (a.usedMiB ?? -1));
    const noMem = list.length && list.every((p) => p.usedMiB == null);
    txt(E['proc-sub'], list.length ? `${list.length} using this GPU${noMem && state.platform === 'win32' ? ' · per-process VRAM is not reported by Windows (WDDM)' : ''}` : '');
    if (state.vendor !== 'nvidia') {
      html(E.procs, `<div class="empty">${state.vendor === 'apple' ? 'Per-process GPU usage is unavailable from macOS IORegistry.' : 'Process listing needs an NVIDIA GPU with nvidia-smi.'}</div>`);
      return;
    }
    html(E.procs, list.length
      ? `<table><thead><tr><th>Process</th><th class="n">PID</th><th class="n">VRAM</th></tr></thead><tbody>${
        list.map((p) => `<tr><td class="name" title="${esc(p.name)}">${esc(p.name)}</td><td class="n">${p.pid}</td><td class="n">${p.usedMiB == null ? 'n/a' : `${fmt(p.usedMiB)} MiB`}</td></tr>`).join('')}</tbody></table>`
      : '<div class="empty">No compute processes on this GPU.</div>');
  }

  function renderSpecs() {
    const sp = state.specs;
    if (!sp) return;
    txt(E.host, `${sp.hostname} · ${sp.os}`);
    const rm = E.remote;
    if (sp.remote) { rm.hidden = false; rm.querySelector('span').textContent = sp.remote; }
    const cards = [
      [ICON.cpu, 'Processor', sp.cpu.model, `${sp.cpu.cores} cores · ${sp.cpu.threads} threads${sp.cpu.maxGHz && sp.cpu.maxGHz > (sp.cpu.baseGHz || 0) ? ` · up to ${sp.cpu.maxGHz} GHz` : sp.cpu.baseGHz ? ` · base ${sp.cpu.baseGHz} GHz` : ''}`],
      ...sp.gpus.map((g) => [ICON.gpu, 'Graphics', g.name,
        [g.vramMiB ? `${(g.vramMiB / 1024).toFixed(0)} GB VRAM` : null,
          ...(g.extra || []).filter((v) => /Metal|unified/i.test(v)),
          g.driver ? `driver ${g.driver}` : null, g.vbios ? `VBIOS ${g.vbios}` : null].filter(Boolean).join(' · ')]),
      [ICON.mem, 'Memory', `${gb(sp.memory.totalBytes)} GB`, sp.memory.modules.join('\n') || ''],
      ...sp.disks.map((d) => [ICON.disk, 'Storage', d.name, `${(d.sizeBytes / 1e12 >= 1 ? (d.sizeBytes / 1e12).toFixed(1) + ' TB' : (d.sizeBytes / 1e9).toFixed(0) + ' GB')} · ${d.type || ''}`]),
      [ICON.os, 'Operating system', sp.os, `kernel ${sp.kernel} · ${sp.arch}`],
      sp.system || sp.board ? [ICON.board, 'System', sp.system || sp.board, sp.system && sp.board ? `board ${sp.board}` : ''] : null,
      sp.cudaVersion ? [ICON.bolt, 'CUDA', `CUDA ${sp.cudaVersion}`, 'max version supported by the driver'] : null,
      [ICON.clock, 'Uptime', dur(sp.uptimeSec), `${sp.hostname}`],
    ].filter(Boolean);
    html(E.specs, cards.map(([ic, k, v, d]) =>
      `<div class="spec"><div class="ic">${ic}</div><div><span>${esc(k)}</span><b>${esc(v)}</b>${d ? `<p>${esc(d).replace(/\n/g, '<br>')}</p>` : ''}</div></div>`).join(''));
  }

  function renderRate() {
    const sec = state.config.intervalMs / 1000;
    txt(E.rate, `live · ${sec < 1 ? sec.toFixed(2) : sec.toFixed(1)}s`);
    // Gauge tweens finish before the next sample, so the animation loop idles between ticks.
    P.setAnimMs(state.config.intervalMs * 0.6);
  }

  // Active alerts (conditions still holding): a small header badge; the titles are in its tooltip.
  function renderAlerts() {
    const a = state.alerts || [];
    if (E.alerts.hidden !== !a.length) E.alerts.hidden = !a.length;
    const warn = a.some((x) => x.level === 'warning');
    E.alerts.classList.toggle('warn', warn);
    // Alert text only goes through DOM properties (title / aria-label), never innerHTML.
    E.alerts.title = a.map((x) => `${x.title}. ${x.detail}`).join('\n');
    E.alerts.setAttribute('aria-label', a.length ? `${a.length} active GPU alert${a.length > 1 ? 's' : ''}: ${a.map((x) => x.title).join('; ')}` : '');
    html(E.alerts, a.length ? `${ICON[warn ? 'warning' : 'info'].replace('<svg', '<svg width="13" height="13" aria-hidden="true"')}<span>${a.length} alert${a.length > 1 ? 's' : ''}</span>` : '');
  }

  function renderAll() {
    gpuChart.setData(state.recent); cpuChart.setData(state.recent); heat.set(state.heatmap);
    renderFleet(); renderFleetSelection(); renderLive(); renderWindows(true); renderInsights(); renderProcs(); renderSpecs(); renderRate(); renderAlerts();
  }

  // ---- liveness dot; paused with everything else while the document is hidden ----
  let staleTimer = 0;
  function checkStale() {
    const stale = Date.now() - state.lastTick > Math.max(4000, state.config.intervalMs * 3);
    if (E.live._stale !== stale) { E.live._stale = stale; E.live.classList.toggle('stale', stale); }
  }
  function startStaleTimer() { if (!staleTimer) staleTimer = setInterval(checkStale, 1000); }
  let dirty = false;
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { clearInterval(staleTimer); staleTimer = 0; return; }
    startStaleTimer();
    if (dirty) { dirty = false; renderAll(); }
  });
  if (!document.hidden) startStaleTimer();

  function applyAnalysis(a) {
    if (!a) return;
    state.windows = a.windows || [];
    state.insights = a.insights || [];
    state.grade = a.grade;
    state.fleet = a.fleet || [];
  }

  // ---- messages: state is always updated; rendering waits while the document is hidden ----
  window.addEventListener('message', (ev) => {
    const m = ev.data;
    switch (m.type) {
      case 'init':
        pendingSel = null;
        Object.assign(state, {
          recent: m.recent || [], heatmap: m.heatmap, sample: m.sample, specs: m.specs, processes: m.processes || [],
          selected: m.selected, config: m.config, vendor: m.vendor, platform: m.platform, lastTick: Date.now(), alerts: m.alerts || [],
        });
        applyAnalysis(m.analysis || m);
        if (document.hidden) dirty = true; else renderAll();
        break;
      case 'tick': {
        const prevSel = state.selected;
        Object.assign(state, { sample: m.sample, selected: pendingSel || m.selected, lastTick: Date.now() });
        if (m.analysis && !pendingSel) applyAnalysis(m.analysis);
        if (m.alerts) state.alerts = m.alerts;
        if (m.point && !pendingSel) {
          state.recent.push(m.point);
          const cutoff = m.point[0] - 60 * 60e3;
          let drop = 0;
          while (drop < state.recent.length && state.recent[drop][0] < cutoff) drop++;
          if (drop) state.recent.splice(0, drop);
        }
        if (document.hidden) { dirty = true; break; }
        gpuChart.setData(state.recent); cpuChart.setData(state.recent);
        renderFleet();
        if (prevSel !== state.selected) renderFleetSelection();
        renderLive();
        if (m.analysis) { renderWindows(false); renderInsights(); }
        if (m.alerts) renderAlerts();
        if (E.live._stale) checkStale();
        break;
      }
      case 'heatmap':
        state.heatmap = m.heatmap;
        if (document.hidden) dirty = true; else { heat.set(m.heatmap); drawSparks(); }
        break;
      case 'processes':
        state.processes = m.processes;
        if (document.hidden) dirty = true; else renderProcs();
        break;
      case 'specs':
        state.specs = m.specs;
        if (document.hidden) dirty = true; else renderSpecs();
        break;
    }
  });

  vscode.postMessage({ type: 'ready' });
})();
