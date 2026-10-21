// GPU Pulse dashboard (webview). Receives messages from the extension host and renders them.
(function () {
  'use strict';
  const P = window.Pulse;
  const vscode = typeof acquireVsCodeApi === 'function' ? acquireVsCodeApi() : { postMessage() {}, getState() {}, setState() {} };
  const saved = vscode.getState() || {};

  const state = {
    recent: [], heatmap: null, windows: [], insights: [], grade: null, sample: null,
    specs: null, processes: [], selected: null, config: { tempWarn: 80, intervalMs: 1000 },
    vendor: 'none', platform: '', lastTick: 0, range: saved.range || 5 * 60e3, cpuRange: saved.cpuRange || 5 * 60e3,
  };

  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const fmt = (v, d = 0) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
  const gb = (bytes) => (bytes / 1073741824).toFixed(bytes >= 100 * 1073741824 ? 0 : 1);
  const mibGb = (m) => (m == null ? '–' : (m / 1024).toFixed(1));
  const dur = (sec) => {
    const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60);
    return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
  };

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
      <div class="read"><div class="val" id="v-${id}">–</div><div class="lbl">${label}</div><div class="extra" id="x-${id}"></div>${big ? '<div id="grade"></div>' : ''}</div></div>`;
  }

  function shell() {
    $('#app').innerHTML = `
    <header class="top">
      <div class="brand">${ICON.pulse}<div><h1>GPU PULSE</h1><small id="host">—</small></div></div>
      <div class="spacer"></div>
      <span class="chip" id="remote" hidden>${ICON.remote.replace('<svg', '<svg width="13" height="13"')}<span></span></span>
      <span class="chip"><span class="live-dot" id="live"></span><span id="rate">live</span></span>
      <button class="btn" data-cmd="gpuPulse.exportHistory" title="Export per-minute history as CSV">Export CSV</button>
      <button class="btn" data-cmd="gpuPulse.clearHistory" title="Delete recorded history">Clear</button>
    </header>
    <nav class="tabs" id="tabs" role="tablist" hidden></nav>
    <main class="grid">
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

      <section class="card span-8" style="animation-delay:.08s">
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

  // ---- events ----
  document.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.cmd) vscode.postMessage({ type: 'command', command: b.dataset.cmd });
    if (b.dataset.gpu) vscode.postMessage({ type: 'selectGpu', uuid: b.dataset.gpu });
    if (b.dataset.ms) {
      const group = b.parentElement;
      const ms = Number(b.dataset.ms);
      group.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      if (group.id === 'rng-gpu') { state.range = ms; gpuChart.setRange(ms); }
      else { state.cpuRange = ms; cpuChart.setRange(ms); }
      vscode.setState({ range: state.range, cpuRange: state.cpuRange });
    }
  });
  new MutationObserver(() => { P.readTokens(); redrawAll(); }).observe(document.body, { attributes: true, attributeFilter: ['class'] });

  function redrawAll() {
    Object.values(gauges).forEach((g) => g.draw());
    gpuChart.draw(); cpuChart.draw(); heat.draw();
    renderWindows();
  }

  // ---- renderers ----
  function currentGpu() {
    const gs = state.sample?.gpus || [];
    return gs.find((g) => g.uuid === state.selected) || gs[0];
  }

  function renderLive() {
    const s = state.sample;
    if (!s) return;
    const g = currentGpu();
    const hour = state.windows[2];
    if (g) {
      $('#gpu-name').textContent = `${g.name}${state.sample.gpus.length > 1 ? ` · GPU ${g.index}` : ''}`;
      $('#pstate').textContent = [g.pstate, g.pcieGen ? `PCIe ${g.pcieGen}.0 x${g.pcieWidth}` : null].filter(Boolean).join(' · ');
      gauges.util.set(g.util, hour?.util);
      $('#v-util').innerHTML = `${fmt(g.util)}<sup>%</sup>`;
      $('#x-util').textContent = hour?.util != null ? `▲ 1h avg ${fmt(hour.util)}%` : '';
      const vp = g.memTotalMiB ? (g.memUsedMiB / g.memTotalMiB) * 100 : null;
      gauges.vram.set(vp);
      $('#v-vram').innerHTML = `${fmt(vp)}<sup>%</sup>`;
      $('#x-vram').textContent = `${mibGb(g.memUsedMiB)} / ${mibGb(g.memTotalMiB)} GB`;
      gauges.temp.opt.warn = state.config.tempWarn;
      gauges.temp.set(g.tempC);
      $('#v-temp').innerHTML = `${fmt(g.tempC)}<sup>°C</sup>`;
      $('#x-temp').textContent = hour?.tempPeak ? `1h peak ${fmt(hour.tempPeak)}°C` : '';
      gauges.power.setMax(g.powerLimitW || 100);
      gauges.power.set(g.powerW);
      $('#v-power').innerHTML = `${fmt(g.powerW)}<sup>W</sup>`;
      $('#x-power').textContent = g.powerLimitW ? `limit ${fmt(g.powerLimitW)} W` : '';
      const ro = [
        ['Core clock', g.clockMHz != null ? `${fmt(g.clockMHz)} MHz` : '–', g.clockMaxMHz ? g.clockMHz / g.clockMaxMHz : null],
        ['Max clock', g.clockMaxMHz != null ? `${fmt(g.clockMaxMHz)} MHz` : '–', null],
        ['Mem clock', g.memClockMHz != null ? `${fmt(g.memClockMHz)} MHz` : '–', null],
        ['Mem bus busy', g.memUtil != null ? `${fmt(g.memUtil)}%` : '–', g.memUtil != null ? g.memUtil / 100 : null],
        ['Fan', g.fanPct != null ? `${fmt(g.fanPct)}%` : '–', g.fanPct != null ? g.fanPct / 100 : null],
        ['Power use', g.powerLimitW ? `${fmt((g.powerW / g.powerLimitW) * 100)}% of cap` : '–', g.powerLimitW ? g.powerW / g.powerLimitW : null],
      ];
      $('#readouts').innerHTML = ro.map(([k, v, f]) =>
        `<div class="ro"><span>${k}</span><b>${esc(v)}</b>${f != null ? `<div class="bar"><div style="width:${Math.max(0, Math.min(100, f * 100))}%"></div></div>` : ''}</div>`).join('');
      const gr = state.grade;
      $('#grade').innerHTML = gr ? `<span class="grade ${gr.level}" title="Based on the last hour's mean utilization"><i></i>${esc(gr.label)}</span>` : '';
    } else {
      $('#gpu-name').textContent = state.vendor === 'none' ? 'no supported GPU found' : 'reading…';
    }
    // CPU / RAM
    const cores = s.cpu.perCore;
    const host = $('#cores');
    if (host.childElementCount !== cores.length) {
      host.style.gridTemplateColumns = `repeat(${cores.length}, minmax(0, 1fr))`;
      host.innerHTML = cores.map((_, i) => `<div class="core" title="core ${i}"><div></div></div>`).join('');
    }
    cores.forEach((v, i) => { host.children[i].firstChild.style.height = `${v}%`; host.children[i].title = `core ${i}: ${v.toFixed(0)}%`; });
    const mp = (s.mem.usedBytes / s.mem.totalBytes) * 100;
    $('#mem-bar').style.width = `${mp}%`;
    $('#mem-txt').textContent = `${gb(s.mem.usedBytes)} / ${gb(s.mem.totalBytes)} GB · ${mp.toFixed(0)}%`;

    const last = state.recent[state.recent.length - 1];
    $('#lg-gpu').innerHTML = legend([['GPU util', 's-gpu', last?.[1]], ['VRAM', 's-vram', last?.[2]]]);
    $('#lg-cpu').innerHTML = legend([['CPU', 's-cpu', last?.[3]], ['RAM', 's-ram', last?.[4]]]);
  }

  function legend(items) {
    return items.map(([l, c, v]) => `<span class="k"><i style="background:var(--${c})"></i>${l} <b>${v == null ? '–' : fmt(v)}%</b></span>`).join('');
  }

  function renderWindows() {
    const host = $('#windows');
    if (!state.windows.length) return;
    const cells = state.heatmap?.cells || [];
    host.innerHTML = state.windows.map((w, i) => {
      const has = w.util != null;
      const idle = w.idleShare ?? 0, busy = w.busyShare ?? 0, mid = Math.max(0, 1 - idle - busy);
      return `<div class="win">
        <div class="head"><span>${esc(w.label)}</span><em>${w.coveredMinutes}/${w.minutes} min recorded</em></div>
        <div class="big">${has ? fmt(w.util) : '–'}<small>% avg</small></div>
        <div class="what">${has ? `peak ${fmt(w.utilPeak)}% · VRAM avg ${fmt(w.vram)}%` : 'no GPU data in this window yet'}</div>
        <div class="stack" role="img" aria-label="idle ${Math.round(idle * 100)}%, active ${Math.round(mid * 100)}%, busy ${Math.round(busy * 100)}%">
          <div class="idle" style="flex-grow:${idle}"></div><div class="mid" style="flex-grow:${mid}"></div><div class="busy" style="flex-grow:${busy}"></div></div>
        <div class="stack-l"><span>idle ${has ? Math.round(idle * 100) : '–'}%</span><span>busy ${has ? Math.round(busy * 100) : '–'}%</span></div>
        <dl class="kv">
          <dt>Avg power</dt><dd>${w.power ? `${fmt(w.power)} W` : '–'}</dd>
          <dt>Peak temp</dt><dd>${w.tempPeak ? `${fmt(w.tempPeak)} °C` : '–'}</dd>
          <dt>VRAM held, GPU idle</dt><dd>${w.heldIdleShare != null ? `${Math.round(w.heldIdleShare * 100)}%` : '–'}</dd>
          <dt>CPU / RAM avg</dt><dd>${fmt(w.cpu)}% / ${fmt(w.mem)}%</dd>
        </dl>
        <canvas class="spark" data-i="${i}" aria-hidden="true"></canvas>
      </div>`;
    }).join('');
    host.querySelectorAll('canvas.spark').forEach((cv) => {
      const w = state.windows[Number(cv.dataset.i)];
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

  let lastInsights = '';
  function renderInsights() {
    // The list is aria-live: only touch the DOM when the content changes, or screen readers re-announce every tick.
    const html = state.insights.length
      ? state.insights.map((x) => `<div class="ins ${x.level}"><div class="ic">${ICON[x.level] || ICON.info}</div>
          <div><b>${esc(x.title)}<span class="lv">${LEVEL_LABEL[x.level] || ''}</span></b><p>${esc(x.detail)}</p></div></div>`).join('')
      : `<div class="ins good"><div class="ic">${ICON.good}</div><div><b>All clear<span class="lv">good</span></b><p>No issues detected in the last hour.</p></div></div>`;
    if (html !== lastInsights) $('#insights').innerHTML = lastInsights = html;
  }

  function renderProcs() {
    const g = currentGpu();
    const list = state.processes.filter((p) => !g || p.gpuUuid === g.uuid)
      .sort((a, b) => (b.usedMiB ?? -1) - (a.usedMiB ?? -1));
    const noMem = list.length && list.every((p) => p.usedMiB == null);
    $('#proc-sub').textContent = list.length ? `${list.length} using this GPU${noMem && state.platform === 'win32' ? ' · per-process VRAM is not reported by Windows (WDDM)' : ''}` : '';
    if (state.vendor !== 'nvidia') {
      $('#procs').innerHTML = '<div class="empty">Process listing needs an NVIDIA GPU with nvidia-smi.</div>';
      return;
    }
    $('#procs').innerHTML = list.length
      ? `<table><thead><tr><th>Process</th><th class="n">PID</th><th class="n">VRAM</th></tr></thead><tbody>${
        list.map((p) => `<tr><td class="name" title="${esc(p.name)}">${esc(p.name)}</td><td class="n">${p.pid}</td><td class="n">${p.usedMiB == null ? 'n/a' : `${fmt(p.usedMiB)} MiB`}</td></tr>`).join('')}</tbody></table>`
      : '<div class="empty">No compute processes on this GPU.</div>';
  }

  function renderSpecs() {
    const sp = state.specs;
    if (!sp) return;
    $('#host').textContent = `${sp.hostname} · ${sp.os}`;
    const rm = $('#remote');
    if (sp.remote) { rm.hidden = false; rm.querySelector('span').textContent = sp.remote; }
    const cards = [
      [ICON.cpu, 'Processor', sp.cpu.model, `${sp.cpu.cores} cores · ${sp.cpu.threads} threads${sp.cpu.maxGHz && sp.cpu.maxGHz > (sp.cpu.baseGHz || 0) ? ` · up to ${sp.cpu.maxGHz} GHz` : sp.cpu.baseGHz ? ` · base ${sp.cpu.baseGHz} GHz` : ''}`],
      ...sp.gpus.map((g) => [ICON.gpu, 'Graphics', g.name,
        [g.vramMiB ? `${(g.vramMiB / 1024).toFixed(0)} GB VRAM` : null, g.driver ? `driver ${g.driver}` : null, g.vbios ? `VBIOS ${g.vbios}` : null].filter(Boolean).join(' · ')]),
      [ICON.mem, 'Memory', `${gb(sp.memory.totalBytes)} GB`, sp.memory.modules.join('\n') || ''],
      ...sp.disks.map((d) => [ICON.disk, 'Storage', d.name, `${(d.sizeBytes / 1e12 >= 1 ? (d.sizeBytes / 1e12).toFixed(1) + ' TB' : (d.sizeBytes / 1e9).toFixed(0) + ' GB')} · ${d.type || ''}`]),
      [ICON.os, 'Operating system', sp.os, `kernel ${sp.kernel} · ${sp.arch}`],
      sp.system || sp.board ? [ICON.board, 'System', sp.system || sp.board, sp.system && sp.board ? `board ${sp.board}` : ''] : null,
      sp.cudaVersion ? [ICON.bolt, 'CUDA', `CUDA ${sp.cudaVersion}`, 'max version supported by the driver'] : null,
      [ICON.clock, 'Uptime', dur(sp.uptimeSec), `${sp.hostname}`],
    ].filter(Boolean);
    $('#specs').innerHTML = cards.map(([ic, k, v, d]) =>
      `<div class="spec"><div class="ic">${ic}</div><div><span>${esc(k)}</span><b>${esc(v)}</b>${d ? `<p>${esc(d).replace(/\n/g, '<br>')}</p>` : ''}</div></div>`).join('');
  }

  function renderTabs() {
    const gs = state.sample?.gpus || [];
    const tabs = $('#tabs');
    tabs.hidden = gs.length < 2;
    if (gs.length < 2) return;
    tabs.innerHTML = gs.map((g) => `<button role="tab" data-gpu="${esc(g.uuid)}" aria-selected="${g.uuid === state.selected}">GPU ${g.index} · ${esc(g.name.replace(/^NVIDIA (GeForce )?/, ''))}</button>`).join('');
  }

  function renderRate() {
    const sec = state.config.intervalMs / 1000;
    $('#rate').textContent = `live · ${sec < 1 ? sec.toFixed(2) : sec.toFixed(1)}s`;
  }

  setInterval(() => {
    const stale = Date.now() - state.lastTick > Math.max(4000, state.config.intervalMs * 3);
    $('#live').classList.toggle('stale', stale);
  }, 1000);

  // ---- messages ----
  window.addEventListener('message', (ev) => {
    const m = ev.data;
    switch (m.type) {
      case 'init':
        Object.assign(state, {
          recent: m.recent || [], heatmap: m.heatmap, windows: m.windows || [], insights: m.insights || [],
          grade: m.grade, sample: m.sample, specs: m.specs, processes: m.processes || [], selected: m.selected,
          config: m.config, vendor: m.vendor, platform: m.platform, lastTick: Date.now(),
        });
        gpuChart.setData(state.recent); cpuChart.setData(state.recent); heat.set(state.heatmap);
        renderTabs(); renderLive(); renderWindows(); renderInsights(); renderProcs(); renderSpecs(); renderRate();
        break;
      case 'tick': {
        const prevSel = state.selected;
        Object.assign(state, { sample: m.sample, windows: m.windows, insights: m.insights, grade: m.grade, selected: m.selected, lastTick: Date.now() });
        if (m.point) {
          state.recent.push(m.point);
          const cutoff = m.point[0] - 60 * 60e3;
          while (state.recent.length && state.recent[0][0] < cutoff) state.recent.shift();
        }
        gpuChart.setData(state.recent); cpuChart.setData(state.recent);
        if (prevSel !== state.selected) renderTabs();
        renderLive(); renderWindows(); renderInsights();
        break;
      }
      case 'heatmap': state.heatmap = m.heatmap; heat.set(m.heatmap); break;
      case 'processes': state.processes = m.processes; renderProcs(); break;
      case 'specs': state.specs = m.specs; renderSpecs(); break;
    }
  });

  vscode.postMessage({ type: 'ready' });
})();
