// GPU Pulse side bar view: compact live readings for narrow widths (~220–500 px).
// Same messages as the dashboard (init / tick); only changed values touch the DOM.
(function () {
  'use strict';
  const P = window.Pulse;
  const vscode = typeof acquireVsCodeApi === 'function' ? acquireVsCodeApi() : { postMessage() {} };
  const SPARK_MS = 5 * 60e3;

  const state = { sample: null, recent: [], insights: [], fleet: [], alerts: [], selected: null, config: { tempWarn: 80, intervalMs: 500 }, vendor: 'none', lastTick: 0 };

  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const fmt = (v, d = 0) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
  const gib = (m) => (m == null ? '–' : (m / 1024).toFixed(1));
  const shortName = (n) => String(n || '').replace(/^NVIDIA (GeForce )?/, '');
  const txt = (el, v) => { if (el._t !== v) { el._t = v; el.textContent = v; } };
  const html = (el, v) => { if (el._h !== v) { el._h = v; el.innerHTML = v; } };
  const width = (el, pct) => { const v = `${Math.round(Math.max(0, Math.min(100, pct || 0)))}%`; if (el._w !== v) { el._w = v; el.style.width = v; } };

  document.getElementById('app').innerHTML = `
    <div class="sb">
      <div class="sb-top"><span class="live-dot" id="live"></span><span id="rate">live</span><span class="sb-sp"></span><span class="sb-alerts" id="alerts" role="status" hidden></span><span id="sum"></span></div>
      <div class="sb-gpus" id="gpus"><div class="sb-empty">Waiting for data…</div></div>
      <section class="sb-sec">
        <div class="sb-h"><span id="spark-title">GPU utilization</span><span class="sb-sub">last 5 min</span></div>
        <div class="sb-spark"><canvas id="spark" role="img" aria-label="GPU utilization over the last 5 minutes"></canvas></div>
      </section>
      <section class="sb-sec">
        <div class="sb-row"><span>CPU</span><b id="cpu-v">–</b></div><div class="sb-bar"><span class="cpu" id="cpu-b"></span></div>
        <div class="sb-row"><span>RAM</span><b id="ram-v">–</b></div><div class="sb-bar"><span class="ram" id="ram-b"></span></div>
      </section>
      <section class="sb-sec">
        <div class="sb-h"><span>Insights</span><span class="sb-sub">last hour</span></div>
        <div id="ins" aria-live="polite"></div>
      </section>
      <button class="sb-open" data-cmd="gpuPulse.openDashboard">Open full dashboard</button>
    </div>`;
  const E = {};
  for (const id of ['live', 'rate', 'alerts', 'sum', 'gpus', 'spark-title', 'spark', 'cpu-v', 'cpu-b', 'ram-v', 'ram-b', 'ins']) E[id] = document.getElementById(id);

  // ---- GPU rows ----
  const rows = new Map();
  let rowsKey = '';
  let pendingSel = null; // ticks in flight carry the old selection until the host's init confirms
  const gpus = () => state.sample?.gpus || [];
  const current = () => gpus().find((g) => g.uuid === state.selected) || gpus()[0];

  function buildRows(gs) {
    const multi = gs.length > 1;
    E.gpus.setAttribute('role', multi ? 'radiogroup' : 'group');
    E.gpus.setAttribute('aria-label', multi ? 'GPU shown in the chart' : 'GPU');
    const tag = multi ? 'button' : 'div';
    E.gpus.innerHTML = gs.map((g) => `<${tag} class="sb-gpu" data-gpu="${esc(g.uuid)}"${multi ? ' role="radio" aria-checked="false" tabindex="-1"' : ''}>
        <span class="sb-ring"><canvas aria-hidden="true"></canvas><span class="sb-ring-v"><b data-f="util">–</b><small>%</small></span></span>
        <span class="sb-info">
          <span class="sb-name"><span class="sb-idx">GPU ${g.index}</span><span class="sb-model" title="${esc(g.name)}">${esc(shortName(g.name))}</span></span>
          <span class="sb-vram"><span class="sb-bar"><span class="vram" data-f="vbar"></span></span><span class="sb-num" data-f="vram">–</span></span>
          <span class="sb-meta"><span data-f="temp">–</span><span data-f="power">–</span><span data-f="avg"></span></span>
        </span></${tag}>`).join('');
    rows.clear();
    for (const el of E.gpus.children) {
      const f = {};
      el.querySelectorAll('[data-f]').forEach((x) => (f[x.dataset.f] = x));
      rows.set(el.dataset.gpu, { el, f, gauge: new P.Gauge(el.querySelector('canvas'), { mini: true, ticks: 0 }) });
    }
  }

  function renderRows() {
    const gs = gpus();
    if (!gs.length) {
      html(E.gpus, `<div class="sb-empty">${state.vendor === 'none' ? 'No supported GPU found. CPU and RAM are still tracked.' : 'Reading…'}</div>`);
      rowsKey = '';
      return;
    }
    const key = gs.map((g) => `${g.uuid}|${g.index}|${g.name}`).join(',');
    if (key !== rowsKey) { rowsKey = key; E.gpus._h = undefined; buildRows(gs); renderSelection(); }
    const avg = new Map((state.fleet || []).map((f) => [f.uuid, f.util]));
    const warn = state.config.tempWarn;
    let sum = 0, n = 0;
    for (const g of gs) {
      const r = rows.get(g.uuid);
      if (!r) continue;
      if (g.util != null) { sum += g.util; n++; }
      r.gauge.set(g.util);
      txt(r.f.util, fmt(g.util));
      width(r.f.vbar, g.memTotalMiB ? (g.memUsedMiB / g.memTotalMiB) * 100 : 0);
      txt(r.f.vram, g.vendor === 'apple' ? `${gib(g.memUsedMiB)} GB shared` : `${gib(g.memUsedMiB)}/${gib(g.memTotalMiB)} GB`);
      txt(r.f.temp, g.tempC == null ? '– °C' : `${fmt(g.tempC)} °C`);
      txt(r.f.power, g.powerW == null ? '– W' : `${fmt(g.powerW)} W`);
      const a = avg.get(g.uuid);
      txt(r.f.avg, a == null ? '' : `10m ${fmt(a)}%`);
      const hot = g.tempC != null && g.tempC >= warn;
      if (r.hot !== hot) { r.hot = hot; r.el.classList.toggle('hot', hot); }
    }
    txt(E.sum, gs.length > 1 ? `${gs.length} GPUs · avg ${n ? fmt(sum / n) : '–'}%` : '');
  }

  function renderSelection() {
    const cur = current()?.uuid;
    if (rows.size < 2) return;
    for (const [uuid, r] of rows) {
      const on = uuid === cur;
      r.el.setAttribute('aria-checked', String(on));
      r.el.tabIndex = on ? 0 : -1;
    }
  }

  function select(uuid) {
    if (!uuid || uuid === current()?.uuid) return;
    state.selected = pendingSel = uuid;
    renderSelection(); renderSpark();
    vscode.postMessage({ type: 'selectGpu', uuid });
  }

  // ---- sparkline, system bars, insights ----
  function renderSpark() {
    const g = current();
    txt(E['spark-title'], g && gpus().length > 1 ? `GPU ${g.index} utilization` : 'GPU utilization');
    const pts = state.recent;
    const cut = (pts.length ? pts[pts.length - 1][0] : Date.now()) - SPARK_MS;
    let vals = [];
    for (let i = pts.length - 1; i >= 0 && pts[i][0] >= cut; i--) vals.push(pts[i][1]);
    vals.reverse();
    const max = Math.max(60, Math.round(E.spark.clientWidth || 200));
    if (vals.length > max) { // keep the peak of each bucket
      const k = vals.length / max, out = [];
      for (let b = 0; b < max; b++) {
        let m = null;
        for (let j = Math.floor(b * k); j < Math.floor((b + 1) * k); j++) if (vals[j] != null && (m == null || vals[j] > m)) m = vals[j];
        out.push(m);
      }
      vals = out;
    }
    P.spark(E.spark, vals, 's-gpu');
  }

  function renderSystem() {
    const s = state.sample;
    if (!s) return;
    txt(E['cpu-v'], `${fmt(s.cpu.util)}%`);
    width(E['cpu-b'], s.cpu.util);
    const mp = s.mem.totalBytes ? (s.mem.usedBytes / s.mem.totalBytes) * 100 : 0;
    txt(E['ram-v'], `${(s.mem.usedBytes / 1073741824).toFixed(1)}/${(s.mem.totalBytes / 1073741824).toFixed(0)} GB · ${fmt(mp)}%`);
    width(E['ram-b'], mp);
  }

  const LEVEL = { critical: 'critical', serious: 'serious', warning: 'warning', info: 'note', good: 'good' };
  function renderInsights() {
    const top = state.insights.slice(0, 2);
    html(E.ins, top.length
      ? top.map((x) => `<div class="sb-ins ${x.level}" title="${esc(x.detail)}"><i aria-hidden="true"></i><div><b>${esc(x.title)}</b><span class="sr"> (${LEVEL[x.level] || ''})</span><p>${esc(x.detail)}</p></div></div>`).join('') +
        (state.insights.length > 2 ? `<div class="sb-more">+${state.insights.length - 2} more in the dashboard</div>` : '')
      : '<div class="sb-ins good"><i aria-hidden="true"></i><div><b>All clear</b><p>No issues detected in the last hour.</p></div></div>');
  }

  function renderRate() {
    const sec = state.config.intervalMs / 1000;
    txt(E.rate, `live · ${sec < 1 ? sec.toFixed(2) : sec.toFixed(1)}s`);
  }

  // Active alerts: a small badge in the top row; the titles are in its tooltip.
  function renderAlerts() {
    const a = state.alerts || [];
    if (E.alerts.hidden !== !a.length) E.alerts.hidden = !a.length;
    E.alerts.classList.toggle('warn', a.some((x) => x.level === 'warning'));
    // Alert text only goes through DOM properties (title / aria-label), never innerHTML.
    E.alerts.title = a.map((x) => `${x.title}. ${x.detail}`).join('\n');
    E.alerts.setAttribute('aria-label', a.length ? `${a.length} active GPU alert${a.length > 1 ? 's' : ''}: ${a.map((x) => x.title).join('; ')}` : '');
    html(E.alerts, a.length ? `<span aria-hidden="true">!</span> ${a.length} alert${a.length > 1 ? 's' : ''}` : '');
  }

  function renderAll() { renderRows(); renderSelection(); renderSpark(); renderSystem(); renderInsights(); renderRate(); renderAlerts(); }

  // ---- events ----
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-gpu], [data-cmd]');
    if (!b) return;
    if (b.dataset.cmd) vscode.postMessage({ type: 'command', command: b.dataset.cmd });
    else if (rows.size > 1) select(b.dataset.gpu);
  });
  E.gpus.addEventListener('keydown', (e) => {
    const btns = [...E.gpus.querySelectorAll('button[data-gpu]')];
    const i = btns.indexOf(document.activeElement);
    if (i < 0) return;
    const next = { ArrowDown: i + 1, ArrowRight: i + 1, ArrowUp: i - 1, ArrowLeft: i - 1, Home: 0, End: btns.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    const b = btns[(next + btns.length) % btns.length];
    b.focus();
    select(b.dataset.gpu);
  });
  new MutationObserver(() => { P.readTokens(); for (const r of rows.values()) r.gauge.draw(); renderSpark(); })
    .observe(document.body, { attributes: true, attributeFilter: ['class'] });
  new ResizeObserver(() => renderSpark()).observe(E.spark);

  // ---- liveness + hidden handling ----
  let staleTimer = 0, dirty = false;
  const checkStale = () => {
    const stale = Date.now() - state.lastTick > Math.max(4000, state.config.intervalMs * 3);
    if (E.live._stale !== stale) { E.live._stale = stale; E.live.classList.toggle('stale', stale); }
  };
  const startStale = () => { if (!staleTimer) staleTimer = setInterval(checkStale, 1000); };
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { clearInterval(staleTimer); staleTimer = 0; return; }
    startStale();
    if (dirty) { dirty = false; renderAll(); }
  });
  if (!document.hidden) startStale();

  const applyAnalysis = (a) => { if (a) { state.insights = a.insights || []; state.fleet = a.fleet || []; } };

  window.addEventListener('message', (ev) => {
    const m = ev.data;
    if (m.type === 'init') {
      pendingSel = null;
      Object.assign(state, { sample: m.sample, recent: m.recent || [], selected: m.selected, config: m.config || state.config, vendor: m.vendor, lastTick: Date.now(), alerts: m.alerts || [] });
      applyAnalysis(m.analysis);
      P.setAnimMs(state.config.intervalMs * 0.6);
      if (document.hidden) dirty = true; else renderAll();
    } else if (m.type === 'tick') {
      Object.assign(state, { sample: m.sample, selected: pendingSel || m.selected, lastTick: Date.now() });
      if (m.alerts) state.alerts = m.alerts;
      if (!pendingSel) {
        if (m.analysis) applyAnalysis(m.analysis);
        if (m.point) {
          state.recent.push(m.point);
          let drop = 0;
          while (drop < state.recent.length && state.recent[drop][0] < m.point[0] - SPARK_MS) drop++;
          if (drop) state.recent.splice(0, drop);
        }
      }
      if (document.hidden) { dirty = true; return; }
      renderRows(); renderSpark(); renderSystem();
      if (m.analysis) renderInsights();
      if (m.alerts) renderAlerts();
      if (E.live._stale) checkStale();
    }
  });

  vscode.postMessage({ type: 'ready' });
})();
