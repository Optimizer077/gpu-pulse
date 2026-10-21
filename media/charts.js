// Canvas drawing primitives for the GPU Pulse dashboard. Exposes window.Pulse.
(function () {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isLight = () => document.body.classList.contains('vscode-light');
  const tokens = {};
  function readTokens() {
    const cs = getComputedStyle(document.body);
    for (const k of ['s-gpu', 's-vram', 's-cpu', 's-ram', 'glow-gpu', 'glow-vram', 'grid', 'hair', 'hair-strong', 'ink', 'ink-2', 'muted',
      'surface', 'heat-0', 'heat-1', 'heat-2', 'heat-3', 'critical', 'warning', 'good']) {
      tokens[k] = cs.getPropertyValue('--' + k).trim();
    }
  }

  function rgb(hex) {
    const h = hex.replace('#', '');
    const n = parseInt(h.length === 3 ? h.replace(/./g, '$&$&') : h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function rgba(hex, a) {
    if (!hex || hex[0] !== '#') return hex;
    const [r, g, b] = rgb(hex);
    return `rgba(${r},${g},${b},${a})`;
  }
  function mix(a, b, t) {
    const x = rgb(a), y = rgb(b);
    return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(',')})`;
  }
  /** Sequential ramp: one hue from near-surface to bright (dark) / deep (light). */
  function heatColor(v) {
    const stops = [tokens['heat-0'], tokens['heat-1'], tokens['heat-2'], tokens['heat-3']];
    const x = Math.max(0, Math.min(1, v / 100)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    return mix(stops[i], stops[i + 1], x - i);
  }

  /** Size a canvas to its CSS box at device pixel ratio; returns a 2D context in CSS pixels. */
  function fit(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  // ---- shared animation loop (runs only while something is moving) ----
  const animating = new Set();
  let raf = 0;
  function loop() {
    raf = 0;
    for (const c of [...animating]) if (!c.step()) animating.delete(c);
    if (animating.size) raf = requestAnimationFrame(loop);
  }
  function animate(c) {
    animating.add(c);
    if (!raf) raf = requestAnimationFrame(loop);
  }

  const START = Math.PI * 0.75, SWEEP = Math.PI * 1.5;

  /** 270° radial gauge with gradient arc, glow, tick ring and an optional marker (e.g. 1h average). */
  class Gauge {
    constructor(canvas, opt) {
      this.canvas = canvas;
      this.opt = Object.assign({ max: 100, color: 's-gpu', glow: 'glow-gpu', ticks: 50, warn: null, big: false }, opt);
      this.value = 0;
      this.target = 0;
      this.marker = null;
      new ResizeObserver(() => this.draw()).observe(canvas);
    }
    set(v, marker) {
      this.target = v == null ? 0 : Math.max(0, Math.min(this.opt.max, v));
      this.marker = marker == null ? null : Math.max(0, Math.min(this.opt.max, marker));
      if (reduceMotion) { this.value = this.target; this.draw(); } else animate(this);
    }
    setMax(m) { if (m && m !== this.opt.max) { this.opt.max = m; this.draw(); } }
    step() {
      const d = this.target - this.value;
      this.value = Math.abs(d) < this.opt.max * 0.002 ? this.target : this.value + d * 0.14;
      this.draw();
      return this.value !== this.target;
    }
    draw() {
      const { ctx, w, h } = fit(this.canvas);
      if (!w || !h) return;
      const o = this.opt;
      const cx = w / 2, cy = h * 0.54;
      const r = Math.min(w / 2, h * 0.54) * 0.84;
      const lw = Math.max(5, r * (o.big ? 0.085 : 0.1));
      const col = tokens[o.color], glow = tokens[o.glow] || col;
      const frac = this.value / o.max;

      // tick ring
      for (let i = 0; i <= o.ticks; i++) {
        const a = START + (SWEEP * i) / o.ticks;
        const major = i % 5 === 0;
        const lit = i / o.ticks <= frac + 1e-6 && frac > 0;
        const r1 = r + lw * 0.95, r2 = r1 + (major ? lw * 0.75 : lw * 0.4);
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
        ctx.lineTo(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2);
        ctx.strokeStyle = lit ? rgba(glow, major ? 0.95 : 0.6) : (major ? tokens['hair-strong'] : tokens.hair);
        ctx.lineWidth = major ? 1.6 : 1;
        ctx.stroke();
      }

      // track
      ctx.beginPath();
      ctx.arc(cx, cy, r, START, START + SWEEP);
      ctx.strokeStyle = tokens.grid;
      ctx.lineWidth = lw;
      ctx.lineCap = 'round';
      ctx.stroke();

      // warning zone
      if (o.warn != null && o.warn < o.max) {
        ctx.beginPath();
        ctx.arc(cx, cy, r - lw * 0.9, START + SWEEP * (o.warn / o.max), START + SWEEP);
        ctx.strokeStyle = rgba(tokens.critical, 0.55);
        ctx.lineWidth = 2;
        ctx.lineCap = 'butt';
        ctx.stroke();
      }

      // value arc
      if (frac > 0.002) {
        const end = START + SWEEP * frac;
        let grad;
        if (ctx.createConicGradient) {
          grad = ctx.createConicGradient(START - 0.1, cx, cy);
          grad.addColorStop(0, rgba(col, 0.55));
          grad.addColorStop(Math.min(0.99, (SWEEP * frac) / (Math.PI * 2)), glow);
          grad.addColorStop(1, rgba(col, 0.55));
        } else grad = col;
        ctx.save();
        if (!isLight()) { ctx.shadowColor = rgba(glow, 0.75); ctx.shadowBlur = o.big ? 22 : 14; }
        ctx.beginPath();
        ctx.arc(cx, cy, r, START, end);
        ctx.strokeStyle = grad;
        ctx.lineWidth = lw;
        ctx.lineCap = 'round';
        ctx.stroke();
        ctx.restore();
        // bright tip
        ctx.beginPath();
        ctx.arc(cx + Math.cos(end) * r, cy + Math.sin(end) * r, lw * 0.32, 0, Math.PI * 2);
        ctx.fillStyle = isLight() ? '#fff' : '#ffffff';
        ctx.fill();
      }

      // inner hairline ring
      ctx.beginPath();
      ctx.arc(cx, cy, r - lw * 1.6, START, START + SWEEP);
      ctx.strokeStyle = tokens.hair;
      ctx.lineWidth = 1;
      ctx.stroke();

      // marker (window average)
      if (this.marker != null) {
        const a = START + SWEEP * (this.marker / o.max);
        const r0 = r - lw * 1.15;
        ctx.save();
        ctx.translate(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
        ctx.rotate(a + Math.PI / 2);
        ctx.beginPath();
        ctx.moveTo(0, -4); ctx.lineTo(4.5, 4); ctx.lineTo(-4.5, 4); ctx.closePath();
        ctx.fillStyle = tokens.ink;
        ctx.fill();
        ctx.restore();
      }
    }
  }

  /** Time-series area chart on a single 0–max axis with crosshair tooltip. */
  class LineChart {
    constructor(canvas, opt) {
      this.canvas = canvas;
      this.opt = Object.assign({ max: 100, unit: '%', series: [], rangeMs: 5 * 60e3, gapMs: 15e3 }, opt);
      this.points = [];
      this.hover = null;
      this.tip = opt.tip;
      new ResizeObserver(() => this.draw()).observe(canvas);
      canvas.addEventListener('mousemove', (e) => { this.hover = e; this.draw(); });
      canvas.addEventListener('mouseleave', () => { this.hover = null; this.tip.hide(); this.draw(); });
    }
    setData(points) { this.points = points; this.draw(); }
    setRange(ms) { this.opt.rangeMs = ms; this.draw(); }
    draw() {
      const { ctx, w, h } = fit(this.canvas);
      if (!w || !h) return;
      const o = this.opt;
      const L = 34, R = 10, T = 8, B = 20;
      const pw = w - L - R, ph = h - T - B;
      const now = this.points.length ? this.points[this.points.length - 1][0] : Date.now();
      const t0 = now - o.rangeMs;
      const X = (t) => L + ((t - t0) / o.rangeMs) * pw;
      const Y = (v) => T + ph - (Math.max(0, Math.min(o.max, v)) / o.max) * ph;

      // grid + y labels
      ctx.font = '10px "Azeret Mono", monospace';
      ctx.textBaseline = 'middle';
      for (let i = 0; i <= 4; i++) {
        const v = (o.max / 4) * i, y = Math.round(Y(v)) + 0.5;
        ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - R, y);
        ctx.strokeStyle = i === 0 ? tokens['hair-strong'] : tokens.grid; ctx.lineWidth = 1; ctx.stroke();
        ctx.fillStyle = tokens.muted; ctx.textAlign = 'right';
        ctx.fillText(`${Math.round(v)}${o.unit}`, L - 6, y);
      }
      // time labels
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const steps = 5;
      for (let i = 0; i <= steps; i++) {
        const t = t0 + (o.rangeMs * i) / steps;
        const ago = Math.round((now - t) / 1000);
        const lbl = ago === 0 ? 'now' : ago >= 60 ? `-${Math.round(ago / 60)}m` : `-${ago}s`;
        ctx.fillStyle = tokens.muted;
        ctx.fillText(lbl, Math.min(w - R - 10, Math.max(L + 10, X(t))), T + ph + 6);
      }

      const pts = this.points.filter((p) => p[0] >= t0 - 2000);
      ctx.save();
      ctx.beginPath(); ctx.rect(L, T - 4, pw, ph + 8); ctx.clip();
      for (const s of o.series) {
        const col = tokens[s.color], glow = tokens[s.glow] || col;
        const runs = [];
        let run = [];
        for (let i = 0; i < pts.length; i++) {
          const v = pts[i][s.idx];
          if (v == null || (run.length && pts[i][0] - run[run.length - 1][0] > o.gapMs)) {
            if (run.length) runs.push(run);
            run = [];
          }
          if (v != null) run.push([pts[i][0], v]);
        }
        if (run.length) runs.push(run);
        for (const r of runs) {
          if (s.fill) {
            const g = ctx.createLinearGradient(0, T, 0, T + ph);
            g.addColorStop(0, rgba(col, isLight() ? 0.22 : 0.34));
            g.addColorStop(1, rgba(col, 0));
            ctx.beginPath();
            ctx.moveTo(X(r[0][0]), Y(0));
            for (const [t, v] of r) ctx.lineTo(X(t), Y(v));
            ctx.lineTo(X(r[r.length - 1][0]), Y(0));
            ctx.closePath();
            ctx.fillStyle = g; ctx.fill();
          }
          ctx.save();
          if (!isLight()) { ctx.shadowColor = rgba(glow, 0.6); ctx.shadowBlur = 8; }
          ctx.beginPath();
          r.forEach(([t, v], i) => (i ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))));
          ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
          ctx.stroke();
          ctx.restore();
        }
        // endpoint dot with surface ring
        const last = pts.length ? pts[pts.length - 1] : null;
        if (last && last[s.idx] != null) {
          ctx.beginPath(); ctx.arc(X(last[0]), Y(last[s.idx]), 5, 0, Math.PI * 2);
          ctx.fillStyle = tokens.surface; ctx.fill();
          ctx.beginPath(); ctx.arc(X(last[0]), Y(last[s.idx]), 3.4, 0, Math.PI * 2);
          ctx.fillStyle = glow; ctx.fill();
        }
      }
      ctx.restore();

      // crosshair + tooltip
      if (this.hover && pts.length) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = this.hover.clientX - rect.left;
        if (mx < L || mx > w - R) { this.tip.hide(); return; }
        const tm = t0 + ((mx - L) / pw) * o.rangeMs;
        let best = pts[0];
        for (const p of pts) if (Math.abs(p[0] - tm) < Math.abs(best[0] - tm)) best = p;
        const x = Math.round(X(best[0])) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ph);
        ctx.strokeStyle = tokens['hair-strong']; ctx.lineWidth = 1; ctx.stroke();
        for (const s of o.series) {
          const v = best[s.idx];
          if (v == null) continue;
          ctx.beginPath(); ctx.arc(x, Y(v), 5.5, 0, Math.PI * 2); ctx.fillStyle = tokens.surface; ctx.fill();
          ctx.beginPath(); ctx.arc(x, Y(v), 4, 0, Math.PI * 2); ctx.fillStyle = tokens[s.color]; ctx.fill();
        }
        this.tip.show(this.hover.clientX, this.hover.clientY, new Date(best[0]).toLocaleTimeString(),
          o.series.map((s) => [s.label, tokens[s.color], best[s.idx] == null ? '–' : `${best[s.idx].toFixed(1)}${o.unit}`]));
      }
    }
  }

  /** Clock-aligned 24 × 60 grid: one row per hour, one cell per minute. */
  class Heatmap {
    constructor(canvas, tip) {
      this.canvas = canvas;
      this.tip = tip;
      this.data = null;
      this.layout = null;
      new ResizeObserver(() => this.draw()).observe(canvas);
      canvas.addEventListener('mousemove', (e) => this.onHover(e));
      canvas.addEventListener('mouseleave', () => tip.hide());
    }
    set(data) { this.data = data; this.draw(); }
    cellAt(t) {
      const d = this.data;
      const i = Math.round((t - d.start) / 60000);
      return i >= 0 && i < d.cells.length ? d.cells[i] : null;
    }
    draw() {
      const { ctx, w, h } = fit(this.canvas);
      if (!w || !h || !this.data) return;
      const L = 46, T = 16, R = 4, B = 2;
      const now = Date.now();
      // Rows align on local clock hours (handles half-hour time zones).
      const hourStart = new Date(now);
      hourStart.setMinutes(0, 0, 0);
      const top = hourStart.getTime() - 23 * 3600e3;
      const cw = (w - L - R) / 60, ch = (h - T - B) / 24;
      const gap = Math.min(2, cw * 0.18);
      this.layout = { L, T, cw, ch, top };
      ctx.font = '10px "Azeret Mono", monospace';
      ctx.fillStyle = tokens.muted;
      ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'center';
      for (const m of [0, 15, 30, 45]) ctx.fillText(`:${String(m).padStart(2, '0')}`, L + m * cw + cw / 2, T - 5);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let r = 0; r < 24; r++) {
        const rowT = top + r * 3600e3;
        const hr = new Date(rowT).getHours();
        if (r === 23 || (hr % 3 === 0 && r < 22)) {
          ctx.fillStyle = r === 23 ? tokens.ink : tokens.muted;
          ctx.fillText(r === 23 ? 'now' : `${String(hr).padStart(2, '0')}:00`, L - 8, T + r * ch + ch / 2);
        }
        for (let c = 0; c < 60; c++) {
          const t = rowT + c * 60e3;
          const x = L + c * cw + gap / 2, y = T + r * ch + gap / 2;
          const cwi = cw - gap, chi = ch - gap;
          if (t > now) continue;
          const v = this.cellAt(t);
          const rad = Math.min(2.5, cwi / 3);
          ctx.beginPath();
          if (ctx.roundRect) ctx.roundRect(x, y, cwi, chi, rad); else ctx.rect(x, y, cwi, chi);
          if (v == null) {
            ctx.fillStyle = rgba(tokens.muted, isLight() ? 0.08 : 0.06);
            ctx.fill();
          } else {
            ctx.fillStyle = heatColor(v);
            if (!isLight() && v > 70) { ctx.save(); ctx.shadowColor = rgba(tokens['glow-gpu'], 0.55); ctx.shadowBlur = 6; ctx.fill(); ctx.restore(); }
            else ctx.fill();
          }
        }
      }
    }
    onHover(e) {
      if (!this.layout || !this.data) return;
      const { L, T, cw, ch, top } = this.layout;
      const rect = this.canvas.getBoundingClientRect();
      const c = Math.floor((e.clientX - rect.left - L) / cw), r = Math.floor((e.clientY - rect.top - T) / ch);
      if (c < 0 || c >= 60 || r < 0 || r >= 24) return this.tip.hide();
      const t = top + r * 3600e3 + c * 60e3;
      if (t > Date.now()) return this.tip.hide();
      const v = this.cellAt(t);
      const d = new Date(t);
      this.tip.show(e.clientX, e.clientY, `${d.toLocaleDateString(undefined, { weekday: 'short' })} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`,
        [['GPU util (mean)', v == null ? tokens.muted : heatColor(v), v == null ? 'not recorded' : `${v.toFixed(1)}%`]]);
    }
  }

  /** Tiny area sparkline drawn behind window cards. */
  function spark(canvas, values, colorKey) {
    const { ctx, w, h } = fit(canvas);
    const vals = values.filter((v) => v != null);
    if (!w || vals.length < 2) return;
    const col = tokens[colorKey];
    const X = (i) => (i / (values.length - 1)) * w;
    const Y = (v) => h - 2 - (v / 100) * (h - 6);
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, rgba(col, 0.32)); g.addColorStop(1, rgba(col, 0));
    ctx.beginPath(); ctx.moveTo(0, h);
    values.forEach((v, i) => ctx.lineTo(X(i), Y(v ?? 0)));
    ctx.lineTo(w, h); ctx.closePath(); ctx.fillStyle = g; ctx.fill();
    ctx.beginPath();
    values.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v ?? 0)) : ctx.moveTo(X(i), Y(v ?? 0))));
    ctx.strokeStyle = rgba(col, 0.9); ctx.lineWidth = 1.5; ctx.stroke();
  }

  /** Shared HTML tooltip. */
  function makeTip() {
    const el = document.createElement('div');
    el.className = 'tip';
    el.setAttribute('role', 'tooltip');
    document.body.appendChild(el);
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
    return {
      show(x, y, title, rows) {
        el.innerHTML = `<div class="t">${esc(title)}</div>` +
          rows.map(([k, c, v]) => `<div class="r"><span><i style="background:${c}"></i>${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
        el.classList.add('on');
        const bw = el.offsetWidth, bh = el.offsetHeight;
        el.style.left = `${Math.min(window.innerWidth - bw - 8, x + 14)}px`;
        el.style.top = `${Math.max(8, y - bh - 12)}px`;
      },
      hide() { el.classList.remove('on'); },
    };
  }

  readTokens();
  window.Pulse = { Gauge, LineChart, Heatmap, spark, makeTip, readTokens, tokens, heatColor };
})();
