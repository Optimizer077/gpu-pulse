import type { GpuSample, Sample } from './types';

/**
 * Alert rule engine (no vscode dependency, so it is unit-testable with fake timestamps).
 * Per sample it does O(GPUs) work: every rule keeps a few numbers per GPU, never a history scan.
 * The terminal tool (pip/) implements the same rules.
 */

export type AlertRule = 'hot' | 'vramFull' | 'idleHeld' | 'jobFinished';
export type AlertLevel = 'info' | 'warning';

export interface Alert {
  rule: AlertRule;
  gpuUuid: string;
  gpuIndex: number;
  level: AlertLevel;
  title: string;
  detail: string;
}

export interface AlertConfig {
  enabled: boolean;
  hot: boolean;
  vramFull: boolean;
  idleHeld: boolean;
  jobFinished: boolean;
  /** gpuPulse.temperatureWarning (°C). */
  tempWarn: number;
  idleHeldMinutes: number;
  jobFinishedMinutes: number;
  /** Current sampling interval; a gap over GAP_FACTOR × this resets the "continuously" timers. */
  intervalMs: number;
}

export const HOT_HOLD_MS = 30_000;
export const HOT_CLEAR_MARGIN_C = 3;
export const VRAM_FULL_PCT = 95;
export const VRAM_CLEAR_PCT = 90;
export const VRAM_HOLD_MS = 20_000;
export const IDLE_UTIL = 10;
export const IDLE_VRAM_PCT = 50;
export const IDLE_CLEAR_UTIL = 15;
export const IDLE_CLEAR_VRAM_PCT = 40;
export const BUSY_UTIL = 15;
export const JOB_END_MS = 60_000;
export const COOLDOWN_MS = 30 * 60_000;
export const GAP_FACTOR = 3;
/** After an interval change, the old (longer) interval still bounds gaps this long: the stream restarts with a delay. */
export const INTERVAL_GRACE_MS = 10_000;

/** A condition held for a duration, latched once it fired until its clear condition (hysteresis) holds. */
interface Latch {
  since: number | null; // start of the current continuous run of the condition
  fired: boolean; // fired (or was suppressed by the cooldown) and has not cleared yet
  alert: Alert | null; // what fired, for active()
}

interface GpuState {
  lastT: number;
  hot: Latch;
  vramFull: Latch;
  idleHeld: Latch;
  // jobFinished: one open busy stretch at a time
  jobStart: number | null; // first busy sample of the stretch
  jobLastBusy: number; // last sample with util >= BUSY_UTIL
  jobLowSince: number | null; // start of the current run of util < IDLE_UTIL
  jobUtilMs: number; // ∑ util × dt over the stretch up to jobLastBusy (time-weighted average)
  jobMs: number; // ∑ dt matching jobUtilMs
  jobPendUtilMs: number; // same, for samples after jobLastBusy (merged when the stretch continues)
  jobPendMs: number;
  lastFired: Partial<Record<AlertRule, number>>;
}

const latch = (): Latch => ({ since: null, fired: false, alert: null });

function newState(t: number): GpuState {
  return {
    lastT: t, hot: latch(), vramFull: latch(), idleHeld: latch(),
    jobStart: null, jobLastBusy: 0, jobLowSince: null, jobUtilMs: 0, jobMs: 0, jobPendUtilMs: 0, jobPendMs: 0,
    lastFired: {},
  };
}

function dropJob(st: GpuState): void {
  st.jobStart = st.jobLowSince = null;
  st.jobUtilMs = st.jobMs = st.jobPendUtilMs = st.jobPendMs = 0;
}

/** A sample gap: restart every "continuously" timer. An open busy stretch survives gaps shorter than the bridge. */
function resetTimers(st: GpuState, gapMs: number): void {
  st.hot.since = st.vramFull.since = st.idleHeld.since = null;
  st.jobLowSince = null;
  if (gapMs < 0 || gapMs >= JOB_END_MS) dropJob(st);
}

/**
 * Share of GPU memory in use. On unified memory (Apple Silicon) memTotalMiB is all of system
 * RAM and memUsedMiB what the GPU driver holds, so vramFull / idleHeld compare GPU-allocated
 * memory with the unified total: "held" there means memory the GPU has allocated.
 */
const vramPct = (g: GpuSample): number | null =>
  g.memUsedMiB !== null && g.memTotalMiB !== null && g.memTotalMiB > 0 ? (g.memUsedMiB / g.memTotalMiB) * 100 : null;
const gb = (mib: number | null, d: number) => (mib === null ? '–' : (mib / 1024).toFixed(d));
const memName = (g: GpuSample) => (g.unifiedMemory ? 'memory' : 'VRAM');

export class AlertEngine {
  private cfg: AlertConfig;
  private readonly gpus = new Map<string, GpuState>();
  private prevIntervalMs = 0;
  private graceUntil = 0;
  private rev = 0;

  /** Increments whenever active() may have changed, so callers can cache what they render from it. */
  get revision(): number {
    return this.rev;
  }

  constructor(cfg: AlertConfig) {
    this.cfg = { ...cfg };
  }

  /** Apply new settings or a new sampling interval live. Disabled rules drop their state. */
  setConfig(cfg: AlertConfig, nowMs = Date.now()): void {
    const old = this.cfg;
    this.cfg = { ...cfg };
    this.rev++;
    if (cfg.intervalMs !== old.intervalMs) {
      this.prevIntervalMs = Math.max(old.intervalMs, nowMs < this.graceUntil ? this.prevIntervalMs : 0);
      this.graceUntil = nowMs + INTERVAL_GRACE_MS;
    }
    if (!cfg.enabled) {
      this.gpus.clear();
      return;
    }
    for (const st of this.gpus.values()) {
      if (!cfg.hot) st.hot = latch();
      if (!cfg.vramFull) st.vramFull = latch();
      if (!cfg.idleHeld) st.idleHeld = latch();
      if (!cfg.jobFinished) dropJob(st);
    }
  }

  /** Evaluate one sample; returns the alerts that fired now (cooldown already applied). */
  update(sample: Sample, nowMs?: number): Alert[] {
    if (!this.cfg.enabled) return [];
    const t = nowMs ?? sample.t;
    const out: Alert[] = [];
    const maxGap = GAP_FACTOR * (t < this.graceUntil ? Math.max(this.cfg.intervalMs, this.prevIntervalMs) : this.cfg.intervalMs);
    for (const g of sample.gpus) {
      let st = this.gpus.get(g.uuid);
      let dt = 0;
      if (!st) {
        st = newState(t);
        this.gpus.set(g.uuid, st);
      } else {
        dt = t - st.lastT;
        // Stale or out-of-order data: nothing was observed "continuously" across the gap.
        if (dt > maxGap || dt < 0) {
          resetTimers(st, dt);
          dt = 0;
        }
        st.lastT = t;
      }
      this.evaluate(g, st, t, dt, out);
    }
    return out;
  }

  /** Conditions that fired and have not cleared yet (for the status bar and the views). */
  active(): Alert[] {
    const out: Alert[] = [];
    for (const st of this.gpus.values()) {
      for (const l of [st.hot, st.vramFull, st.idleHeld]) if (l.fired && l.alert) out.push(l.alert);
    }
    return out;
  }

  private evaluate(g: GpuSample, st: GpuState, t: number, dt: number, out: Alert[]): void {
    const c = this.cfg;
    const pct = vramPct(g);
    const util = g.util;
    if (c.hot) {
      const temp = g.tempC;
      this.step(g, st, st.hot, 'hot', t, out, HOT_HOLD_MS,
        temp !== null && temp >= c.tempWarn,
        temp !== null && temp < c.tempWarn - HOT_CLEAR_MARGIN_C,
        () => ({ level: 'warning', title: `GPU ${g.index} running hot: ${Math.round(temp!)} °C`,
          detail: `At or above the ${c.tempWarn} °C warning threshold (gpuPulse.temperatureWarning) for over ${HOT_HOLD_MS / 1000} s.` }));
    }
    if (c.vramFull) {
      this.step(g, st, st.vramFull, 'vramFull', t, out, VRAM_HOLD_MS,
        pct !== null && pct >= VRAM_FULL_PCT,
        pct !== null && pct < VRAM_CLEAR_PCT,
        () => ({ level: 'warning',
          title: `GPU ${g.index} ${memName(g)} nearly full (${gb(g.memUsedMiB, 1)}/${gb(g.memTotalMiB, 0)} GB${g.unifiedMemory ? ' unified' : ''})`,
          detail: `Above ${VRAM_FULL_PCT}% for over ${VRAM_HOLD_MS / 1000} s; new allocations may fail with out-of-memory errors.` }));
    }
    if (c.idleHeld) {
      const holdMs = c.idleHeldMinutes * 60_000;
      this.step(g, st, st.idleHeld, 'idleHeld', t, out, holdMs,
        util !== null && pct !== null && util < IDLE_UTIL && pct >= IDLE_VRAM_PCT,
        (util !== null && util >= IDLE_CLEAR_UTIL) || (pct !== null && pct < IDLE_CLEAR_VRAM_PCT),
        () => ({ level: 'info',
          title: `GPU ${g.index} idle for ${Math.round((t - st.idleHeld.since!) / 60_000)} min while holding ${gb(g.memUsedMiB, 1)} GB of ${g.unifiedMemory ? 'unified memory' : 'VRAM'}`,
          detail: 'An open notebook kernel or a finished script may still hold this memory; free it to let others use the GPU.' }));
    }
    if (c.jobFinished) this.job(g, st, t, dt, out);
  }

  /** One latched rule: fire once after `holdMs` of continuous `cond`; re-arm only after `clear`. */
  private step(g: GpuSample, st: GpuState, l: Latch, rule: AlertRule, t: number, out: Alert[], holdMs: number, cond: boolean, clear: boolean,
    make: () => Pick<Alert, 'level' | 'title' | 'detail'>): void {
    if (cond) l.since ??= t;
    else l.since = null;
    if (l.fired) {
      if (clear) {
        l.fired = false;
        l.alert = null;
        this.rev++;
      }
      return;
    }
    if (l.since === null || t - l.since < holdMs) return;
    l.fired = true;
    this.rev++;
    l.alert = { rule, gpuUuid: g.uuid, gpuIndex: g.index, ...make() };
    this.emit(st, l.alert, t, out);
  }

  private job(g: GpuSample, st: GpuState, t: number, dt: number, out: Alert[]): void {
    const u = g.util;
    const w = u === null ? 0 : dt;
    if (u !== null && u >= BUSY_UTIL) {
      st.jobLowSince = null;
      if (st.jobStart === null || t - st.jobLastBusy >= JOB_END_MS) {
        // New stretch.
        st.jobStart = t;
        st.jobUtilMs = st.jobMs = 0;
      } else {
        // Bridge the dip since the last busy sample into this stretch.
        st.jobUtilMs += st.jobPendUtilMs + u * w;
        st.jobMs += st.jobPendMs + w;
      }
      st.jobPendUtilMs = st.jobPendMs = 0;
      st.jobLastBusy = t;
      return;
    }
    if (st.jobStart === null) return;
    if (u !== null) {
      st.jobPendUtilMs += u * w;
      st.jobPendMs += w;
    }
    if (u !== null && u < IDLE_UTIL) st.jobLowSince ??= t;
    else st.jobLowSince = null;
    if (st.jobLowSince === null) {
      // No quiet spell started within the bridge: the stretch did not end cleanly, forget it.
      if (t - st.jobLastBusy >= JOB_END_MS) dropJob(st);
      return;
    }
    if (t - st.jobLowSince < JOB_END_MS) return;
    const ranMs = st.jobLastBusy - st.jobStart;
    const avg = st.jobMs > 0 ? st.jobUtilMs / st.jobMs : null;
    dropJob(st);
    if (ranMs < this.cfg.jobFinishedMinutes * 60_000) return;
    this.emit(st, {
      rule: 'jobFinished', gpuUuid: g.uuid, gpuIndex: g.index, level: 'info',
      title: `Job on GPU ${g.index} finished after ${Math.round(ranMs / 60_000)} min (avg ${avg === null ? '–' : Math.round(avg)}%)`,
      detail: `Utilization stayed below ${IDLE_UTIL}% for ${JOB_END_MS / 1000} s after a busy stretch.`,
    }, t, out);
  }

  /** Apply the per-(rule, GPU) cooldown and queue the alert. */
  private emit(st: GpuState, a: Alert, t: number, out: Alert[]): void {
    const last = st.lastFired[a.rule];
    if (last !== undefined && t >= last && t - last < COOLDOWN_MS) return;
    st.lastFired[a.rule] = t;
    out.push(a);
  }
}
