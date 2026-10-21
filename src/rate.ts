export interface RateInputs {
  dashboardVisible: boolean;
  sidebarVisible: boolean;
  /** gpuPulse.recordInBackground */
  background: boolean;
  /** gpuPulse.statusBar.enabled */
  statusBar: boolean;
  fastMs: number;
  slowMs: number;
}

export interface SamplingPlan {
  /** Sample at all (false = pause the sampler so a laptop dGPU can sleep). */
  run: boolean;
  intervalMs: number;
  /** Poll the GPU process list (only the full dashboard shows it). */
  processes: boolean;
}

/**
 * One sampler serves every surface. Any visible live view (dashboard or side bar) gets the
 * fast interval — once, however many views are open; otherwise history and the status bar
 * are fed at the slow interval, or sampling pauses when nothing needs it.
 */
export function samplingPlan(i: RateInputs): SamplingPlan {
  const live = i.dashboardVisible || i.sidebarVisible;
  return {
    run: live || i.background || i.statusBar,
    intervalMs: live ? i.fastMs : i.slowMs,
    processes: i.dashboardVisible,
  };
}
