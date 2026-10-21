import * as os from 'os';
import si from 'systeminformation';
import type { Specs } from '../types';

function settled<T>(r: PromiseSettledResult<T>): T | undefined {
  return r.status === 'fulfilled' ? r.value : undefined;
}

function joinParts(...parts: (string | undefined | null)[]): string | null {
  const s = parts.filter((p) => p && p.trim() && !/^(default string|to be filled|system product name)/i.test(p)).join(' ');
  return s.trim() || null;
}

/** Collect static machine specs. Each probe is independent; failures leave fields blank. */
export async function collectSpecs(
  nvidia: { driver: string | null; cuda: string | null; vbios: string[] },
  remoteName: string | undefined,
): Promise<Specs> {
  const [cpuR, osR, memR, layoutR, sysR, boardR, diskR, gfxR] = await Promise.allSettled([
    si.cpu(),
    si.osInfo(),
    si.mem(),
    si.memLayout(),
    si.system(),
    si.baseboard(),
    si.diskLayout(),
    si.graphics(),
  ]);
  const cpu = settled(cpuR);
  const osi = settled(osR);
  const mem = settled(memR);
  const layout = settled(layoutR) ?? [];
  const sys = settled(sysR);
  const board = settled(boardR);
  const disks = settled(diskR) ?? [];
  const gfx = settled(gfxR);

  const modules = layout
    .filter((m) => m.size > 0)
    .map((m) =>
      [`${Math.round(m.size / 1073741824)} GB`, m.type, m.clockSpeed ? `${m.clockSpeed} MT/s` : '', m.manufacturer]
        .filter(Boolean)
        .join(' · '),
    );

  // Virtual adapters (RDP, basic display) are not real GPUs.
  const controllers = (gfx?.controllers ?? []).filter(
    (c) => c.model && !/microsoft (remote|basic)|virtual|vmware svga/i.test(c.model),
  );
  let nvIndex = 0;
  const gpus = controllers.map((c) => {
    const isNv = /nvidia/i.test(`${c.vendor} ${c.model}`);
    const vbios = isNv ? nvidia.vbios[nvIndex++] : undefined;
    return {
      name: c.model,
      vramMiB: c.memoryTotal || c.vram || null,
      driver: c.driverVersion || (isNv ? nvidia.driver : null),
      vbios: vbios || null,
      extra: [c.vendor, c.bus].filter((x): x is string => !!x),
    };
  });

  return {
    hostname: os.hostname(),
    os: joinParts(osi?.distro, osi?.release) ?? `${os.type()} ${os.release()}`,
    kernel: osi?.kernel ?? os.release(),
    arch: osi?.arch ?? os.arch(),
    uptimeSec: os.uptime(),
    cpu: {
      model: joinParts(cpu?.manufacturer, cpu?.brand) ?? os.cpus()[0]?.model ?? 'Unknown CPU',
      cores: cpu?.physicalCores ?? os.cpus().length,
      threads: cpu?.cores ?? os.cpus().length,
      baseGHz: cpu?.speed || null,
      maxGHz: cpu?.speedMax || null,
    },
    memory: { totalBytes: mem?.total ?? os.totalmem(), modules },
    board: joinParts(board?.manufacturer, board?.model),
    system: joinParts(sys?.manufacturer, sys?.model),
    disks: disks.map((d) => ({ name: d.name || d.device, sizeBytes: d.size, type: d.type || d.interfaceType })),
    gpus,
    cudaVersion: nvidia.cuda,
    remote: remoteName ?? null,
  };
}
