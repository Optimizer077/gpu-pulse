"""Terminal renderers: sizes stay exact so panels never wrap."""

from __future__ import annotations

import time

from gpu_pulse import render as R
from gpu_pulse.collectors.amd_linux import AmdLinuxCollector


def test_bar_is_exact_width() -> None:
    for frac in (0, 0.01, 0.333, 0.5, 0.999, 1, None, 1.7):
        assert R.bar(frac, 20, R.GPU, R.GPU_GLOW).cell_len == 20


def test_resample_averages_and_keeps_gaps() -> None:
    assert R.resample([0, 100, 50, 50], 2) == [50, 50]
    assert R.resample([None, None, 10, 30], 2) == [None, 20]
    assert len(R.resample([1, 2, 3], 7)) == 7


def test_area_chart_shape() -> None:
    lines = R.area_chart([0, 50, 100, None], 25, 6, R.GPU, R.GPU_GLOW).plain.split("\n")
    assert len(lines) == 6
    assert all(len(line) == 25 for line in lines)
    assert lines[0].strip().startswith("100%")


def test_heatmap_rows() -> None:
    now = time.time()
    start = int(now // 60 * 60_000) - 1499 * 60_000
    text = R.heatmap(start, [50.0] * 1500, now=now).plain.split("\n")
    assert len(text) == 13  # header + 12 double-hour rows
    assert text[-1].strip().startswith("now")


def test_heatmap_tall_and_wide() -> None:
    now = time.time()
    start = int(now // 60 * 60_000) - 1499 * 60_000
    text = R.heatmap(start, [50.0] * 1500, now=now, col_w=2, tall=True).plain.split("\n")
    assert len(text) == 25  # header + one row per hour
    assert all(len(line) <= 6 + 120 for line in text)


def test_braille_traces_exact_size() -> None:
    from gpu_pulse import anim

    vals = [None, 0.0, 5.0, 100.0, 40.0]
    for h in (1, 2, 4):
        lines = anim.trace(vals, 30, h, R.GPU, R.GPU_GLOW).plain.split("\n")
        assert len(lines) == h and all(len(x) == 30 for x in lines)
    m = anim.mirror(vals, vals[::-1], 25, 4, (R.DISK, R.DISK_GLOW), (R.NET, R.NET_GLOW)).plain.split("\n")
    assert len(m) == 4 and all(len(x) == 25 for x in m)


def test_heat_color_monotonic_brightness() -> None:
    lum = [sum(R._rgb(R.heat_color(v))) for v in (0, 25, 50, 75, 100)]
    assert lum == sorted(lum)


def test_amd_sysfs_units(tmp_path) -> None:  # noqa: ANN001
    dev = tmp_path / "card0" / "device"
    hw = dev / "hwmon" / "hwmon3"
    hw.mkdir(parents=True)
    (dev / "gpu_busy_percent").write_text("42\n")
    (dev / "mem_info_vram_used").write_text(str(2 * 1048576 * 1024))
    (dev / "mem_info_vram_total").write_text(str(16 * 1048576 * 1024))
    (hw / "temp1_input").write_text("65000")
    (hw / "power1_average").write_text("150000000")
    c = AmdLinuxCollector(root=tmp_path)
    assert c.detect()
    (g,) = c.sample()
    assert (g.util, g.mem_used_mib, g.mem_total_mib, g.temp_c, g.power_w) == (42, 2048, 16384, 65, 150)
