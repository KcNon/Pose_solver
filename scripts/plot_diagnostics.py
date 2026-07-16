#!/usr/bin/env python
"""Plot the depth-stability and part-state diagnostics as PNG figures.

Reads the JSON written by ``diagnose_depth_stability.py`` and
``detect_part_states.py`` and renders:

    diagnostics/depth_stability.png         per-frame shift series + noise bars
    diagnostics/depth_stability_region.png  never-occluded region on the RGB frame
    diagnostics/part_states.png             state timeline + motion signals

Run with any venv providing matplotlib/cv2 (pose_solver's venv does not), e.g.:

    /data_ft_9_10/wentai/projects/depth-anything-3/.venv/bin/python \
        scripts/plot_diagnostics.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io_utils import load_json

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e7e6e2"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"
NEUTRAL = "#d9d8d3"
NEUTRAL_DARK = "#8a8983"

STATE_COLORS = {
    "moving": BLUE,
    "static": NEUTRAL,
    "occluded": YELLOW,
    "assembled": AQUA,
    "unobserved": NEUTRAL_DARK,
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": INK_2,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def style_axis(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def plot_depth_stability(report: dict, out_png: Path) -> None:
    views = report["views"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4), dpi=150, gridspec_kw={"width_ratios": [2.2, 1.0]})

    palette = [BLUE, AQUA, YELLOW, RED]
    for i, (view, stats) in enumerate(views.items()):
        series = stats["per_frame_shift_mm"]["series"]
        ax1.plot(series, color=palette[i % len(palette)], linewidth=2, label=view)
        ax1.annotate(view, (len(series) - 1, series[-1]),
                     xytext=(6, 0), textcoords="offset points",
                     color=INK_2, fontsize=9, va="center")
    ax1.axhline(0, color=INK_2, linewidth=0.8)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("per-frame global depth shift (mm)")
    ax1.set_title("DA3 depth: per-frame global shift on the never-occluded body region",
                  loc="left", fontsize=11)
    ax1.legend(frameon=False, loc="upper left")
    style_axis(ax1)

    labels = list(views)
    raw = [views[v]["per_pixel_temporal_std_mm"]["raw"] for v in labels]
    residual = [views[v]["per_pixel_temporal_std_mm"]["after_per_frame_shift"] for v in labels]
    x = np.arange(len(labels))
    width = 0.32
    bars1 = ax2.bar(x - width / 2, raw, width, color=BLUE, label="raw")
    bars2 = ax2.bar(x + width / 2, residual, width, color=AQUA, label="shift removed")
    for bars in (bars1, bars2):
        for b in bars:
            ax2.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", color=INK_2, fontsize=9)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("temporal std per pixel (mm)")
    ax2.set_title("noise before/after per-frame correction", loc="left", fontsize=11)
    ax2.legend(frameon=False)
    style_axis(ax2)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_raw_vs_gauge(report: dict, out_png: Path) -> None:
    """Compact acceptance plot for the gauge A/B experiment."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
    ax_summary, ax_views, ax_cloud = axes

    keys = ["per_pixel_raw_std_mm", "per_frame_shift_std_mm",
            "shift_removed_residual_std_mm"]
    labels = ["pixel temporal", "global shift", "residual floor"]
    raw = [report["depth_temporal"][key]["raw"] for key in keys]
    gauge = [report["depth_temporal"][key]["gauge"] for key in keys]
    x = np.arange(len(keys))
    width = 0.34
    ax_summary.bar(x - width / 2, raw, width, color=BLUE, label="raw")
    ax_summary.bar(x + width / 2, gauge, width, color=AQUA, label="gauge")
    ax_summary.set_xticks(x, labels, rotation=18, ha="right")
    ax_summary.set_ylabel("mm")
    ax_summary.set_title("fixed-region temporal depth", loc="left", fontsize=11)
    ax_summary.legend(frameon=False)
    style_axis(ax_summary)

    views = list(report["per_view"])
    raw_view = [report["per_view"][view]["raw_std_mm"] for view in views]
    gauge_view = [report["per_view"][view]["gauge_std_mm"] for view in views]
    x = np.arange(len(views))
    ax_views.bar(x - width / 2, raw_view, width, color=BLUE, label="raw")
    ax_views.bar(x + width / 2, gauge_view, width, color=AQUA, label="gauge")
    ax_views.set_xticks(x, views)
    ax_views.set_ylabel("temporal std (mm)")
    ax_views.set_title("all six views", loc="left", fontsize=11)
    style_axis(ax_views)

    cloud_keys = ["centroid_spread_visibility_sensitive",
                  "centroid_step_visibility_sensitive",
                  "surface_nn_step_visibility_sensitive"]
    cloud_labels = ["centroid spread", "centroid step", "surface NN step"]
    raw_cloud = [report["raw_cloud"][key]["median_mm"] for key in cloud_keys]
    gauge_cloud = [report["gauge_cloud"][key]["median_mm"] for key in cloud_keys]
    x = np.arange(len(cloud_keys))
    ax_cloud.bar(x - width / 2, raw_cloud, width, color=BLUE, label="raw")
    ax_cloud.bar(x + width / 2, gauge_cloud, width, color=AQUA, label="gauge")
    ax_cloud.set_xticks(x, cloud_labels, rotation=18, ha="right")
    ax_cloud.set_ylabel("mm")
    ax_cloud.set_title("fused cloud (visibility-sensitive)", loc="left", fontsize=11)
    style_axis(ax_cloud)

    fig.suptitle("DA3 depth gauge: raw vs corrected", x=0.02, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_region_overlay(cfg: dict, report: dict, out_png: Path, erode: int) -> None:
    import cv2
    from PIL import Image

    frames = [f"{f:06d}" for f in range(report["frames"][0], report["frames"][1] + 1)]
    part_id = 2  # body in the palette PNGs
    panels = []
    min_support = int(np.ceil(report.get("min_support_ratio", 1.0) * len(frames)))
    for view in report["views"]:
        support = None
        for timestamp in frames:
            labels = np.asarray(Image.open(Path(cfg["masks_dir"]) / timestamp / f"{view}.png"))
            m = labels == part_id
            support = m.astype(np.uint16) if support is None else support + m
        stable = support >= min_support
        stable = cv2.erode(stable.astype(np.uint8), np.ones((5, 5), np.uint8),
                           iterations=erode).astype(bool)
        rgb = cv2.imread(str(Path(cfg["frames_dir"]) / view / f"{frames[0]}.jpg"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(float)
        tint = np.array([42, 120, 214], float)
        rgb[stable] = 0.4 * rgb[stable] + 0.6 * tint
        contours, _ = cv2.findContours(stable.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        rgb = rgb.astype(np.uint8).copy()
        cv2.drawContours(rgb, contours, -1, (255, 255, 255), 2)
        cv2.putText(rgb, view, (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        panels.append(rgb)
    montage = np.concatenate(panels, axis=1)
    cv2.imwrite(str(out_png), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"wrote {out_png}")


def state_runs(states: list[str]) -> list[tuple[int, int, str]]:
    runs = []
    for f, state in enumerate(states):
        if runs and runs[-1][2] == state and runs[-1][0] + runs[-1][1] == f:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1, state)
        else:
            runs.append((f, 1, state))
    return runs


def plot_part_states(report: dict, out_png: Path) -> None:
    parts = list(report["parts"])
    start = report["frames"][0]
    thresholds = report["thresholds"]
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 10), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0, 1.0]})
    ax_tl, ax_px, ax_lid, ax_pot = axes

    for row, part in enumerate(parts):
        info = report["parts"][part]
        states = [info["states"][k]["state"] for k in sorted(info["states"])]
        for f0, length, state in state_runs(states):
            ax_tl.broken_barh([(start + f0, length)], (row - 0.3, 0.6),
                              facecolors=STATE_COLORS[state], edgecolor="none")
        for a, b in info["manual_dynamic_ranges"]:
            ax_tl.plot([a, b], [row + 0.42, row + 0.42], color=INK, linewidth=1.6)
            ax_tl.plot([a, a], [row + 0.36, row + 0.48], color=INK, linewidth=1.6)
            ax_tl.plot([b, b], [row + 0.36, row + 0.48], color=INK, linewidth=1.6)
    ax_tl.set_yticks(range(len(parts)), parts)
    ax_tl.set_ylim(len(parts) - 0.4, -0.6)
    ax_tl.set_title("detected states (bars) vs manual dynamic ranges (brackets)",
                    loc="left", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in STATE_COLORS.values()]
    ax_tl.legend(handles, list(STATE_COLORS), frameon=False, ncol=5,
                 loc="lower left", bbox_to_anchor=(0.0, -0.32), fontsize=9)
    ax_tl.grid(False)

    def signal(part: str, key: str) -> np.ndarray:
        info = report["parts"][part]["states"]
        return np.asarray([
            np.nan if info[k][key] is None else info[k][key] for k in sorted(info)])

    def shade_moving(ax, part: str) -> None:
        for a, b in report["parts"][part]["detected_moving_ranges"]:
            ax.axvspan(a, b, color=BLUE, alpha=0.10, linewidth=0)

    ax_px.plot(np.arange(len(signal("lid", "motion_px"))) + start,
               signal("lid", "motion_px"), color=BLUE, linewidth=2)
    ax_px.axhline(thresholds["disp_hi"], color=INK_2, linewidth=1, linestyle="--")
    ax_px.axhline(thresholds["disp_lo"], color=INK_2, linewidth=1, linestyle=":")
    shade_moving(ax_px, "lid")
    ax_px.set_ylabel("lid envelope shift (px/frame)")
    ax_px.set_title("lid: occlusion-gated 2D envelope displacement "
                    "(dashed: enter, dotted: exit)", loc="left", fontsize=11)
    style_axis(ax_px)

    for ax, part in ((ax_lid, "lid"), (ax_pot, "inner_pot")):
        values = signal(part, "surface_shift_mm")
        ax.plot(np.arange(len(values)) + start, values, color=AQUA, linewidth=2)
        ax.axhline(thresholds["surf_hi_mm"], color=INK_2, linewidth=1, linestyle="--")
        ax.axhline(thresholds["surf_lo_mm"], color=INK_2, linewidth=1, linestyle=":")
        ceiling = 40.0
        if np.nanmax(values) > ceiling:
            ax.set_ylim(0, ceiling)
            f_peak = int(np.nanargmax(values)) + start
            ax.annotate(f"peak {np.nanmax(values):.0f} mm @ {f_peak} (clipped)",
                        (f_peak, ceiling), xytext=(8, -14), textcoords="offset points",
                        color=INK_2, fontsize=9)
        shade_moving(ax, part)
        ax.set_ylabel(f"{part} surface shift (mm)")
        ax.set_title(f"{part}: 3D cloud-to-previous-cloud median NN distance",
                     loc="left", fontsize=11)
        style_axis(ax)
    ax_pot.set_xlabel("frame")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pose_multiview_111.json"))
    parser.add_argument("--erode", type=int, default=2)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    diag = Path(cfg["output_root"]) / "diagnostics"
    support_report = diag / "depth_stability_support70_raw.json"
    depth_report = load_json(support_report if support_report.exists()
                             else diag / "depth_stability.json")
    states_report = load_json(diag / "part_states.json")

    plot_depth_stability(depth_report, diag / "depth_stability.png")
    plot_region_overlay(cfg, depth_report, diag / "depth_stability_region.png", args.erode)
    plot_part_states(states_report, diag / "part_states.png")
    comparison = diag / "raw_vs_gauge.json"
    if comparison.exists():
        plot_raw_vs_gauge(load_json(comparison), diag / "raw_vs_gauge.png")


if __name__ == "__main__":
    main()
