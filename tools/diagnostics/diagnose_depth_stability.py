#!/usr/bin/env python
"""Quantify temporal stability of the recon depth on a never-occluded region.

The static reference part (body) is visible in every frame.  Pixels whose part
mask survives an AND across all frames were never occluded, so any temporal
depth variation there is reconstruction noise, not scene motion.  The variation
is decomposed into a per-frame global component (a shift and a gain, i.e. the
part of the error a per-frame gauge could remove) and the pixel-level residual
that no rigid-pose machinery downstream can ever undo.

Numbers are reported in millimetres so they can be compared directly with the
registration thresholds (8 mm fitness gate, 3-12 mm voxels).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.backproject_utils import load_palette_masks
from common.depth_gauge import apply_depth_gauge, load_depth_gauge
from common.io_utils import load_json, write_json
from common.normalized_recon import load_recon


def robust_std(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """1.4826 * MAD, insensitive to the occasional depth spike."""
    med = np.nanmedian(values, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(values - med), axis=axis)


def analyze_view(depth_stack: np.ndarray, valid_stack: np.ndarray,
                 stable: np.ndarray, min_frame_pixels: int) -> dict | None:
    """Decompose temporal variation on pixels supported in most frames."""
    if stable.sum() < 200:
        return None
    D = depth_stack[:, stable].astype(np.float64)  # (F, N)
    D[~valid_stack[:, stable]] = np.nan
    support = np.isfinite(D).sum(axis=1)
    frame_supported = support >= min_frame_pixels
    D[~frame_supported] = np.nan
    with warnings.catch_warnings():
        # Unsupported frames are intentionally all-NaN and interpolated below.
        warnings.simplefilter("ignore", RuntimeWarning)
        reference = np.nanmedian(D, axis=0)         # per-pixel temporal median
        bias = np.nanmedian(D - reference, axis=1)  # per-frame shift (m)
        gain = np.nanmedian(D / reference, axis=1)  # multiplicative drift
    measured = np.isfinite(bias) & np.isfinite(gain) & frame_supported
    if not measured.any():
        return None
    index = np.arange(len(bias))
    bias[~measured] = np.interp(index[~measured], index[measured], bias[measured])
    gain[~measured] = np.interp(index[~measured], index[measured], gain[measured])

    raw_std = robust_std(D, axis=0)
    debias_std = robust_std(D - bias[:, None], axis=0)
    degain_std = robust_std(D / gain[:, None], axis=0)

    mm = 1000.0
    return {
        "n_pixels": int(stable.sum()),
        "n_observations": int(np.isfinite(D).sum()),
        "median_depth_m": float(np.nanmedian(reference)),
        "per_pixel_temporal_std_mm": {
            "raw": float(np.nanmedian(raw_std) * mm),
            "after_per_frame_shift": float(np.nanmedian(debias_std) * mm),
            "after_per_frame_gain": float(np.nanmedian(degain_std) * mm),
            "raw_p95": float(np.nanquantile(raw_std, 0.95) * mm),
        },
        "per_frame_shift_mm": {
            "std": float(np.std(bias) * mm),
            "min": float(bias.min() * mm),
            "max": float(bias.max() * mm),
            "series": [float(b * mm) for b in bias],
            "support_pixels": [int(n) for n in support],
            "interpolated": [bool(not value) for value in measured],
        },
        "per_frame_gain_pct": {
            "std": float(np.std(gain - 1.0) * 100),
            "min": float((gain.min() - 1.0) * 100),
            "max": float((gain.max() - 1.0) * 100),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", default=None, help="never-occluded part; defaults to reference_part")
    parser.add_argument("--erode", type=int, default=2, help="mask erosion iterations at depth resolution")
    parser.add_argument("--min-support-ratio", type=float, default=0.7,
                        help="pixel must be valid in at least this fraction of frames")
    parser.add_argument("--min-frame-pixels", type=int, default=300,
                        help="minimum stable-region overlap before a frame shift is measured")
    parser.add_argument("--depth-gauge", default=None,
                        help="optional depth_gauge.json applied before measuring residual stability")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    part = args.part or cfg["reference_part"]
    views = cfg["views"]
    backend = cfg["recon_backend"]
    start, end = int(cfg["frames"]["start"]), int(cfg["frames"]["end"])
    frames = [f"{f:06d}" for f in range(start, end + 1)]

    depth_stack = None
    valid_stack = None
    kernel = np.ones((3, 3), np.uint8)
    gauge = load_depth_gauge(args.depth_gauge) if args.depth_gauge else None
    for index, timestamp in enumerate(frames):
        recon = load_recon(cfg, timestamp, backend=backend)
        depth = recon["depth"]  # (V, H, W)
        if gauge is not None:
            depth = apply_depth_gauge(depth, gauge, timestamp)
        if depth_stack is None:
            depth_stack = np.empty((len(frames),) + depth.shape, np.float32)
            valid_stack = np.zeros((len(frames),) + depth.shape, bool)
        depth_stack[index] = depth
        masks = load_palette_masks(
            cfg["masks_dir"], timestamp, [part], recon["depth_hw"],
            views=cfg.get("views"),
            part_ids=cfg.get("part_ids"),
        )[part]
        for v in range(depth.shape[0]):
            valid_stack[index, v] = masks[v] & np.isfinite(depth[v]) & (depth[v] > 1e-3)
        if index % 20 == 0:
            print(f"loaded {timestamp} ({index + 1}/{len(frames)})", flush=True)

    report = {"config": args.config, "backend": backend, "part": part,
              "frames": [start, end], "min_support_ratio": args.min_support_ratio,
              "min_frame_pixels": args.min_frame_pixels,
              "depth_gauge": args.depth_gauge, "views": {}}
    weighted = []
    min_support = int(np.ceil(args.min_support_ratio * len(frames)))
    for v, view in enumerate(views):
        stable = valid_stack[:, v].sum(axis=0) >= min_support
        stable = cv2.erode(stable.astype(np.uint8), kernel, iterations=args.erode).astype(bool)
        stats = analyze_view(depth_stack[:, v], valid_stack[:, v], stable,
                             args.min_frame_pixels)
        if stats is None:
            print(f"{view}: stable region too small ({int(stable.sum())} px), skipped")
            continue
        report["views"][view] = stats
        px = stats["per_pixel_temporal_std_mm"]
        sh = stats["per_frame_shift_mm"]
        print(f"{view}: {stats['n_pixels']:6d} px @ {stats['median_depth_m']:.3f} m | "
              f"raw {px['raw']:.2f} mm -> shift-removed {px['after_per_frame_shift']:.2f} mm "
              f"(gain-removed {px['after_per_frame_gain']:.2f} mm) | "
              f"frame shift [{sh['min']:+.2f}, {sh['max']:+.2f}] mm std {sh['std']:.2f}",
              flush=True)
        weighted.append((stats["n_pixels"], px["raw"], px["after_per_frame_shift"], sh["std"]))

    if weighted:
        n = np.asarray([w[0] for w in weighted], float)
        summary = {
            "raw_std_mm": float(np.average([w[1] for w in weighted], weights=n)),
            "residual_std_mm": float(np.average([w[2] for w in weighted], weights=n)),
            "frame_shift_std_mm": float(np.average([w[3] for w in weighted], weights=n)),
        }
        report["summary"] = summary
        print(f"\nsummary (pixel-weighted): raw temporal std {summary['raw_std_mm']:.2f} mm, "
              f"per-frame global shift std {summary['frame_shift_std_mm']:.2f} mm, "
              f"irreducible pixel residual {summary['residual_std_mm']:.2f} mm")

    default_name = "depth_stability_gauge.json" if gauge is not None else "depth_stability.json"
    out = Path(args.out) if args.out else Path(cfg["output_root"]) / "diagnostics" / default_name
    write_json(out, report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
