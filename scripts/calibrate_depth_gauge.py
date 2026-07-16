#!/usr/bin/env python
"""Calibrate the per-frame depth gauge from the static reference part.

Writes ``depth_gauge.json`` next to the other diagnostics; feed it to
``backproject_normalized.py --depth-gauge`` to remove the per-frame global
depth drift before the part clouds are fused.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.depth_gauge import compute_depth_gauge, compute_view_bias
from common.io_utils import write_json
from common.normalized_recon import all_timestamps, load_pipeline, output_root, resolve_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pose_multiview_111.json"))
    parser.add_argument("--backend", default=None)
    parser.add_argument("--part", default=None, help="static reference part; defaults to config reference_part")
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--min-pixels", type=int, default=300)
    parser.add_argument("--cross-view", action="store_true",
                        help="additionally align the views to each other with one "
                             "constant depth bias per view")
    parser.add_argument("--cross-view-stride", type=int, default=5,
                        help="frame subsampling for the cross-view estimate")
    parser.add_argument("--out", default=None, help="defaults to <output_root>/diagnostics/depth_gauge.json")
    args = parser.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.backend)
    if "frames" in cfg and isinstance(cfg["frames"], dict):
        start, end = int(cfg["frames"]["start"]), int(cfg["frames"]["end"])
        timestamps = [f"{f:06d}" for f in range(start, end + 1)]
    else:
        timestamps = all_timestamps(cfg)

    gauge = compute_depth_gauge(
        cfg, backend, timestamps, args.part,
        min_support=args.min_support, min_pixels=args.min_pixels)

    for v, (std, rng) in enumerate(zip(gauge["shift_std_mm"], gauge["shift_range_mm"])):
        interpolated = sum(entry["interpolated"][v] for entry in gauge["frames"].values())
        print(f"view {v}: shift std {std:.2f} mm, range [{rng[0]:+.2f}, {rng[1]:+.2f}] mm, "
              f"{interpolated}/{len(timestamps)} frames interpolated")

    if args.cross_view:
        bias = compute_view_bias(cfg, backend, timestamps[::args.cross_view_stride],
                                 args.part, gauge)
        gauge["view_bias_m"] = bias["view_bias_m"]
        gauge["view_bias_info"] = {k: v for k, v in bias.items() if k != "view_bias_m"}
        for v, (b, spread) in enumerate(zip(bias["view_bias_m"], bias["per_frame_spread_mm"])):
            print(f"view {v}: cross-view bias {b * 1000:+.2f} mm "
                  f"(per-frame spread {spread:.2f} mm over {bias['n_frames_used']} frames)")

    out = args.out or os.path.join(output_root(cfg), "diagnostics", "depth_gauge.json")
    write_json(out, gauge)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
