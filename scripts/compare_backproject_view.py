#!/usr/bin/env python
"""Generate side-by-side backproject/projection comparison for selected frames/views.

Compares:
  A) global conf + depth occlusion (old default)
  B) adaptive conf + no occlusion (new)

Example:
    .venv/bin/python scripts/compare_backproject_view.py --timestamps 000003 000004 --views 2-5
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.mask_io import VIEW_NAMES
from common.normalized_recon import load_pipeline, proj_vis_dir, resolve_backend
from scripts.backproject_normalized import backproject_timestamp, parts_ply_root
from scripts.visualize_raw_ply import visualize_timestamp


def load_overlay_only(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    # pair image: photo | gap | overlay
    w = img.shape[1]
    return img[:, w // 2 + 2:]


def make_panel(images: list[np.ndarray], labels: list[str], title: str) -> np.ndarray:
    h = max(im.shape[0] for im in images)
    cols = []
    for im, lab in zip(images, labels):
        if im.shape[0] != h:
            im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        cv2.rectangle(im, (0, 0), (im.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(im, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cols.append(im)
        cols.append(np.full((h, 4, 3), 255, np.uint8))
    row = np.hstack(cols[:-1])
    banner = np.zeros((40, row.shape[1], 3), np.uint8)
    cv2.putText(banner, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([banner, row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamps", nargs="+", default=["000003", "000004"])
    ap.add_argument("--views", nargs="+", default=["2-5"])
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline_normalized.json"))
    ap.add_argument("--backend", default="da3_self_cond")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.backend)
    parts = cfg.get("parts", ["lid", "body", "inner_pot"])

    # 1) adaptive backproject
    print("==> adaptive backproject")
    for ts in args.timestamps:
        print(ts)
        backproject_timestamp(
            cfg, backend, ts, parts,
            conf_mode="adaptive", conf_quantile=0.25,
            stride=2, max_pts=80000, tag="adaptive",
        )

    # 2) four projection variants
    variants = [
        ("global", None, "raw_global", False, "A: global conf + occlusion"),
        ("global", None, "raw_global_nocc", True, "B: global conf + no occlusion"),
        ("adaptive", "adaptive", "raw_adaptive", False, "C: adaptive conf + occlusion"),
        ("adaptive", "adaptive", "raw_adaptive_nocc", True, "D: adaptive conf + no occlusion"),
    ]

    for ts in args.timestamps:
        print(f"==> visualize {ts}")
        # ensure global ply exists (use existing default parts_ply)
        for _, ply_tag, subdir, no_occ, _label in variants:
            visualize_timestamp(
                cfg, backend, ts, parts,
                depth_tol=0.05, no_occlusion=no_occ,
                ply_tag=ply_tag, subdir=subdir, views=args.views,
            )

    out_root = args.out or os.path.join(
        proj_vis_dir(cfg, backend), "compare_backproject",
    )
    os.makedirs(out_root, exist_ok=True)

    for ts in args.timestamps:
        for view in args.views:
            vi = VIEW_NAMES.index(view)
            overlays = []
            labels = []
            for _, ply_tag, subdir, no_occ, label in variants:
                path = os.path.join(
                    proj_vis_dir(cfg, backend), subdir, ts, f"view{vi}_{view}.jpg",
                )
                overlays.append(load_overlay_only(path))
                labels.append(label)
            if any(x is None for x in overlays):
                raise FileNotFoundError(f"missing overlay for {ts} {view}")

            title = f"{ts} / {view}  (overlay only, left->right)"
            panel = make_panel(overlays, labels, title)
            out_path = os.path.join(out_root, f"{ts}_{view}_compare.jpg")
            cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"wrote {out_path}")

            # stats text
            old_ply = parts_ply_root(cfg, backend, None)
            new_ply = parts_ply_root(cfg, backend, "adaptive")
            for part in parts:
                def count_ply(root, ts, part):
                    p = os.path.join(root, ts, f"{part}.ply")
                    if not os.path.exists(p):
                        return 0
                    n = 0
                    with open(p) as f:
                        for line in f:
                            if line.strip() == "end_header":
                                break
                        for line in f:
                            if line.strip():
                                n += 1
                    return n
                o = count_ply(old_ply, ts, part)
                n = count_ply(new_ply, ts, part)
                print(f"  {part}: global_pts={o} adaptive_pts={n}")


if __name__ == "__main__":
    main()
