#!/usr/bin/env python
"""Export per-view per-part images with black background from saved masks.

Uses depth-resolution masks (upscaled to full frame) when --skip-seg was used.
For best quality, re-run seg_backproject_parts.py without --skip-seg.

Example:
    .venv/bin/python scripts/export_part_images.py --timestamps 000029 000019
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.seg_backproject_parts import (
    OUT_MASK,
    OUT_PART_IMAGES,
    VIEW_NAMES,
    load_parts_config,
    part_on_black,
)

MASK_DIR = OUT_MASK
OUT_DIR = OUT_PART_IMAGES


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def upscale_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    up = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return up.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamps", nargs="+", required=True)
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline.json"))
    ap.add_argument("--mask-dir", default=MASK_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--frames-dir", default=None, help="override pipeline frames_dir")
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    frames_base = args.frames_dir or cfg["frames_dir"]

    for ts in args.timestamps:
        parts, _ = load_parts_config(ts)
        mask_root = os.path.join(args.mask_dir, ts)
        out_root = os.path.join(args.out_dir, ts)
        os.makedirs(out_root, exist_ok=True)

        for v, vname in enumerate(VIEW_NAMES):
            frame_path = os.path.join(frames_base, ts, f"{vname}.png")
            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is None:
                raise FileNotFoundError(frame_path)
            H, W = frame_bgr.shape[:2]

            for pname in parts:
                mp = os.path.join(mask_root, f"{v}_{pname}.png")
                if not os.path.exists(mp):
                    print(f"skip missing mask: {mp}")
                    continue
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
                m_full = upscale_mask(m, (H, W))
                out = part_on_black(frame_bgr, m_full)
                out_path = os.path.join(out_root, f"{vname}_{pname}.png")
                cv2.imwrite(out_path, out)
                print(f"saved {out_path}  px={int(m_full.sum())}")

    print("done")


if __name__ == "__main__":
    main()
