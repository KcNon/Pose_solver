#!/usr/bin/env python
"""Pack ICP projection montages into a single MP4 video (chronological by src frame).

Reads pair order from icp/.../chain_summary.json, loads each
proj_vis/{backend}/{ref}_to_{src}/montage.jpg, writes video under
outputs/normalized/videos/.

Example:
    .venv/bin/python scripts/make_icp_video.py --pipeline configs/pipeline_normalized.json
    .venv/bin/python scripts/make_icp_video.py --backend da3_self_cond --fps 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.normalized_recon import (
    icp_out_dir,
    load_pipeline,
    output_root,
    proj_vis_dir,
    resolve_backend,
)


def pair_sort_key(pair_key: str) -> int:
    m = re.match(r"(\d+)_to_", pair_key)
    return int(m.group(1)) if m else 0


def collect_pairs(cfg, backend: str) -> list[tuple[str, str, str]]:
    """Return [(ref, src, pair_key), ...] in chronological order."""
    summary_path = os.path.join(icp_out_dir(cfg, backend), "chain_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"chain_summary.json not found: {summary_path}")
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    pairs: list[tuple[str, str, str]] = []
    for seg in summary.get("segments", []):
        for pair_key in seg.get("pairs", []):
            src, ref = pair_key.split("_to_")
            pairs.append((ref, src, pair_key))
    pairs.sort(key=lambda x: pair_sort_key(x[2]))
    return pairs


def load_montage(cfg, backend: str, ref: str, src: str) -> np.ndarray | None:
    path = os.path.join(proj_vis_dir(cfg, backend), f"{ref}_to_{src}", "montage.jpg")
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img


def annotate(img: np.ndarray, ref: str, src: str, idx: int, total: int) -> np.ndarray:
    out = img.copy()
    label = f"[{idx + 1}/{total}] {src} -> {ref}  (ICP forward/backward)"
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_video_cv2(frames: list[np.ndarray], out_path: str, fps: float) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed: {out_path}")
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        writer.write(f)
    writer.release()


def write_video_ffmpeg(frames: list[np.ndarray], out_path: str, fps: float) -> None:
    tmp_dir = out_path + ".frames"
    os.makedirs(tmp_dir, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    pattern = os.path.join(tmp_dir, "%05d.jpg")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path,
    ]
    subprocess.run(cmd, check=True)
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline_normalized.json"))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--fps", type=float, default=2.0, help="frames per second in output video")
    ap.add_argument("--out", default=None, help="output mp4 path")
    ap.add_argument("--use-ffmpeg", action="store_true", help="prefer ffmpeg libx264 (better compatibility)")
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.backend)
    pairs = collect_pairs(cfg, backend)
    if not pairs:
        raise SystemExit("no pairs found in chain_summary.json")

    frames: list[np.ndarray] = []
    missing = []
    for i, (ref, src, pair_key) in enumerate(pairs):
        img = load_montage(cfg, backend, ref, src)
        if img is None:
            missing.append(pair_key)
            continue
        frames.append(annotate(img, ref, src, i, len(pairs)))

    if missing:
        print(f"warning: {len(missing)} montages missing (run visualize_icp_chain.py first)")
        for k in missing[:5]:
            print(f"  missing: {k}")
        if not frames:
            raise SystemExit("no montage images found")

    vid_dir = os.path.join(output_root(cfg), "videos")
    os.makedirs(vid_dir, exist_ok=True)
    out_path = args.out or os.path.join(vid_dir, f"{backend}_icp_chain.mp4")

    if args.use_ffmpeg:
        write_video_ffmpeg(frames, out_path, args.fps)
    else:
        write_video_cv2(frames, out_path, args.fps)
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            print("cv2 mp4v failed or tiny file, retrying with ffmpeg...")
            write_video_ffmpeg(frames, out_path, args.fps)

    print(f"wrote {out_path}  ({len(frames)} frames @ {args.fps} fps, ~{len(frames)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
