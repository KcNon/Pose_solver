#!/usr/bin/env python
"""SAM3 part segmentation at full resolution; save palette-indexed PNG masks.

Reads bboxes from masks/bbox.json (normalized layout). No depth backprojection.

Example:
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/seg_masks_only.py \\
        --pipeline configs/pipeline_normalized.json --timestamp 000000
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/seg_masks_only.py \\
        --pipeline configs/pipeline_normalized.json --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.mask_io import (
    VIEW_NAMES,
    DEFAULT_PARTS,
    DEFAULT_PROMPTS,
    frame_path,
    list_timestamps,
    load_bbox_json,
    qwen_bboxes_from_json,
    save_palette_png,
    masks_to_label_map,
)
from scripts.seg_backproject_parts import segment_view


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def process_timestamp(processor, bf16, args, cfg, timestamp: str) -> None:
    layout = cfg.get("frames_layout", "legacy")
    if layout != "normalized":
        raise ValueError("seg_masks_only.py requires frames_layout=normalized")
    frames_dir = cfg["frames_dir"]
    masks_dir = cfg["masks_dir"]
    parts = cfg.get("parts", DEFAULT_PARTS)
    prompts = cfg.get("prompts", DEFAULT_PROMPTS)
    for p in parts:
        prompts.setdefault(p, DEFAULT_PROMPTS.get(p, [p.replace("_", " ")]))

    out_dir = os.path.join(masks_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    for vname in VIEW_NAMES:
        frame_path_str = frame_path(frames_dir, layout, timestamp, vname)
        if not os.path.exists(frame_path_str):
            raise FileNotFoundError(frame_path_str)

        frame_bgr = cv2.imread(frame_path_str)
        if frame_bgr is None:
            raise RuntimeError(f"failed to read {frame_path_str}")

        qwen_bboxes = None if args.text_only else qwen_bboxes_from_json(masks_dir, timestamp, vname)
        masks, meta, _ = segment_view(
            processor, frame_bgr, parts, prompts, args.conf, bf16, qwen_bboxes
        )
        label = masks_to_label_map(masks, parts)
        png_path = os.path.join(out_dir, f"{vname}.png")
        save_palette_png(label, png_path, parts)

        info = {p: meta[p] for p in parts}
        info["cooker"] = meta.get("cooker", {})
        print(f"{timestamp} {vname}: saved {png_path}  {info}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--timestamp")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline_normalized.json"))
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--text-only", action="store_true")
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    layout = cfg.get("frames_layout", "legacy")
    frames_dir = cfg["frames_dir"]
    masks_dir = cfg["masks_dir"]
    sam_ckpt = cfg["sam_ckpt"]

    if args.all:
        timestamps = list_timestamps(frames_dir, layout)
        print(f"Segmenting {len(timestamps)} timestamps")
    else:
        timestamps = [args.timestamp]

    bbox_path = os.path.join(masks_dir, "bbox.json")
    if not args.text_only and not os.path.exists(bbox_path):
        raise FileNotFoundError(f"bbox.json not found: {bbox_path} (run detect_bbox_batch.py first)")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=sam_ckpt, load_from_HF=False, device="cuda", eval_mode=True
    )
    processor = Sam3Processor(model, confidence_threshold=args.conf)
    bf16 = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    data = load_bbox_json(masks_dir)
    print(f"parts: {json.dumps(data.get('parts', {}), ensure_ascii=False)}")

    for i, ts in enumerate(timestamps):
        out_dir = os.path.join(masks_dir, ts)
        if args.all and all(
            os.path.exists(os.path.join(out_dir, f"{v}.png")) for v in VIEW_NAMES
        ):
            print(f"\n######## [{i + 1}/{len(timestamps)}] {ts} (skip, masks exist) ########")
            continue
        print(f"\n######## [{i + 1}/{len(timestamps)}] {ts} ########")
        process_timestamp(processor, bf16, args, cfg, ts)

    print(f"\ndone. masks -> {masks_dir}/<timestamp>/{{view}}.png")


if __name__ == "__main__":
    main()
