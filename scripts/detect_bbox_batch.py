#!/usr/bin/env python
"""Batch Qwen3-VL bbox detection for all 6 views of one or more timestamps.

Supports legacy layout (frames/{ts}/{view}.png) and normalized layout
(frames/{view}/{ts}.jpg). Normalized mode writes into masks/bbox.json.

Example:
    CUDA_VISIBLE_DEVICES=5 qwen3-vl/.venv/bin/python scripts/detect_bbox_batch.py \\
        --pipeline configs/pipeline_normalized.json --timestamp 000000 --vis
    CUDA_VISIBLE_DEVICES=5 qwen3-vl/.venv/bin/python scripts/detect_bbox_batch.py \\
        --pipeline configs/pipeline_normalized.json --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.mask_io import (
    VIEW_NAMES,
    frame_path,
    list_timestamps,
    load_bbox_json,
    save_bbox_json,
    parts_meta,
)
from common.qwen_bbox import BATCH_PROMPT, parse_boxes, visualize


def pick_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        print("flash_attn not installed, falling back to sdpa")
        return "sdpa"


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detect_one(model, processor, image_path: str, prompt: str, max_new_tokens: int):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return raw, parse_boxes(raw)


def process_timestamp(
    model,
    processor,
    args,
    cfg: dict,
    timestamp: str,
    bbox_store: dict[str, Any] | None = None,
) -> None:
    layout = cfg.get("frames_layout", "legacy")
    frames_dir = cfg["frames_dir"]
    parts_list = cfg.get("parts")
    masks_dir = cfg.get("masks_dir")
    out_root = args.out_root

    if layout == "normalized":
        if not masks_dir:
            raise ValueError("masks_dir required in pipeline config for normalized layout")
        vis_dir = os.path.join(masks_dir, "_bbox_vis", timestamp) if args.vis else None
    else:
        out_dir = os.path.join(out_root, timestamp)
        os.makedirs(out_dir, exist_ok=True)
        vis_dir = out_dir if args.vis else None

    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    summary = {"timestamp": timestamp, "views": {}}

    for vname in VIEW_NAMES:
        image_path = frame_path(frames_dir, layout, timestamp, vname)
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        print(f"\n===== {timestamp} / {vname} ({w}x{h}) =====")
        raw, parts = detect_one(model, processor, image_path, args.prompt, args.max_new_tokens)
        print("RAW:", raw)
        print("PARTS:", parts)

        if layout == "normalized":
            assert bbox_store is not None
            bbox_store["parts"] = parts_meta(parts_list)
            bbox_store["frames"].setdefault(timestamp, {})
            bbox_store["frames"][timestamp][vname] = {
                "image_path": image_path,
                "image_size": [w, h],
                "parts": parts,
            }
        else:
            record = {
                "timestamp": timestamp,
                "view": vname,
                "image_path": image_path,
                "image_size": [w, h],
                "parts": parts,
                "raw_output": raw,
            }
            json_path = os.path.join(out_dir, f"{vname}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

        if args.vis and vis_dir and parts:
            visualize(image, parts, os.path.join(vis_dir, f"{vname}_bbox.png"))
        summary["views"][vname] = {"parts": [p["label"] for p in parts], "count": len(parts)}

    if layout == "normalized":
        assert bbox_store is not None and masks_dir
        save_bbox_json(masks_dir, bbox_store)
        print(f"\nSaved {masks_dir}/bbox.json for {timestamp} ({len(bbox_store['frames'])} frames total)")
    else:
        with open(os.path.join(out_dir, "_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSummary -> {out_dir}/_summary.json")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--timestamp")
    g.add_argument("--all", action="store_true", help="process every timestamp under frames_dir")
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline.json"))
    ap.add_argument("--prompt", default=BATCH_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-pixels", type=int, default=2048 * 32 * 32)
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="reset masks/bbox.json before running (normalized --all only)")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "outputs", "bboxes"))
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    layout = cfg.get("frames_layout", "legacy")
    frames_dir = cfg["frames_dir"]
    model_path = cfg["qwen_model"]
    masks_dir = cfg.get("masks_dir")

    if args.all:
        timestamps = list_timestamps(frames_dir, layout)
        print(f"Processing {len(timestamps)} timestamps ({layout} layout)")
    else:
        timestamps = [args.timestamp]

    bbox_store: dict[str, Any] | None = None
    if layout == "normalized":
        if not masks_dir:
            raise ValueError("masks_dir required in pipeline config for normalized layout")
        os.makedirs(masks_dir, exist_ok=True)
        if args.fresh:
            bbox_store = {"parts": parts_meta(cfg.get("parts")), "frames": {}}
            save_bbox_json(masks_dir, bbox_store)
            print(f"Reset {masks_dir}/bbox.json")
        elif args.all:
            bbox_store = load_bbox_json(masks_dir)
            print(f"Loaded {masks_dir}/bbox.json ({len(bbox_store['frames'])} existing frames)")
        else:
            bbox_store = load_bbox_json(masks_dir)

    print("Loading model from", model_path)
    attn = pick_attn_implementation()
    print("attn_implementation:", attn)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.image_processor.size = {
        "longest_edge": args.max_pixels,
        "shortest_edge": 256 * 32 * 32,
    }

    for i, ts in enumerate(timestamps):
        if layout == "normalized" and bbox_store is not None and args.all:
            views = bbox_store.get("frames", {}).get(ts, {})
            if len(views) >= len(VIEW_NAMES):
                print(f"\n######## [{i + 1}/{len(timestamps)}] {ts} (skip, bbox complete) ########")
                continue
        print(f"\n######## [{i + 1}/{len(timestamps)}] {ts} ########")
        process_timestamp(model, processor, args, cfg, ts, bbox_store=bbox_store)

    if layout == "normalized" and bbox_store is not None:
        print(f"\nAll bbox data -> {masks_dir}/bbox.json ({len(bbox_store['frames'])} frames)")


if __name__ == "__main__":
    main()
