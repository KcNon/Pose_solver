#!/usr/bin/env python
"""Detect reusable per-part seed boxes with Qwen3-VL.

Run this entry point in the Qwen environment.  It writes only bbox metadata;
SAM tracks and final palette masks are separate stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.masking.io import (
    frame_path,
    load_bbox_json,
    save_bbox_json,
    validate_synchronized_frames,
)
from common.masking.schema import load_mask_pipeline_config
from common.qwen_bbox import build_batch_prompt, parse_boxes, visualize


def _attention() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _detect(model, processor, image_path: Path, prompt: str, max_new_tokens: int):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    generated = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return raw, parse_boxes(raw)


def _request_fingerprint(
    *,
    image_path: Path,
    prompt: str,
    model_path: str,
    max_new_tokens: int,
    max_pixels: int,
) -> str:
    stat = image_path.stat()
    payload = {
        "image": str(image_path.resolve()),
        "image_size": stat.st_size,
        "image_mtime_ns": stat.st_mtime_ns,
        "prompt": prompt,
        "model_path": str(model_path),
        "max_new_tokens": int(max_new_tokens),
        "max_pixels": int(max_pixels),
    }
    return hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--pipeline", dest="config", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--timestamp")
    selection.add_argument("--timestamps", nargs="+")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--views", nargs="+")
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=2048 * 32 * 32)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--force", "--fresh", action="store_true")
    args = parser.parse_args()

    config = load_mask_pipeline_config(args.config)
    views = args.views or list(config.views)
    unknown_views = set(views).difference(config.views)
    if unknown_views:
        raise ValueError(f"unknown views: {sorted(unknown_views)}")
    selected_names = args.parts or config.part_names
    unknown_parts = set(selected_names).difference(config.part_names)
    if unknown_parts:
        raise ValueError(f"unknown parts: {sorted(unknown_parts)}")
    selected_parts = [config.part_map[name] for name in selected_names]
    frame_ids = validate_synchronized_frames(config.frames_dir, views)
    timestamps = (
        frame_ids if args.all
        else list(args.timestamps or [args.timestamp])
    )
    unknown_timestamps = set(timestamps).difference(frame_ids)
    if unknown_timestamps:
        raise ValueError(f"unknown timestamps: {sorted(unknown_timestamps)}")

    bbox_data = (
        {"parts": {}, "frames": {}}
        if args.force else load_bbox_json(config.bbox_path)
    )
    bbox_data["parts"] = {
        part.name: {
            "id": part.id,
            "color": list(part.color),
            "start_frame": part.start_frame,
        }
        for part in config.parts
    }

    model_path = config.raw["qwen_model"]
    model = None
    processor = None

    for timestamp in timestamps:
        active = [
            part for part in selected_parts
            if int(timestamp) >= part.start_frame
        ]
        if not active:
            print(f"{timestamp}: no configured part exists yet; skip", flush=True)
            continue
        prompt = args.prompt or build_batch_prompt(active)
        allowed = {part.name for part in active}
        for view in views:
            image_path = frame_path(config.frames_dir, view, timestamp)
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            request_fingerprint = _request_fingerprint(
                image_path=image_path,
                prompt=prompt,
                model_path=model_path,
                max_new_tokens=args.max_new_tokens,
                max_pixels=args.max_pixels,
            )
            existing = bbox_data.get("frames", {}).get(timestamp, {}).get(view)
            requested_before = set(existing.get("requested_parts", ())) if existing else set()
            if existing is not None and not requested_before:
                # Old bbox files were always produced with every active part.
                requested_before = set(allowed)
            if (
                existing is not None
                and allowed.issubset(requested_before)
                and existing.get("request_fingerprint") == request_fingerprint
                and not args.force
            ):
                print(f"{timestamp}/{view}: reuse bbox", flush=True)
                continue
            if model is None or processor is None:
                import torch
                from transformers import (
                    AutoModelForImageTextToText,
                    AutoProcessor,
                )

                attention = _attention()
                print(
                    f"loading Qwen from {model_path} ({attention})",
                    flush=True,
                )
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    attn_implementation=attention,
                    device_map="auto",
                )
                processor = AutoProcessor.from_pretrained(model_path)
                processor.image_processor.size = {
                    "longest_edge": args.max_pixels,
                    "shortest_edge": 256 * 32 * 32,
                }
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            raw, boxes = _detect(
                model, processor, image_path, prompt, args.max_new_tokens
            )
            boxes = [box for box in boxes if box["label"] in allowed]
            bbox_data.setdefault("frames", {}).setdefault(timestamp, {})[view] = {
                "image_path": str(image_path),
                "image_size": list(image.size),
                "parts": boxes,
                "raw_output": raw,
                "requested_parts": sorted(allowed),
                "request_fingerprint": request_fingerprint,
            }
            if args.vis:
                output = config.bbox_path.parent / "preview" / timestamp / f"{view}.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                visualize(image, boxes, output)
            print(
                f"{timestamp}/{view}: {[box['label'] for box in boxes]}",
                flush=True,
            )
        save_bbox_json(config.bbox_path, bbox_data)
    print(f"bbox seeds -> {config.bbox_path}", flush=True)


if __name__ == "__main__":
    main()
