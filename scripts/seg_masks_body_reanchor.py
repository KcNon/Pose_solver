#!/usr/bin/env python
"""Segment the fixed cooker body in every frame from one Qwen seed box.

Unlike video propagation, each frame is independently refreshed by SAM3 image
segmentation.  The camera and cooker body are fixed, so the Qwen box from the
seed frame remains a stable geometric prompt through hand, lid, and inner-pot
occlusions.  Output uses the canonical palette id ``body=2``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.mask_io import DEFAULT_PARTS, DEFAULT_PROMPTS, masks_to_label_map, save_palette_png


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def list_frames(view_dir: Path) -> list[tuple[str, Path]]:
    return sorted(
        (
            (path.stem, path)
            for path in view_dir.iterdir()
            if path.stem.isdigit() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=lambda item: int(item[0]),
    )


def seed_body_box(data: dict[str, Any], timestamp: str, view: str) -> list[float]:
    items = data.get("frames", {}).get(timestamp, {}).get(view, {}).get("parts", [])
    for item in items:
        if item.get("label") == "body" and len(item.get("bbox_2d", [])) == 4:
            return [float(value) for value in item["bbox_2d"]]
    raise RuntimeError(f"missing body bbox for {timestamp}/{view}")


def bbox_to_cxcywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((np.clip(x1 / 1000.0, 0, 1), np.clip(x2 / 1000.0, 0, 1)))
    y1, y2 = sorted((np.clip(y1 / 1000.0, 0, 1), np.clip(y2 / 1000.0, 0, 1)))
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def largest_component(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, 8)
    if count <= 1:
        return source.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == keep


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = largest_component(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    padded = np.pad(closed, 1)
    flooded = padded.copy()
    workspace = np.zeros((flooded.shape[0] + 2, flooded.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, workspace, (0, 0), 1)
    holes = flooded == 0
    return (padded.astype(bool) | holes)[1:-1, 1:-1]


def predict_body(processor, state: dict[str, Any], prompts: list[str], box: list[float],
                 confidence: float, bf16) -> tuple[np.ndarray | None, dict[str, Any]]:
    processor.set_confidence_threshold(confidence)
    geometry = bbox_to_cxcywh(box)
    best_mask, best_score, best_prompt = None, -1.0, ""
    for prompt in prompts:
        with bf16:
            work = dict(state)
            processor.reset_all_prompts(work)
            processor.set_text_prompt(prompt=prompt, state=work)
            output = processor.add_geometric_prompt(box=geometry, label=True, state=work)
        if output["scores"].numel() == 0:
            continue
        scores = output["scores"].float().cpu().numpy()
        masks = output["masks"].squeeze(1).cpu().numpy().astype(bool)
        index = int(np.argmax(scores))
        if float(scores[index]) > best_score:
            best_mask = clean_mask(masks[index])
            best_score = float(scores[index])
            best_prompt = prompt
    if best_mask is None or best_mask.sum() < 500:
        return None, {"status": "empty"}
    return best_mask, {
        "status": "ok",
        "score": best_score,
        "prompt": best_prompt,
        "pixels": int(best_mask.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--init-timestamp", default="000000")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = load_json(args.pipeline)
    frames = list_frames(Path(config["frames_dir"]) / args.view)
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    frame_ids = [timestamp for timestamp, _ in frames]
    if not frame_ids or args.init_timestamp not in frame_ids:
        raise ValueError(f"seed timestamp {args.init_timestamp} not found for {args.view}")

    output_dir = Path(config["masks_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox_path = Path(config.get("bbox_json", output_dir / "bbox.json"))
    bbox_data = load_json(bbox_path)
    body_box = seed_body_box(bbox_data, args.init_timestamp, args.view)

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    print(f"loading SAM3 image model for {args.view}", flush=True)
    model = build_sam3_image_model(
        checkpoint_path=config["sam_ckpt"], load_from_HF=False, device="cuda", eval_mode=True
    )
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    prompts = config.get("prompts", {}).get("body", DEFAULT_PROMPTS["body"])
    bf16 = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    previous: np.ndarray | None = None
    details: dict[str, Any] = {}
    for timestamp, image_path in frames:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with bf16:
            state = processor.set_image(Image.fromarray(rgb))
        mask, info = predict_body(
            processor, state, prompts, body_box, args.confidence, bf16
        )
        if mask is None:
            if previous is None:
                mask = np.zeros(rgb.shape[:2], dtype=bool)
            else:
                mask = previous.copy()
                info = {"status": "held_previous", "pixels": int(mask.sum())}
        else:
            previous = mask
        save_palette_png(
            masks_to_label_map({"body": mask}, DEFAULT_PARTS),
            str(output_dir / timestamp / f"{args.view}.png"),
            DEFAULT_PARTS,
        )
        details[timestamp] = info
        print(f"{timestamp}: body={int(mask.sum())}", flush=True)

    summary = {
        "method": "sam3_image_fixed_qwen_body_reanchor",
        "view": args.view,
        "init_timestamp": args.init_timestamp,
        "qwen_frames": [args.init_timestamp],
        "frames_written": len(frames),
        "details": details,
    }
    with open(output_dir / f"body_masks_{args.view}.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"done -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
