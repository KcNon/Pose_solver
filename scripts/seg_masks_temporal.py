#!/usr/bin/env python
"""Per-view temporal SAM3.1 masks seeded by Qwen boxes on one key frame.

The normalized cameras are fixed, so every view directory is a separate video
sequence.  Qwen is run only for ``--init-timestamp``; SAM3.1 receives that box
and text prompt, then tracks the selected instance forward and backward through
the sequence.  Palette masks remain compatible with backproject_normalized.py.

Example:
    # First create only the six-view boxes for frame 000000.
    QWEN_PY=/data_ft_9_10/wentai/projects/qwen3-vl/.venv/bin/python
    CUDA_VISIBLE_DEVICES=5 $QWEN_PY scripts/detect_bbox_batch.py \\
        --pipeline configs/pipeline_normalized.json --timestamp 000000

    CUDA_VISIBLE_DEVICES=4 /data_ft_9_10/wentai/projects/sam3/.venv/bin/python \\
        scripts/seg_masks_temporal.py --pipeline configs/pipeline_normalized.json \\
        --all --init-timestamp 000000 --gpu 4

The script does not run Qwen itself: Qwen and SAM use separate virtual
environments.  For a later re-anchor, run Qwen for another timestamp and call
        this script with that timestamp plus --range-start/--range-end for the segment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.mask_io import (
    DEFAULT_PARTS,
    DEFAULT_PROMPTS,
    load_bbox_json,
    masks_to_label_map,
    save_palette_png,
    view_names,
)


def load_pipeline(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_view_frames(view_dir: str) -> list[tuple[str, str]]:
    files = []
    for name in os.listdir(view_dir):
        stem, ext = os.path.splitext(name)
        if stem.isdigit() and ext.lower() in {".jpg", ".jpeg", ".png"}:
            files.append((stem, os.path.join(view_dir, name)))
    return sorted(files, key=lambda item: int(item[0]))


def box_to_xywh(box_2d: list[float]) -> list[float]:
    """Convert Qwen 0..1000 xyxy into SAM normalized xywh."""
    x1, y1, x2, y2 = (float(v) / 1000.0 for v in box_2d)
    x1, x2 = sorted((max(0.0, x1), min(1.0, x2)))
    y1, y2 = sorted((max(0.0, y1), min(1.0, y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid Qwen bbox: {box_2d}")
    return [x1, y1, x2 - x1, y2 - y1]


def largest_cc(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n <= 1:
        return mask_u8.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == keep


def bbox_iou(box_a: np.ndarray, box_b_xywh: list[float]) -> float:
    """IoU for normalized xywh boxes."""
    ax, ay, aw, ah = (float(v) for v in box_a)
    bx, by, bw, bh = box_b_xywh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def pick_seed_object(outputs: dict[str, Any], qwen_xywh: list[float]) -> int | None:
    """Pick the detector instance overlapping the Qwen box on the seed frame."""
    obj_ids = np.asarray(outputs.get("out_obj_ids", []))
    boxes = np.asarray(outputs.get("out_boxes_xywh", []))
    if len(obj_ids) == 0:
        return None
    if len(boxes) != len(obj_ids):
        return int(obj_ids[0])
    scores = [bbox_iou(box, qwen_xywh) for box in boxes]
    return int(obj_ids[int(np.argmax(scores))])


def collect_masks(
    generator,
    object_id: int,
    frame_masks: dict[int, np.ndarray],
) -> None:
    for frame_idx, output in generator:
        obj_ids = np.asarray(output.get("out_obj_ids", []))
        masks = np.asarray(output.get("out_binary_masks", []))
        match = np.flatnonzero(obj_ids == object_id)
        if len(match):
            frame_masks[int(frame_idx)] = largest_cc(masks[int(match[0])])


def track_part(
    model,
    view_dir: str,
    seed_idx: int,
    qwen_box: list[float],
    text_prompt: str,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Track exactly the seed instance in both temporal directions."""
    state = model.init_state(
        resource_path=view_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=False,
        use_cv2=True,
    )
    qwen_xywh = box_to_xywh(qwen_box)
    _, seed_output = model.add_prompt(
        inference_state=state,
        frame_idx=seed_idx,
        text_str=text_prompt,
        boxes_xywh=[qwen_xywh],
        box_labels=[1],
    )
    object_id = pick_seed_object(seed_output, qwen_xywh)
    if object_id is None:
        return {}, {"status": "seed_detector_empty", "seed_object_id": None}

    masks: dict[int, np.ndarray] = {}
    seed_ids = np.asarray(seed_output.get("out_obj_ids", []))
    seed_masks = np.asarray(seed_output.get("out_binary_masks", []))
    seed_match = np.flatnonzero(seed_ids == object_id)
    if len(seed_match):
        masks[seed_idx] = largest_cc(seed_masks[int(seed_match[0])])

    collect_masks(
        model.propagate_in_video(state, start_frame_idx=seed_idx, reverse=False),
        object_id,
        masks,
    )
    collect_masks(
        model.propagate_in_video(state, start_frame_idx=seed_idx, reverse=True),
        object_id,
        masks,
    )
    return masks, {
        "status": "ok",
        "seed_object_id": object_id,
        "seed_box_xywh": qwen_xywh,
        "tracked_frames": len(masks),
    }


def qwen_boxes_for_view(data: dict[str, Any], timestamp: str, view: str) -> dict[str, list[float]]:
    items = data.get("frames", {}).get(timestamp, {}).get(view, {}).get("parts", [])
    boxes: dict[str, list[float]] = {}
    for item in items:
        label = item.get("label")
        box = item.get("bbox_2d")
        if label and box and len(box) == 4:
            boxes[str(label)] = [float(v) for v in box]
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="write masks for every timestamp")
    group.add_argument("--timestamps", nargs="+", help="write only these timestamps")
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline_normalized.json"))
    ap.add_argument("--init-timestamp", required=True,
                    help="timestamp whose Qwen boxes initialize this temporal segment")
    ap.add_argument("--range-start", help="first timestamp to write (inclusive); tracking remains bidirectional")
    ap.add_argument("--range-end", help="last timestamp to write (inclusive); tracking remains bidirectional")
    ap.add_argument("--views", nargs="+", default=None)
    ap.add_argument(
        "--parts", nargs="+", choices=DEFAULT_PARTS,
        help="parts to track (default: pipeline temporal_parts, or lid inner_pot)",
    )
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    if cfg.get("frames_layout") != "normalized":
        raise ValueError("seg_masks_temporal.py requires frames_layout=normalized")
    frames_dir = cfg["frames_dir"]
    masks_dir = cfg["masks_dir"]
    parts = args.parts or cfg.get("temporal_parts", ["lid", "inner_pot"])
    prompts = cfg.get("prompts", DEFAULT_PROMPTS)
    bbox_data = load_bbox_json(masks_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from sam3.model_builder import build_sam3_predictor

    print("Loading SAM3.1 multiplex video model")
    predictor = build_sam3_predictor(
        version="sam3.1",
        checkpoint_path=cfg["sam_ckpt"],
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        async_loading_frames=False,
    )
    model = predictor.model

    views = args.views or view_names(cfg)
    all_view_frames = {view: list_view_frames(os.path.join(frames_dir, view)) for view in views}
    reference_timestamps = [ts for ts, _ in all_view_frames[views[0]]]
    if args.init_timestamp not in reference_timestamps:
        raise ValueError(f"init timestamp {args.init_timestamp} not found in {frames_dir}")
    wanted = set(reference_timestamps if args.all else args.timestamps)
    if args.range_start is not None:
        wanted = {ts for ts in wanted if int(ts) >= int(args.range_start)}
    if args.range_end is not None:
        wanted = {ts for ts in wanted if int(ts) <= int(args.range_end)}
    if not wanted:
        raise ValueError("the requested output range contains no video frames")
    unknown = wanted.difference(reference_timestamps)
    if unknown:
        raise ValueError(f"timestamps not found in video: {sorted(unknown)}")

    accumulated: dict[str, dict[str, dict[int, np.ndarray]]] = defaultdict(dict)
    summary: dict[str, Any] = {
        "method": "sam3.1_video_qwen_seed",
        "init_timestamp": args.init_timestamp,
        "views": {},
    }
    for view in views:
        frames = all_view_frames[view]
        timestamps = [ts for ts, _ in frames]
        if timestamps != reference_timestamps:
            raise ValueError(f"{view} has a different timestamp sequence")
        seed_idx = timestamps.index(args.init_timestamp)
        boxes = qwen_boxes_for_view(bbox_data, args.init_timestamp, view)
        view_summary: dict[str, Any] = {}
        print(f"\n===== {view}: {len(frames)} frames, seed index {seed_idx} =====")
        for part in parts:
            box = boxes.get(part)
            if box is None:
                view_summary[part] = {"status": "missing_init_bbox", "tracked_frames": 0}
                accumulated[view][part] = {}
                print(f"{view} {part}: no Qwen box at {args.init_timestamp}; writing background")
                continue
            prompt_list = prompts.get(part, DEFAULT_PROMPTS.get(part, [part.replace("_", " ")]))
            text_prompt = prompt_list[0]
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, info = track_part(model, os.path.join(frames_dir, view), seed_idx, box, text_prompt)
            accumulated[view][part] = masks
            view_summary[part] = {**info, "qwen_bbox_2d": box, "text_prompt": text_prompt}
            print(f"{view} {part}: {info}")
        summary["views"][view] = view_summary

    for frame_idx, timestamp in enumerate(reference_timestamps):
        if timestamp not in wanted:
            continue
        out_dir = os.path.join(masks_dir, timestamp)
        out_paths = [os.path.join(out_dir, f"{view}.png") for view in views]
        if args.skip_existing and all(os.path.exists(path) for path in out_paths):
            print(f"{timestamp}: skip (masks exist)")
            continue
        os.makedirs(out_dir, exist_ok=True)
        for view in views:
            per_part = accumulated[view]
            shape = None
            masks: dict[str, np.ndarray] = {}
            for part in parts:
                mask = per_part.get(part, {}).get(frame_idx)
                if mask is not None:
                    shape = mask.shape
                masks[part] = mask
            if shape is None:
                image = cv2.imread(all_view_frames[view][frame_idx][1], cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"failed to read {all_view_frames[view][frame_idx][1]}")
                shape = image.shape[:2]
            for part, mask in masks.items():
                if mask is None:
                    masks[part] = np.zeros(shape, dtype=bool)
            # When body is prompted as the whole cooker, convert it to the
            # exterior housing by removing the other two tracked components.
            if "body" in masks:
                if "lid" in masks:
                    masks["body"] &= ~masks["lid"]
                if "inner_pot" in masks:
                    masks["body"] &= ~masks["inner_pot"]
                if masks["body"].any():
                    masks["body"] = largest_cc(masks["body"])
            # Always keep the canonical palette ids: lid=1, body=2,
            # inner_pot=3, even when only a subset is tracked.
            save_palette_png(
                masks_to_label_map(masks, DEFAULT_PARTS),
                os.path.join(out_dir, f"{view}.png"),
                DEFAULT_PARTS,
            )

    summary_path = os.path.join(masks_dir, "temporal_masks.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"done. temporal masks -> {masks_dir}/<timestamp>/{{view}}.png")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
