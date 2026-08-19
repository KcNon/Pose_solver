#!/usr/bin/env python
"""Validate Qwen/SAM initialization masks with leave-one-view-out geometry.

The user's configured part start is authoritative.  A short initialization
window accommodates a component entering different camera fields of view at
slightly different times.  Qwen proposals are segmented independently, then
the calibrated views vote on each target view.  A missing or geometrically
disjoint proposal is re-prompted with the projected consensus box.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.masking.io import (
    frame_path,
    load_bbox_json,
    save_binary_mask,
    validated_seed_path,
    validate_synchronized_frames,
    write_json,
)
from common.masking.multiview import multiview_geometric_prior
from common.masking.sam import (
    build_sam31_instance_box_processor,
    predict_image_mask,
    select_qwen_box,
)
from common.masking.schema import load_mask_pipeline_config
from common.normalized_recon import load_recon


def _mask_iou(first: np.ndarray | None, second: np.ndarray) -> float:
    if first is None:
        return 0.0
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / union if union else 0.0


def _mask_pair_overlap(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    """Return pair IoU and overlap fraction of the smaller mask."""
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    smaller = min(int(a.sum()), int(b.sum()))
    return (
        intersection / union if union else 0.0,
        intersection / smaller if smaller else 0.0,
    )


def _prior_coverage(mask: np.ndarray | None, prior: np.ndarray) -> float:
    pixels = int(prior.sum())
    if mask is None or not pixels:
        return 0.0
    return float(np.logical_and(mask, prior).sum()) / pixels


def _box_from_mask(mask: np.ndarray, padding: float) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    height, width = mask.shape
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    dx = max(2.0, (x2 - x1) * padding)
    dy = max(2.0, (y2 - y1) * padding)
    return [
        max(0.0, x1 - dx) / width * 1000.0,
        max(0.0, y1 - dy) / height * 1000.0,
        min(float(width), x2 + dx) / width * 1000.0,
        min(float(height), y2 + dy) / height * 1000.0,
    ]


def _box_iou(first: list[float], second: list[float]) -> float:
    """Return IoU for two normalized xyxy boxes."""
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _ambiguous_duplicate_parts(
    bbox_data: dict[str, Any],
    timestamp: str,
    view: str,
    parts: list[str],
    *,
    overrides: dict[str, Any] | None,
    iou_threshold: float,
) -> set[str]:
    """Find different labels that Qwen assigned essentially the same box.

    Such a box cannot be a trustworthy instance anchor for both physical
    components.  Rejecting both labels lets adjacent temporal anchors bridge
    the frame without guessing which label Qwen intended.
    """
    boxes: dict[str, list[float]] = {}
    for part_name in parts:
        try:
            box, _ = select_qwen_box(
                bbox_data,
                timestamp,
                view,
                part_name,
                overrides=overrides,
            )
        except RuntimeError:
            continue
        boxes[part_name] = box

    ambiguous: set[str] = set()
    names = list(boxes)
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            if _box_iou(boxes[first_name], boxes[second_name]) >= iou_threshold:
                ambiguous.update((first_name, second_name))
    return ambiguous


def _active_parts(config, timestamp: str) -> list[str]:
    length = int(config.raw.get("qwen_seed_window", {}).get("length", 0))
    frame = int(timestamp)
    if config.raw.get("qwen_reanchor_on_part_events", False):
        return [
            part.name for part in config.parts
            if part.start_frame <= frame
        ]
    return [
        part.name
        for part in config.parts
        if part.start_frame <= frame <= part.start_frame + length
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--timestamps", nargs="+", required=True)
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--views", nargs="+")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--minimum-pixels", type=int, default=500)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = load_mask_pipeline_config(args.config)
    views = list(args.views or config.views)
    selected_parts = set(args.parts or config.part_names)
    frame_ids = validate_synchronized_frames(config.frames_dir, views)
    unknown = set(args.timestamps).difference(frame_ids)
    if unknown:
        raise ValueError(f"unknown timestamps: {sorted(unknown)}")
    bbox_data = load_bbox_json(config.bbox_path)
    settings = config.raw.get("multiview_seed_validation", {})
    minimum_source_views = int(settings.get("minimum_source_views", 2))
    minimum_source_pixels = int(settings.get("minimum_source_pixels", 40))
    minimum_prior_pixels = int(settings.get("minimum_prior_pixels", 40))
    minimum_coverage = float(settings.get("minimum_prior_coverage", 0.20))
    repair_coverage = float(settings.get("repair_prior_coverage", 0.45))
    improvement = float(settings.get("minimum_coverage_improvement", 0.15))
    repair_disjoint_existing = bool(
        settings.get("repair_disjoint_existing", False)
    )
    repair_missing_seeds = bool(settings.get("repair_missing_seeds", True))
    depth_tolerance = float(settings.get("depth_tolerance", 0.03))
    bbox_padding = float(settings.get("bbox_padding", 0.18))
    reject_duplicate_boxes = bool(
        settings.get("reject_cross_part_duplicate_boxes", True)
    )
    duplicate_box_iou = float(settings.get("duplicate_bbox_iou", 0.95))
    maximum_cross_part_iou = float(
        settings.get("maximum_cross_part_mask_iou", 1.0)
    )
    maximum_cross_part_containment = float(
        settings.get("maximum_cross_part_mask_containment", 1.0)
    )
    maximum_mask_area_fraction = float(
        settings.get("maximum_mask_area_fraction", 1.0)
    )
    minimum_existing_prior_iou = float(
        settings.get("minimum_existing_prior_iou", 0.0)
    )
    prompt_mode = str(
        config.raw.get("sam_image_prompt_mode", "grounded_text_box")
    )

    import cv2
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if prompt_mode == "instance_box":
        processor = build_sam31_instance_box_processor(
            config.raw["sam_ckpt"]
        )
    else:
        model = build_sam3_image_model(
            checkpoint_path=config.raw["sam_ckpt"],
            load_from_HF=False,
            device="cuda",
            eval_mode=True,
        )
        processor = Sam3Processor(model, confidence_threshold=args.confidence)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    report: dict[str, Any] = {"frames": {}, "summary": {}}
    counts = {
        "qwen_segmented": 0,
        "geometry_repaired": 0,
        "geometry_confirmed": 0,
        "ambiguous_duplicate_bbox_rejected": 0,
        "cross_part_mask_rejected": 0,
        "oversized_mask_rejected": 0,
        "geometry_disjoint_mask_rejected": 0,
        "unresolved": 0,
    }

    for timestamp in args.timestamps:
        active = [
            name for name in _active_parts(config, timestamp)
            if name in selected_parts
        ]
        if not active:
            continue
        recon = load_recon(config.raw, timestamp)
        view_indices = [config.views.index(view) for view in views]
        depth = recon["depth"][view_indices]
        intrinsics = recon["intrinsics"][view_indices]
        extrinsics = recon["extrinsics"][view_indices]
        small_height, small_width = recon["depth_hw"]
        images = [
            Image.open(frame_path(config.frames_dir, view, timestamp)).convert("RGB")
            for view in views
        ]
        ambiguous_by_view = {
            view: (
                _ambiguous_duplicate_parts(
                    bbox_data,
                    timestamp,
                    view,
                    active,
                    overrides=config.raw.get("bbox_overrides"),
                    iou_threshold=duplicate_box_iou,
                )
                if reject_duplicate_boxes
                else set()
            )
            for view in views
        }
        timestamp_report: dict[str, Any] = {}
        selected_by_part: dict[str, list[np.ndarray | None]] = {}
        for part_name in active:
            part = config.part_map[part_name]
            masks: list[np.ndarray | None] = []
            details: list[dict[str, Any]] = []
            for view, image in zip(views, images):
                if part_name in ambiguous_by_view[view]:
                    masks.append(None)
                    details.append(
                        {
                            "status": "ambiguous_duplicate_bbox",
                            "duplicate_bbox_iou_threshold": duplicate_box_iou,
                        }
                    )
                    counts["ambiguous_duplicate_bbox_rejected"] += 1
                    continue
                try:
                    box, source = select_qwen_box(
                        bbox_data,
                        timestamp,
                        view,
                        part_name,
                        overrides=config.raw.get("bbox_overrides"),
                    )
                except RuntimeError:
                    masks.append(None)
                    details.append({"status": "qwen_bbox_missing"})
                    continue
                with autocast:
                    state = processor.set_image(image)
                mask, info = predict_image_mask(
                    processor,
                    state,
                    part.prompts,
                    box,
                    confidence=args.confidence,
                    autocast=autocast,
                    minimum_pixels=args.minimum_pixels,
                    prompt_mode=prompt_mode,
                )
                masks.append(mask)
                details.append({**info, "bbox_source": source})
                if mask is not None:
                    counts["qwen_segmented"] += 1

            small_masks = [
                (
                    cv2.resize(
                        mask.astype(np.uint8),
                        (small_width, small_height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    if mask is not None
                    else np.zeros((small_height, small_width), dtype=bool)
                )
                for mask in masks
            ]
            reliable = [
                int(mask.sum()) >= minimum_source_pixels for mask in small_masks
            ]
            part_report = {}
            for target_index, (view, image) in enumerate(zip(views, images)):
                prior_small, geometry = multiview_geometric_prior(
                    small_masks,
                    reliable,
                    depth,
                    intrinsics,
                    extrinsics,
                    target_index,
                    minimum_source_views=minimum_source_views,
                    depth_tolerance=depth_tolerance,
                    minimum_pixels=minimum_source_pixels,
                )
                prior = cv2.resize(
                    prior_small.astype(np.uint8),
                    image.size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                current = masks[target_index]
                current_coverage = _prior_coverage(current, prior)
                selected = current
                action = "confirmed"
                candidate_info: dict[str, Any] | None = None
                prior_pixels = int(prior_small.sum())
                needs_repair = (
                    repair_missing_seeds
                    and prior_pixels >= minimum_prior_pixels
                    and (
                        current is None
                        or (
                            repair_disjoint_existing
                            and current_coverage < minimum_coverage
                        )
                    )
                )
                if needs_repair:
                    geometry_box = _box_from_mask(prior, bbox_padding)
                    if geometry_box is not None:
                        with autocast:
                            state = processor.set_image(image)
                        candidate, candidate_info = predict_image_mask(
                            processor,
                            state,
                            part.prompts,
                            geometry_box,
                            confidence=args.confidence,
                            autocast=autocast,
                            minimum_pixels=args.minimum_pixels,
                            prompt_mode=prompt_mode,
                        )
                        candidate_coverage = _prior_coverage(candidate, prior)
                        if (
                            candidate is not None
                            and candidate_coverage >= repair_coverage
                            and candidate_coverage
                            >= current_coverage + improvement
                        ):
                            selected = candidate
                            action = "geometry_repaired"
                            counts["geometry_repaired"] += 1
                        else:
                            action = "repair_rejected"
                    else:
                        action = "prior_bbox_empty"
                elif current is not None:
                    counts["geometry_confirmed"] += 1

                selected_iou = _mask_iou(selected, prior)
                selected_area_fraction = (
                    float(selected.sum()) / (selected.size)
                    if selected is not None else 0.0
                )
                if (
                    selected is not None
                    and selected_area_fraction > maximum_mask_area_fraction
                ):
                    selected = None
                    action = "oversized_mask_rejected"
                    counts["oversized_mask_rejected"] += 1
                elif (
                    selected is not None
                    and prior_pixels >= minimum_prior_pixels
                    and selected_iou < minimum_existing_prior_iou
                ):
                    selected = None
                    action = "geometry_disjoint_mask_rejected"
                    counts["geometry_disjoint_mask_rejected"] += 1

                if selected is not None:
                    save_binary_mask(
                        validated_seed_path(
                            config.work_root, part_name, timestamp, view
                        ),
                        selected,
                    )
                else:
                    # A force/retry run must not silently reuse a seed that a
                    # previous policy accepted.  Missing now means missing.
                    validated_seed_path(
                        config.work_root, part_name, timestamp, view
                    ).unlink(missing_ok=True)
                    counts["unresolved"] += 1
                part_report[view] = {
                    **details[target_index],
                    "action": action,
                    "prior_pixels": prior_pixels,
                    "prior_coverage_before": current_coverage,
                    "iou_before": _mask_iou(current, prior),
                    "selected_iou": selected_iou,
                    "selected_area_fraction": selected_area_fraction,
                    "geometry": geometry,
                    "candidate": candidate_info,
                    "output_pixels": int(selected.sum()) if selected is not None else 0,
                }
                masks[target_index] = selected
            timestamp_report[part_name] = part_report
            selected_by_part[part_name] = masks

        # A part-wise validator must also verify that two physical links were
        # not assigned the same semantic whole-object mask.  Reject both sides
        # of an ambiguous pair; a neighboring temporal anchor is safer than
        # guessing which label owns the pixels.
        for view_index, view in enumerate(views):
            conflicts: dict[str, list[dict[str, Any]]] = {}
            for first_index, first_name in enumerate(active):
                first = selected_by_part.get(first_name, [None] * len(views))[view_index]
                if first is None:
                    continue
                for second_name in active[first_index + 1 :]:
                    second = selected_by_part.get(
                        second_name, [None] * len(views)
                    )[view_index]
                    if second is None:
                        continue
                    pair_iou, containment = _mask_pair_overlap(first, second)
                    if (
                        pair_iou <= maximum_cross_part_iou
                        and containment <= maximum_cross_part_containment
                    ):
                        continue
                    detail = {
                        "other_part": second_name,
                        "mask_iou": pair_iou,
                        "smaller_mask_containment": containment,
                    }
                    conflicts.setdefault(first_name, []).append(detail)
                    conflicts.setdefault(second_name, []).append(
                        {**detail, "other_part": first_name}
                    )
            for part_name, conflict in conflicts.items():
                path = validated_seed_path(
                    config.work_root, part_name, timestamp, view
                )
                path.unlink(missing_ok=True)
                selected_by_part[part_name][view_index] = None
                timestamp_report[part_name][view]["action"] = (
                    "cross_part_mask_rejected"
                )
                timestamp_report[part_name][view]["cross_part_conflicts"] = conflict
                timestamp_report[part_name][view]["output_pixels"] = 0
                counts["cross_part_mask_rejected"] += 1
                counts["unresolved"] += 1
        for image in images:
            image.close()
        report["frames"][timestamp] = timestamp_report
        print(f"{timestamp}: validated {active}", flush=True)

    report["summary"] = counts
    output = config.work_root / "validated_seeds" / "report.json"
    write_json(output, report)
    print(f"validated seeds -> {output}: {counts}", flush=True)


if __name__ == "__main__":
    main()
