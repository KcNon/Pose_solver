#!/usr/bin/env python
"""Generate independent binary part tracks with SAM3.

Modes:
  video        mesh-validated Qwen seeds followed by segmented propagation
  image        every requested frame uses its own Qwen box
  fixed-image  one Qwen seed box is reused for every requested frame

Run this entry point in the SAM environment.  It never writes final palette
masks, so rerunning one part cannot erase another part.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.masking.io import (
    frame_path,
    load_bbox_json,
    load_binary_mask,
    load_label_mask,
    save_binary_mask,
    save_label_mask,
    track_path,
    validated_seed_path,
    validate_synchronized_frames,
    write_json,
)
from common.masking.sam import (
    build_sam31_instance_box_processor,
    build_sam31_instance_tracker,
    predict_image_mask,
    select_qwen_box,
    track_video_part,
    track_video_part_from_mask,
)
from common.masking.schema import MaskPipelineConfig, load_mask_pipeline_config


def _visibility_reference(
    config: MaskPipelineConfig,
    view: str,
    timestamps: list[str],
) -> tuple[set[str], dict[str, Any] | None]:
    """Select frames where a whole-object reference says the object is visible."""

    settings = config.raw.get("mask_quality", {})
    root_value = settings.get("visibility_reference_masks")
    if root_value is None:
        return set(timestamps), None
    root = Path(root_value)
    counts: dict[str, int] = {}
    missing: list[str] = []
    for timestamp in timestamps:
        path = root / timestamp / f"{view}.png"
        if not path.exists():
            missing.append(str(path))
            continue
        counts[timestamp] = int((load_label_mask(path) > 0).sum())
    if missing:
        raise FileNotFoundError(
            "visibility reference masks are incomplete; first missing path: "
            f"{missing[0]}"
        )
    positive = [value for value in counts.values() if value > 0]
    median_pixels = float(np.median(positive)) if positive else 0.0
    minimum_pixels = int(settings.get("visibility_reference_minimum_pixels", 1))
    minimum_fraction = float(
        settings.get("visibility_reference_minimum_median_fraction", 0.0)
    )
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError(
            "visibility_reference_minimum_median_fraction must lie in [0, 1]"
        )
    threshold = max(minimum_pixels, int(math.ceil(median_pixels * minimum_fraction)))
    eligible = {
        timestamp for timestamp, pixels in counts.items()
        if pixels >= threshold
    }
    if not eligible:
        raise RuntimeError(
            f"visibility reference rejected every frame for {view}; "
            f"threshold={threshold} pixels"
        )
    return eligible, {
        "root": str(root),
        "median_positive_pixels": median_pixels,
        "minimum_pixels": minimum_pixels,
        "minimum_median_fraction": minimum_fraction,
        "threshold_pixels": threshold,
        "eligible_frames": len(eligible),
        "excluded_frames": len(timestamps) - len(eligible),
    }


def _seed_for(
    config: MaskPipelineConfig,
    part: str,
    view: str,
    cli_seed: str | None,
) -> str:
    if cli_seed is not None:
        return str(cli_seed)
    values = None
    configured_parts = config.raw.get("parts", {})
    if isinstance(configured_parts, dict):
        part_config = configured_parts.get(part, {})
        tracking = part_config.get("tracking", part_config.get("tracker", {}))
        if isinstance(tracking, dict):
            values = tracking.get("seed_frames", tracking.get("seed_frame"))
    if values is None:
        values = config.raw.get("seed_frames", {}).get(part)
    if values is None:
        values = config.raw.get("temporal_seed_timestamps", {}).get(part)
    if isinstance(values, dict):
        values = values.get(view, values.get("default"))
    if isinstance(values, list):
        values = values[0] if values else None
    if values is None:
        values = config.part_map[part].start_frame
    return f"{int(values):06d}" if str(values).isdigit() else str(values)


def _fallback_labels(config: MaskPipelineConfig, part: str) -> list[str]:
    return [
        str(value)
        for value in config.raw.get("bbox_fallback_labels", {}).get(part, [])
    ]


def _bbox_crop_sharpness(
    config: MaskPipelineConfig,
    timestamp: str,
    view: str,
    box: list[float],
) -> float:
    """Estimate local image sharpness inside a normalized xyxy bbox."""
    from PIL import Image, UnidentifiedImageError

    path = frame_path(config.frames_dir, view, timestamp)
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            width, height = gray.size
            x1 = max(0, min(width - 1, round(box[0] / 1000.0 * width)))
            y1 = max(0, min(height - 1, round(box[1] / 1000.0 * height)))
            x2 = max(x1 + 1, min(width, round(box[2] / 1000.0 * width)))
            y2 = max(y1 + 1, min(height, round(box[3] / 1000.0 * height)))
            crop = gray.crop((x1, y1, x2, y2))
            crop.thumbnail((256, 256))
            values = np.asarray(crop, dtype=np.float32)
    except (OSError, UnidentifiedImageError):
        return 0.0
    if values.shape[0] < 2 or values.shape[1] < 2:
        return 0.0
    horizontal = np.diff(values, axis=1)
    vertical = np.diff(values, axis=0)
    return float(np.mean(horizontal * horizontal) + np.mean(vertical * vertical))


def _validated_seed(
    config: MaskPipelineConfig,
    bbox_data: dict[str, Any],
    frame_ids: list[str],
    part: str,
    view: str,
    configured_seed: str,
) -> tuple[str, str]:
    """Use a mesh-validated seed, falling back to the closest valid record."""

    mesh_dir = Path(
        config.raw.get("mesh_dir", config.frames_dir.parent / "meshes")
    )
    require_mesh_assignment = bool(
        config.raw.get(
            "require_mesh_assignment",
            (mesh_dir / f"{part}.glb").exists(),
        )
    )

    def valid(timestamp: str) -> bool:
        if config.raw.get("validated_seeds_required", False):
            return validated_seed_path(
                config.work_root, part, timestamp, view
            ).exists()
        record = (
            bbox_data.get("frames", {}).get(timestamp, {}).get(view, {})
        )
        has_part = any(
            row.get("label") == part for row in record.get("parts", [])
        )
        if not has_part:
            return False
        if not require_mesh_assignment:
            return True
        return record.get("mesh_assignment", {}).get("status") == "ok"

    if configured_seed in frame_ids and valid(configured_seed):
        return configured_seed, "configured"
    candidates = [timestamp for timestamp in frame_ids if valid(timestamp)]
    if not candidates:
        raise RuntimeError(
            f"no mesh-validated Qwen seed for {part}/{view}; "
            f"configured seed was {configured_seed}"
        )
    target = int(configured_seed) if configured_seed.isdigit() else len(frame_ids) // 2
    selected = min(candidates, key=lambda value: abs(int(value) - target))
    return selected, "nearest_mesh_validated"


def _trusted_seed_candidates(
    config: MaskPipelineConfig,
    bbox_data: dict[str, Any],
    frame_ids: list[str],
    timestamps: list[str],
    part: str,
    view: str,
    configured_seed: str,
    support_views: list[str] | None = None,
) -> list[str]:
    """Return all trustworthy Qwen anchors in the requested time span."""

    mesh_dir = Path(
        config.raw.get("mesh_dir", config.frames_dir.parent / "meshes")
    )
    require_mesh_assignment = bool(
        config.raw.get(
            "require_mesh_assignment",
            (mesh_dir / f"{part}.glb").exists(),
        )
    )
    requested_indices = [frame_ids.index(value) for value in timestamps]
    first, last = min(requested_indices), max(requested_indices)
    seed_window = config.raw.get("qwen_seed_window", {})
    seed_window_length = int(seed_window.get("length", 0))
    reanchor_on_part_events = bool(
        config.raw.get("qwen_reanchor_on_part_events", False)
    )
    configured_seed_number = (
        int(configured_seed) if configured_seed.isdigit() else None
    )
    def valid(timestamp: str, candidate_view: str) -> bool:
        seed_mask = validated_seed_path(
            config.work_root, part, timestamp, candidate_view
        )
        record = (
            bbox_data.get("frames", {}).get(timestamp, {}).get(candidate_view, {})
        )
        if config.raw.get("validated_seeds_required", False):
            if not seed_mask.exists():
                return False
        elif not any(
            row.get("label") == part for row in record.get("parts", [])
        ):
            return False
        return not (
            require_mesh_assignment
            and record.get("mesh_assignment", {}).get("status") != "ok"
        )

    candidates: list[str] = []
    for timestamp in frame_ids[first:last + 1]:
        if (
            seed_window_length > 0
            and not reanchor_on_part_events
            and configured_seed_number is not None
            and not (
                configured_seed_number
                <= int(timestamp)
                <= configured_seed_number + seed_window_length
            )
        ):
            continue
        if not valid(timestamp, view):
            continue
        candidates.append(timestamp)

    if candidates:
        selection = config.raw.get("qwen_initial_anchor_selection", {})
        if selection.get("enabled", False) and seed_window_length > 0:
            start = config.part_map[part].start_frame
            end = start + seed_window_length
            minimum_delay = max(0, int(selection.get("minimum_delay", 0)))
            initial = [
                timestamp for timestamp in candidates
                if start + minimum_delay <= int(timestamp) <= end
            ]
            later = [timestamp for timestamp in candidates if int(timestamp) > end]
            if initial:
                scored = []
                selected_views = list(support_views or config.views)
                minimum_views = min(
                    len(selected_views),
                    max(
                        int(selection.get("minimum_views", 2)),
                        math.ceil(
                            float(selection.get("minimum_view_fraction", 0.5))
                            * len(selected_views)
                        ),
                    ),
                )
                border_margin = float(
                    selection.get("minimum_border_margin", 3.0)
                )
                reject_clipped = bool(
                    selection.get("reject_border_clipped", True)
                )
                for timestamp in initial:
                    record = (
                        bbox_data.get("frames", {})
                        .get(timestamp, {})
                        .get(view, {})
                    )
                    boxes = [
                        [float(value) for value in row.get("bbox_2d", [])]
                        for row in record.get("parts", [])
                        if row.get("label") == part
                        and len(row.get("bbox_2d", [])) == 4
                    ]
                    if not boxes:
                        continue
                    box = max(
                        boxes,
                        key=lambda value: max(0.0, value[2] - value[0])
                        * max(0.0, value[3] - value[1]),
                    )
                    margin = min(
                        box[0], box[1], 1000.0 - box[2], 1000.0 - box[3]
                    )
                    support = sum(
                        valid(timestamp, candidate_view)
                        for candidate_view in selected_views
                    )
                    if support < minimum_views:
                        continue
                    if reject_clipped and margin <= border_margin:
                        continue
                    area = max(0.0, box[2] - box[0]) * max(
                        0.0, box[3] - box[1]
                    )
                    sharpness = _bbox_crop_sharpness(
                        config, timestamp, view, box
                    )
                    scored.append((
                        support,
                        sharpness,
                        margin,
                        area,
                        -int(timestamp),
                        timestamp,
                    ))
                if scored:
                    initial = [max(scored)[-1]]
                else:
                    initial = []
            candidates = sorted(initial + later, key=int)
            if not candidates:
                raise RuntimeError(
                    f"no reliable initialization anchor for {part}/{view}; "
                    f"searched {start + minimum_delay:06d}..{end:06d}"
                )
        return candidates
    seed, _ = _validated_seed(
        config,
        bbox_data,
        frame_ids,
        part,
        view,
        configured_seed,
    )
    return [seed]


def _anchor_segments(
    frame_ids: list[str],
    timestamps: list[str],
    anchors: list[str],
) -> list[dict[str, Any]]:
    """Assign requested frames to their nearest temporal Qwen anchor."""

    frame_position = {value: index for index, value in enumerate(frame_ids)}
    ordered_anchors = sorted(set(anchors), key=frame_position.__getitem__)
    groups: dict[str, list[str]] = {anchor: [] for anchor in ordered_anchors}
    for timestamp in timestamps:
        nearest = min(
            ordered_anchors,
            key=lambda anchor: (
                abs(frame_position[anchor] - frame_position[timestamp]),
                frame_position[anchor],
            ),
        )
        groups[nearest].append(timestamp)

    segments: list[dict[str, Any]] = []
    for anchor in ordered_anchors:
        requested = groups[anchor]
        if not requested:
            continue
        positions = [frame_position[anchor]] + [
            frame_position[value] for value in requested
        ]
        first, last = min(positions), max(positions)
        segments.append({
            "anchor": anchor,
            "requested": requested,
            "window_ids": frame_ids[first:last + 1],
            "window_offset": first,
        })
    return segments


def _tracking_window(
    frame_ids: list[str],
    timestamps: list[str],
    seed: str,
) -> tuple[list[str], int]:
    """Return the smallest contiguous video containing outputs and seed."""

    requested = [frame_ids.index(value) for value in timestamps]
    seed_index = frame_ids.index(seed)
    first = min(seed_index, min(requested))
    last = max(seed_index, max(requested))
    return frame_ids[first:last + 1], first


def _requested_frames(args, frame_ids: list[str], bbox_data: dict[str, Any]) -> list[str]:
    if args.timestamps:
        selected = list(args.timestamps)
    elif args.all:
        selected = list(frame_ids)
    elif args.mode == "image":
        selected = sorted(bbox_data.get("frames", {}), key=int)
    else:
        selected = list(frame_ids)
    if args.range_start is not None:
        selected = [value for value in selected if int(value) >= int(args.range_start)]
    if args.range_end is not None:
        selected = [value for value in selected if int(value) <= int(args.range_end)]
    unknown = set(selected).difference(frame_ids)
    if unknown:
        raise ValueError(f"unknown timestamps: {sorted(unknown)}")
    return selected[:args.max_frames] if args.max_frames is not None else selected


def _save_track(
    args,
    config: MaskPipelineConfig,
    timestamp: str,
    view: str,
    mask: np.ndarray,
) -> None:
    save_binary_mask(
        track_path(config.tracks_root, args.part, timestamp, view),
        mask,
    )
    if args.legacy_palette_output:
        legacy_root = config.raw.get("masks_dir")
        if not legacy_root:
            raise ValueError("--legacy-palette-output requires config masks_dir")
        label = np.zeros(mask.shape, dtype=np.uint8)
        label[np.asarray(mask, dtype=bool)] = config.part_map[args.part].id
        save_label_mask(
            Path(legacy_root) / timestamp / f"{view}.png",
            label,
            config.parts,
        )


def _image_mode(
    args,
    config: MaskPipelineConfig,
    bbox_data: dict[str, Any],
    views: list[str],
    timestamps: list[str],
) -> dict[str, Any]:
    import cv2
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    prompt_mode = str(
        config.raw.get("sam_image_prompt_mode", "grounded_text_box")
    )
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
    part = config.part_map[args.part]
    details: dict[str, Any] = {}
    for view in views:
        seed = _seed_for(config, args.part, view, args.seed_frame)
        previous: np.ndarray | None = None
        view_details = {}
        for timestamp in timestamps:
            output_path = track_path(
                config.tracks_root, args.part, timestamp, view
            )
            if args.skip_existing and output_path.exists():
                view_details[timestamp] = {"status": "reused"}
                continue
            box_timestamp = timestamp if args.mode == "image" else seed
            box, source = select_qwen_box(
                bbox_data,
                box_timestamp,
                view,
                args.part,
                overrides=config.raw.get("bbox_overrides"),
                fallback_labels=_fallback_labels(config, args.part),
            )
            image_path = frame_path(config.frames_dir, view, timestamp)
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"failed to read {image_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with autocast:
                state = processor.set_image(Image.fromarray(rgb))
            mask, info = predict_image_mask(
                processor,
                state,
                part.prompts,
                box,
                confidence=args.confidence,
                autocast=autocast,
                minimum_pixels=args.minimum_pixels,
                prompt_mode=str(
                    config.raw.get(
                        "sam_image_prompt_mode", "grounded_text_box"
                    )
                ),
            )
            if mask is None:
                if args.hold_previous and previous is not None:
                    mask = previous.copy()
                    info = {"status": "held_previous", "pixels": int(mask.sum())}
                else:
                    mask = np.zeros(rgb.shape[:2], dtype=bool)
            else:
                previous = mask
            _save_track(args, config, timestamp, view, mask)
            view_details[timestamp] = {**info, "bbox_source": source}
            print(f"{view}/{timestamp}: {args.part}={int(mask.sum())}", flush=True)
        details[view] = view_details
    return details


def _video_mode(
    args,
    config: MaskPipelineConfig,
    bbox_data: dict[str, Any],
    views: list[str],
    frame_ids: list[str],
    timestamps: list[str],
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    tracker = build_sam31_instance_tracker(config.raw["sam_ckpt"])
    prompt_mode = str(
        config.raw.get("sam_image_prompt_mode", "grounded_text_box")
    )
    if prompt_mode == "instance_box":
        processor = build_sam31_instance_box_processor(
            config.raw["sam_ckpt"], tracker=tracker
        )
    else:
        image_model = build_sam3_image_model(
            checkpoint_path=config.raw["sam_ckpt"],
            load_from_HF=False,
            device="cuda",
            eval_mode=True,
        )
        processor = Sam3Processor(
            image_model, confidence_threshold=args.confidence
        )
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    part = config.part_map[args.part]
    details: dict[str, Any] = {}
    for view in views:
        configured_seed = _seed_for(config, args.part, view, args.seed_frame)
        anchors = _trusted_seed_candidates(
            config,
            bbox_data,
            frame_ids,
            timestamps,
            args.part,
            view,
            configured_seed,
            views,
        )
        segments = _anchor_segments(frame_ids, timestamps, anchors)
        masks: dict[int, np.ndarray] = {}
        segment_details: list[dict[str, Any]] = []
        for segment in segments:
            seed = segment["anchor"]
            cached_seed = validated_seed_path(
                config.work_root, args.part, seed, view
            )
            if cached_seed.exists():
                seed_mask = load_binary_mask(cached_seed)
                seed_info = {
                    "status": "validated_multiview_seed",
                    "pixels": int(seed_mask.sum()),
                }
                box = None
                source = "validated_multiview_seed"
            else:
                box, source = select_qwen_box(
                    bbox_data,
                    seed,
                    view,
                    args.part,
                    overrides=config.raw.get("bbox_overrides"),
                    fallback_labels=_fallback_labels(config, args.part),
                )
                image_path = frame_path(config.frames_dir, view, seed)
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    with autocast:
                        image_state = processor.set_image(rgb)
                    seed_mask, seed_info = predict_image_mask(
                        processor,
                        image_state,
                        part.prompts,
                        box,
                        confidence=args.confidence,
                        autocast=autocast,
                        minimum_pixels=args.minimum_pixels,
                        prompt_mode=str(
                            config.raw.get(
                                "sam_image_prompt_mode", "grounded_text_box"
                            )
                        ),
                    )
            if seed_mask is None:
                raise RuntimeError(
                    f"SAM seed mask is empty for {args.part}/{view}/{seed}: "
                    f"{seed_info}"
                )

            window_ids = segment["window_ids"]
            temporary = tempfile.TemporaryDirectory(
                prefix=f"pose_solver_sam_{view}_"
            )
            source_dir = Path(temporary.name)
            for timestamp in window_ids:
                frame_source = frame_path(config.frames_dir, view, timestamp)
                (source_dir / frame_source.name).symlink_to(frame_source.resolve())
            try:
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    segment_masks, info = track_video_part_from_mask(
                        tracker,
                        str(source_dir),
                        window_ids.index(seed),
                        seed_mask,
                        object_id=part.id,
                    )
            finally:
                temporary.cleanup()
            masks.update({
                segment["window_offset"] + int(index): mask
                for index, mask in segment_masks.items()
            })
            segment_details.append({
                **info,
                "seed_frame": seed,
                "bbox_source": source,
                "qwen_bbox_2d": box,
                "seed_segmentation": seed_info,
                "requested_range": [
                    segment["requested"][0], segment["requested"][-1]
                ],
                "requested_frames": len(segment["requested"]),
                "video_frames_loaded": len(window_ids),
            })
            print(
                f"{view}: anchor {seed}, window "
                f"{window_ids[0]}..{window_ids[-1]}, coverage "
                f"{info.get('coverage', 0.0):.3f}",
                flush=True,
            )

        with Image.open(frame_path(config.frames_dir, view, timestamps[0])) as image:
            shape = (image.height, image.width)
        empty_frames = 0
        nonempty_timestamps: set[str] = set()
        for timestamp in timestamps:
            frame_index = frame_ids.index(timestamp)
            mask = masks.get(frame_index)
            if mask is None or not np.asarray(mask, dtype=bool).any():
                mask = np.zeros(shape, dtype=bool)
                empty_frames += 1
            else:
                nonempty_timestamps.add(timestamp)
            path = track_path(config.tracks_root, args.part, timestamp, view)
            if args.skip_existing and path.exists():
                continue
            _save_track(args, config, timestamp, view, mask)
        raw_coverage = (len(timestamps) - empty_frames) / max(len(timestamps), 1)
        visible_timestamps, visibility_info = _visibility_reference(
            config, view, timestamps
        )
        coverage = len(nonempty_timestamps.intersection(visible_timestamps)) / max(
            len(visible_timestamps), 1
        )
        anchor_nonempty = sum(
            bool(
                frame_ids.index(anchor) in masks
                and np.asarray(masks[frame_ids.index(anchor)], dtype=bool).any()
            )
            for anchor in anchors
            if anchor in frame_ids
        )
        anchor_coverage = anchor_nonempty / max(len(anchors), 1)
        details[view] = {
            "backend": "sam31_instance_tracker_segmented",
            "coverage": coverage,
            "raw_coverage": raw_coverage,
            "visibility_reference": visibility_info,
            "anchor_coverage": anchor_coverage,
            "anchor_frames": anchors,
            "configured_seed_frame": configured_seed,
            "segments": segment_details,
            "written_frames": len(timestamps),
            "empty_frames": empty_frames,
            "video_frames_loaded": sum(
                item["video_frames_loaded"] for item in segment_details
            ),
        }
        quality_settings = config.raw.get("mask_quality", {})
        minimum_anchor_coverage = float(
            quality_settings.get("minimum_anchor_coverage", 1.0)
        )
        if anchor_coverage < minimum_anchor_coverage:
            raise RuntimeError(
                f"Qwen-anchor coverage gate failed for {args.part}/{view}: "
                f"{anchor_coverage:.3f} < {minimum_anchor_coverage:.3f}"
            )
        minimum_coverage = quality_settings.get("minimum_video_coverage")
        if minimum_coverage is not None and coverage < float(minimum_coverage):
            raise RuntimeError(
                f"mask coverage gate failed for {args.part}/{view}: "
                f"{coverage:.3f} < {float(minimum_coverage):.3f}"
            )
        print(f"{view}: {details[view]}", flush=True)
    return details


def main(default_mode: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--pipeline", dest="config", required=True)
    parser.add_argument(
        "--mode",
        choices=("video", "image", "fixed-image"),
        default=default_mode or "video",
    )
    parser.add_argument("--part", required=True)
    parser.add_argument("--view")
    parser.add_argument("--views", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--timestamps", nargs="+")
    parser.add_argument("--seed-frame", "--init-timestamp", dest="seed_frame")
    parser.add_argument("--range-start")
    parser.add_argument("--range-end")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--minimum-pixels", type=int, default=500)
    parser.add_argument("--hold-previous", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--legacy-palette-output",
        action="store_true",
        help="also write one-part indexed masks to legacy config masks_dir",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = load_mask_pipeline_config(args.config)
    if args.part not in config.part_map:
        raise ValueError(f"unknown part {args.part!r}")
    if args.view and args.views:
        raise ValueError("--view and --views are mutually exclusive")
    views = args.views or ([args.view] if args.view else list(config.views))
    unknown_views = set(views).difference(config.views)
    if unknown_views:
        raise ValueError(f"unknown views: {sorted(unknown_views)}")
    frame_ids = validate_synchronized_frames(config.frames_dir, views)
    bbox_data = load_bbox_json(config.bbox_path)
    timestamps = _requested_frames(args, frame_ids, bbox_data)
    timestamps = [
        timestamp for timestamp in timestamps
        if int(timestamp) >= config.part_map[args.part].start_frame
    ]
    if not timestamps:
        raise ValueError("no requested frames remain after the part start frame")
    if args.mode == "video":
        details = _video_mode(
            args, config, bbox_data, views, frame_ids, timestamps
        )
    else:
        details = _image_mode(args, config, bbox_data, views, timestamps)
    manifest = {
        "method": (
            "sam31_seed_mask_instance_video"
            if args.mode == "video"
            else f"sam3_{args.mode}"
        ),
        "part": args.part,
        "views": views,
        "timestamps": [timestamps[0], timestamps[-1]],
        "frames_requested": len(timestamps),
        "details": details,
    }
    write_json(
        config.work_root / "manifests" / f"track_{args.part}_{args.mode}.json",
        manifest,
    )
    print(f"track -> {config.tracks_root / args.part}", flush=True)


if __name__ == "__main__":
    main()
