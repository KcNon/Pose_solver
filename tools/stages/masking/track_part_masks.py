#!/usr/bin/env python
"""Generate independent binary part tracks with SAM3.

Modes:
  video        one Qwen seed per view, followed by bidirectional propagation
  image        every requested frame uses its own Qwen box
  fixed-image  one Qwen seed box is reused for every requested frame

Run this entry point in the SAM environment.  It never writes final palette
masks, so rerunning one part cannot erase another part.
"""
from __future__ import annotations

import argparse
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
    save_binary_mask,
    save_label_mask,
    track_path,
    validate_synchronized_frames,
    write_json,
)
from common.masking.sam import (
    predict_image_mask,
    select_qwen_box,
    track_video_part,
)
from common.masking.schema import MaskPipelineConfig, load_mask_pipeline_config


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
    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        version="sam3.1",
        checkpoint_path=config.raw["sam_ckpt"],
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        async_loading_frames=False,
    )
    model = predictor.model
    part = config.part_map[args.part]
    details: dict[str, Any] = {}
    for view in views:
        seed = _seed_for(config, args.part, view, args.seed_frame)
        if seed not in frame_ids:
            raise ValueError(f"seed frame {seed} is missing for {view}")
        box, source = select_qwen_box(
            bbox_data,
            seed,
            view,
            args.part,
            overrides=config.raw.get("bbox_overrides"),
            fallback_labels=_fallback_labels(config, args.part),
        )
        window_ids, window_offset = _tracking_window(
            frame_ids, timestamps, seed
        )
        temporary = None
        source_dir = config.frames_dir / view
        if len(window_ids) != len(frame_ids):
            temporary = tempfile.TemporaryDirectory(
                prefix=f"pose_solver_sam_{view}_"
            )
            source_dir = Path(temporary.name)
            for timestamp in window_ids:
                frame_source = frame_path(
                    config.frames_dir, view, timestamp
                )
                (source_dir / frame_source.name).symlink_to(
                    frame_source.resolve()
                )
        try:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                masks, info = track_video_part(
                    model,
                    str(source_dir),
                    window_ids.index(seed),
                    box,
                    part.prompts[0],
                )
        finally:
            if temporary is not None:
                temporary.cleanup()
        masks = {
            window_offset + int(index): mask
            for index, mask in masks.items()
        }
        with Image.open(frame_path(config.frames_dir, view, timestamps[0])) as image:
            shape = (image.height, image.width)
        empty_frames = 0
        for timestamp in timestamps:
            frame_index = frame_ids.index(timestamp)
            mask = masks.get(frame_index)
            if mask is None:
                mask = np.zeros(shape, dtype=bool)
                empty_frames += 1
            path = track_path(config.tracks_root, args.part, timestamp, view)
            if args.skip_existing and path.exists():
                continue
            _save_track(args, config, timestamp, view, mask)
        details[view] = {
            **info,
            "seed_frame": seed,
            "bbox_source": source,
            "qwen_bbox_2d": box,
            "written_frames": len(timestamps),
            "empty_frames": empty_frames,
            "video_frames_loaded": len(window_ids),
        }
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
        "method": f"sam3_{args.mode}",
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
