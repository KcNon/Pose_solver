#!/usr/bin/env python3
"""Propagate bounded multi-part seed masks with SAM3.1 video tracking."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


MAX_VIEWS = 8
MAX_PARTS = 8
MAX_FRAMES = 1_000
MAX_SOURCE_PIXELS = 4096 * 4096


def _load_spec(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    views = raw.get("views", {})
    parts = raw.get("parts", {})
    if not 1 <= len(views) <= MAX_VIEWS:
        raise ValueError(f"spec must contain 1..{MAX_VIEWS} views")
    if not 1 <= len(parts) <= MAX_PARTS:
        raise ValueError(f"spec must contain 1..{MAX_PARTS} parts")
    frame_id = str(raw["frame_id"])
    if not frame_id.isdigit():
        raise ValueError("frame_id must be numeric")
    ids = [int(value["id"]) for value in parts.values()]
    if any(value < 1 or value > 255 for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("part IDs must be unique values in [1, 255]")
    order = raw.get("front_to_back", list(parts))
    if len(order) != len(parts) or set(order) != set(parts):
        raise ValueError("front_to_back must contain every part exactly once")
    return raw


def _frame_ids(frames_root: Path, view: str, maximum: int) -> list[str]:
    directory = frames_root / view
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    values = sorted(
        (
            path.stem
            for path in directory.iterdir()
            if path.is_file()
            and path.stem.isdigit()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=int,
    )
    if not 1 <= len(values) <= maximum:
        raise ValueError(f"{view} must contain 1..{maximum} numeric frames")
    return values


def _load_seed(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim != 2 or values.shape != shape:
        raise ValueError(f"invalid seed mask shape {values.shape}, expected {shape}: {path}")
    mask = values != 0
    if not np.any(mask):
        raise ValueError(f"empty seed mask: {path}")
    return mask


def _load_palette_seeds(
    path: Path,
    shape: tuple[int, int],
    part_names: list[str],
    parts: dict[str, Any],
) -> list[np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        if image.mode != "P":
            raise ValueError(f"seed palette mask must use P mode: {path}")
        labels = np.asarray(image)
    if labels.ndim != 2 or labels.shape != shape:
        raise ValueError(
            f"invalid palette seed shape {labels.shape}, expected {shape}: {path}"
        )
    seeds = [labels == int(parts[part]["id"]) for part in part_names]
    empty = [part for part, seed in zip(part_names, seeds) if not np.any(seed)]
    if empty:
        raise ValueError(f"empty parts {empty} in palette seed: {path}")
    return seeds


def _save_binary(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def _save_palette(
    path: Path, labels: np.ndarray, parts: dict[str, Any]
) -> None:
    palette = [0] * (256 * 3)
    for values in parts.values():
        offset = int(values["id"]) * 3
        palette[offset : offset + 3] = [int(value) for value in values["color_rgb"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(labels, dtype=np.uint8), mode="P")
    image.putpalette(palette)
    image.save(path, format="PNG", optimize=True)


def _init_state(tracker, view_dir: Path):
    import sam3.model.video_tracking_multiplex_demo as tracker_demo
    from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

    original_loader = tracker_demo.load_video_frames

    def compatible_loader(
        video_path,
        image_size,
        offload_video_to_cpu,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
        **kwargs,
    ):
        return original_loader(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            video_loader_type="torchcodec" if use_torchcodec else "cv2",
        )

    tracker_demo.load_video_frames = compatible_loader
    try:
        return VideoTrackingMultiplexDemo.init_state(
            tracker,
            video_path=str(view_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
            use_cv2=True,
        )
    finally:
        tracker_demo.load_video_frames = original_loader


def _propagate_view(
    tracker,
    view_dir: Path,
    seed_index: int,
    object_ids: list[int],
    seed_masks: list[np.ndarray],
    frame_count: int,
) -> dict[int, dict[int, np.ndarray]]:
    import torch

    from common.masking.sam import largest_component

    state = _init_state(tracker, view_dir)
    if int(state["num_frames"]) != frame_count:
        raise RuntimeError(
            f"SAM3 loaded {state['num_frames']} frames, expected {frame_count}: {view_dir}"
        )
    collected: dict[int, dict[int, np.ndarray]] = {
        object_id: {seed_index: seed.copy()}
        for object_id, seed in zip(object_ids, seed_masks)
    }
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        tracker.add_new_masks(
            state,
            frame_idx=seed_index,
            obj_ids=object_ids,
            masks=torch.from_numpy(np.stack(seed_masks)),
        )
        tracker.propagate_in_video_preflight(state, run_mem_encoder=True)
        generator = tracker.propagate_in_video(
            state,
            start_frame_idx=seed_index,
            max_frame_num_to_track=frame_count,
            reverse=False,
            tqdm_disable=False,
        )
        for output in generator:
            frame_index, returned_ids, _, video_masks = output[:4]
            ids = np.asarray(returned_ids)
            for object_id in object_ids:
                matches = np.flatnonzero(ids == object_id)
                if not len(matches):
                    continue
                logits = video_masks[int(matches[0])]
                binary = logits.squeeze().float().cpu().numpy() > 0.0
                collected[object_id][int(frame_index)] = largest_component(binary)
    # The requested first frame is the authoritative user-approved seed.
    for object_id, seed in zip(object_ids, seed_masks):
        collected[object_id][seed_index] = seed.copy()
    del state
    torch.cuda.empty_cache()
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--seed-parts-root", type=Path)
    seeds.add_argument("--seed-palette-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_frames <= MAX_FRAMES:
        raise ValueError(f"max-frames must be in [1, {MAX_FRAMES}]")

    import torch

    from common.masking.sam import build_sam31_instance_tracker

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SAM3 propagation requires exactly one visible CUDA device")
    spec = _load_spec(args.spec.resolve())
    frames_root = args.frames_root.resolve()
    seed_parts_root = (
        args.seed_parts_root.resolve() if args.seed_parts_root is not None else None
    )
    seed_palette_root = (
        args.seed_palette_root.resolve() if args.seed_palette_root is not None else None
    )
    output_root = args.output_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (output_root / "manifest.json").exists() or (output_root / "masks").exists():
        raise FileExistsError(f"propagation output already exists: {output_root}")

    views = list(spec["views"])
    reference_ids = _frame_ids(frames_root, views[0], args.max_frames)
    for view in views[1:]:
        if _frame_ids(frames_root, view, args.max_frames) != reference_ids:
            raise RuntimeError(f"frame IDs are not synchronized for {view}")
    seed_id = str(spec["frame_id"])
    if seed_id not in reference_ids:
        raise ValueError(f"seed frame {seed_id} is absent")
    seed_index = reference_ids.index(seed_id)
    if seed_index != 0:
        raise ValueError("this bounded forward propagator requires the seed to be first")

    first_path = frames_root / views[0] / f"{seed_id}.jpg"
    with Image.open(first_path) as first_image:
        shape = (first_image.height, first_image.width)
    if shape[0] * shape[1] > MAX_SOURCE_PIXELS:
        raise ValueError("source image exceeds pixel budget")

    parts = spec["parts"]
    part_names = list(parts)
    object_ids = [int(parts[name]["id"]) for name in part_names]
    tracker = build_sam31_instance_tracker(str(checkpoint))
    report: dict[str, Any] = {
        "method": "sam31_multi_object_seed_mask_forward",
        "checkpoint": str(checkpoint),
        "seed_frame": seed_id,
        "frame_range": [reference_ids[0], reference_ids[-1]],
        "frames": len(reference_ids),
        "views": {},
    }
    for view in views:
        if seed_parts_root is not None:
            seeds = [
                _load_seed(
                    seed_parts_root / part / view / f"{seed_id}.png", shape
                )
                for part in part_names
            ]
            seed_source = "independent_binary_masks"
        else:
            assert seed_palette_root is not None
            seeds = _load_palette_seeds(
                seed_palette_root / view / f"{seed_id}.png",
                shape,
                part_names,
                parts,
            )
            seed_source = "P_mode_palette_labels"
        tracked = _propagate_view(
            tracker,
            frames_root / view,
            seed_index,
            object_ids,
            seeds,
            len(reference_ids),
        )
        view_report: dict[str, Any] = {
            "seed_source": seed_source,
            "parts": {},
        }
        for part, object_id in zip(part_names, object_ids):
            missing = [index for index in range(len(reference_ids)) if index not in tracked[object_id]]
            if missing:
                raise RuntimeError(f"incomplete propagation for {view}/{part}: {missing}")
            pixel_counts = []
            for index, frame_id in enumerate(reference_ids):
                mask = tracked[object_id][index]
                if mask.shape != shape:
                    raise RuntimeError(f"invalid propagated mask for {view}/{part}/{frame_id}")
                _save_binary(
                    output_root / "part_masks" / part / view / f"{frame_id}.png",
                    mask,
                )
                pixel_counts.append(int(mask.sum()))
            view_report["parts"][part] = {
                "frames": len(pixel_counts),
                "nonempty_frames": sum(value > 0 for value in pixel_counts),
                "minimum_pixels": min(pixel_counts),
                "maximum_pixels": max(pixel_counts),
                "seed_pixels": pixel_counts[0],
                "final_pixels": pixel_counts[-1],
            }
        for index, frame_id in enumerate(reference_ids):
            labels = np.zeros(shape, dtype=np.uint8)
            for part in reversed(spec["front_to_back"]):
                object_id = int(parts[part]["id"])
                labels[tracked[object_id][index]] = object_id
            _save_palette(output_root / "masks" / view / f"{frame_id}.png", labels, parts)
        report["views"][view] = view_report
        print(f"{view}: propagated {len(reference_ids)} frames x {len(parts)} parts", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"propagated masks -> {output_root / 'masks'}", flush=True)


if __name__ == "__main__":
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/diagnostics/propagate_sam3_seed_masks.py")
    main()
