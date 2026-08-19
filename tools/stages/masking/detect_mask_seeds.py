#!/usr/bin/env python
"""Detect reusable per-part seed boxes with Qwen3-VL.

Run this entry point in the Qwen environment.  It writes only bbox metadata;
SAM tracks and final palette masks are separate stages.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import re
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


def _detect(
    model,
    processor,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    reference_images: dict[str, Path] | None = None,
):
    content = []
    for label, reference in (reference_images or {}).items():
        content.extend([
            {"type": "image", "image": str(reference)},
            {
                "type": "text",
                "text": (
                    f"The preceding reference image is the configured part "
                    f"with exact label `{label}`. It may be shown from several angles."
                ),
            },
        ])
    content.extend([
        {"type": "image", "image": str(image_path)},
        {
            "type": "text",
            "text": (
                "The final image above is the target scene. Use the preceding "
                "mesh-preview references to distinguish similarly colored parts, "
                "and return boxes only for the final target scene.\n" + prompt
            ),
        },
    ])
    messages = [{
        "role": "user",
        "content": content,
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


def _detect_batch(
    model,
    processor,
    image_paths: list[Path],
    prompt: str,
    max_new_tokens: int,
) -> list[tuple[str, list[dict]]]:
    """Run independent view prompts as one padded Qwen batch."""

    conversations = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {
                "type": "text",
                "text": (
                    "The image above is the target scene. Return boxes only "
                    "for this target scene.\n" + prompt
                ),
            },
        ],
    }] for image_path in image_paths]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    generated = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    input_length = inputs.input_ids.shape[1]
    raw_values = processor.batch_decode(
        generated[:, input_length:], skip_special_tokens=True
    )
    return [(raw, parse_boxes(raw)) for raw in raw_values]


def _request_fingerprint(
    *,
    image_path: Path,
    prompt: str,
    model_path: str,
    max_new_tokens: int,
    max_pixels: int,
    reference_images: dict[str, Path] | None = None,
    mesh_minimum_similarity: float = 0.20,
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
        "references": {
            label: {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for label, path in sorted((reference_images or {}).items())
        },
        "mesh_assignment_version": 5,
        "mesh_minimum_similarity": float(mesh_minimum_similarity),
    }
    return hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _mesh_reference_images(config, parts) -> dict[str, Path]:
    """Resolve ReconViaGen preview images through configured mesh symlinks."""

    mesh_dir = Path(
        config.raw.get("mesh_dir", config.frames_dir.parent / "meshes")
    )
    references = {}
    for part in parts:
        mesh_path = mesh_dir / f"{part.name}.glb"
        if not mesh_path.exists():
            continue
        preview = mesh_path.resolve().parent / "preview.jpg"
        if preview.exists():
            references[part.name] = preview
    return references


def _box_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _unique_candidate_boxes(boxes: list[dict]) -> list[list[float]]:
    candidates: list[list[float]] = []
    for row in boxes:
        box = [float(value) for value in row["bbox_2d"]]
        if not any(_box_iou(box, existing) >= 0.9 for existing in candidates):
            candidates.append(box)
    return candidates


def _canonicalize_candidate_labels(
    boxes: list[dict],
    allowed: set[str],
    *,
    keep_unmatched: bool,
) -> list[dict]:
    """Map parser-normalized labels back to exact mesh part names."""

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    label_map = {normalized(label): label for label in allowed}
    result = []
    for box in boxes:
        mapped = label_map.get(normalized(str(box.get("label", ""))))
        if mapped is not None:
            result.append({**box, "label": mapped})
        elif keep_unmatched:
            result.append(box)
    return result


def _preview_views(path: Path):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    count = max(1, min(8, round(image.width / max(image.height, 1))))
    return [
        image.crop((
            round(index * image.width / count),
            0,
            round((index + 1) * image.width / count),
            image.height,
        ))
        for index in range(count)
    ]


def _target_crop(image, box: list[float], padding: float = 0.25):
    width, height = image.size
    x1, y1, x2, y2 = box
    x1, x2 = x1 / 1000 * width, x2 / 1000 * width
    y1, y2 = y1 / 1000 * height, y2 / 1000 * height
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return image.crop((
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(width, x2 + dx),
        min(height, y2 + dy),
    ))


def _load_dino(model_path: str):
    import torch
    from torchvision.transforms import v2
    from transformers import AutoModel

    transform = v2.Compose([
        v2.Resize((518, 518), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).cuda().eval()
    return model, transform


def _dino_embeddings(model, transform, images):
    import torch

    batch = torch.stack([transform(image) for image in images]).cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        vectors = model(pixel_values=batch).pooler_output.float()
    return torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy()


def _assign_candidates_from_mesh(
    image,
    boxes: list[dict],
    parts,
    references: dict[str, Path],
    dino_model,
    dino_transform,
    minimum_similarity: float = 0.20,
) -> tuple[list[dict], dict]:
    """Assign Qwen proposals to mesh part names with global one-to-one matching."""

    import numpy as np

    candidates = _unique_candidate_boxes(boxes)
    matched_parts = [part for part in parts if part.name in references]
    if not candidates or not matched_parts:
        return boxes, {"status": "not_applicable"}

    reference_views = {
        part.name: _preview_views(references[part.name])
        for part in matched_parts
    }
    images = []
    spans = {}
    for part in matched_parts:
        start = len(images)
        images.extend(reference_views[part.name])
        spans[part.name] = (start, len(images))
    candidate_start = len(images)
    images.extend(_target_crop(image, box) for box in candidates)
    vectors = _dino_embeddings(dino_model, dino_transform, images)

    similarities = np.zeros((len(matched_parts), len(candidates)), dtype=float)
    for part_index, part in enumerate(matched_parts):
        start, end = spans[part.name]
        similarities[part_index] = np.max(
            vectors[start:end] @ vectors[candidate_start:].T,
            axis=0,
        )

    # Qwen is usually better at naming two touching parts, while DINO is most
    # valuable for correcting an unsupported name (for example, a first-frame
    # partial view).  A purely global DINO assignment can swap two correctly
    # named proposals when its similarity margin is only a few hundredths.
    # Lock unique Qwen labels that are independently supported by the matching
    # mesh, then use global one-to-one assignment only for the remainder.
    part_index_by_name = {
        part.name: index for index, part in enumerate(matched_parts)
    }
    candidate_labels: list[str | None] = []
    for candidate in candidates:
        labels = {
            str(row.get("label", ""))
            for row in boxes
            if _box_iou(candidate, row["bbox_2d"]) >= 0.9
            and str(row.get("label", "")) in part_index_by_name
        }
        candidate_labels.append(next(iter(labels)) if len(labels) == 1 else None)
    label_counts = {
        label: candidate_labels.count(label)
        for label in set(candidate_labels)
        if label is not None
    }
    coherent_multi_label_set = (
        len(label_counts) >= 2
        and all(count == 1 for count in label_counts.values())
    )
    locked_pairs = []
    for candidate, label in enumerate(candidate_labels):
        if label is None or label_counts[label] != 1:
            continue
        part = part_index_by_name[label]
        raw_is_mesh_best = part == int(np.argmax(similarities[:, candidate]))
        if (
            float(similarities[part, candidate]) >= minimum_similarity
            and (coherent_multi_label_set or raw_is_mesh_best)
        ):
            locked_pairs.append((part, candidate))

    locked_parts = {part for part, _ in locked_pairs}
    locked_candidates = {candidate for _, candidate in locked_pairs}
    remaining_parts = [
        index for index in range(len(matched_parts))
        if index not in locked_parts
    ]
    remaining_candidates = [
        index for index in range(len(candidates))
        if index not in locked_candidates
    ]
    pair_count = min(len(remaining_parts), len(remaining_candidates))
    best_score = float("-inf")
    best_pairs = list(locked_pairs)
    best_remaining_pairs = []
    if len(remaining_candidates) >= len(remaining_parts):
        for candidate_order in permutations(
            remaining_candidates, len(remaining_parts)
        ):
            score = sum(
                similarities[part, candidate]
                for part, candidate in zip(remaining_parts, candidate_order)
            )
            if score > best_score:
                best_score = float(score)
                best_remaining_pairs = list(zip(remaining_parts, candidate_order))
    else:
        for part_order in permutations(remaining_parts, pair_count):
            score = sum(
                similarities[part, candidate]
                for candidate, part in zip(remaining_candidates, part_order)
            )
            if score > best_score:
                best_score = float(score)
                best_remaining_pairs = [
                    (part, candidate)
                    for candidate, part in zip(remaining_candidates, part_order)
                ]
    best_pairs.extend(best_remaining_pairs)
    best_pairs.sort()

    accepted_pairs = [
        (part, candidate) for part, candidate in best_pairs
        if float(similarities[part, candidate]) >= minimum_similarity
    ]
    assigned = [
        {
            "bbox_2d": candidates[candidate],
            "label": matched_parts[part].name,
        }
        for part, candidate in accepted_pairs
    ]
    report = {
        "status": "ok",
        "candidate_boxes": candidates,
        "parts": [part.name for part in matched_parts],
        "similarities": similarities.tolist(),
        "minimum_similarity": float(minimum_similarity),
        "qwen_locked_parts": [
            matched_parts[part].name for part, _ in locked_pairs
        ],
        "assignments": {
            matched_parts[part].name: {
                "candidate_index": int(candidate),
                "score": float(similarities[part, candidate]),
            }
            for part, candidate in best_pairs
        },
        "accepted_parts": [
            matched_parts[part].name for part, _ in accepted_pairs
        ],
    }
    return assigned, report


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--force", "--fresh", action="store_true")
    parser.add_argument(
        "--mesh-references",
        action="store_true",
        help="include ReconViaGen preview images as visual part references",
    )
    parser.add_argument(
        "--separate-parts",
        action="store_true",
        help="detect each part independently to avoid cross-label confusion",
    )
    parser.add_argument(
        "--only-starting-parts",
        action="store_true",
        help=(
            "at each timestamp, request only parts whose configured start frame "
            "equals that timestamp"
        ),
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=0,
        help=(
            "at each timestamp, request parts from their configured start "
            "through start+N inclusive"
        ),
    )
    parser.add_argument(
        "--reanchor-active-parts",
        action="store_true",
        help=(
            "during configured event windows, also request every part that "
            "has already appeared so long tracks receive semantic re-anchors"
        ),
    )
    args = parser.parse_args()

    config = load_mask_pipeline_config(args.config)
    mesh_minimum_similarity = float(
        config.raw.get("mesh_assignment", {}).get(
            "minimum_similarity", 0.20
        )
    )
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

    bbox_data = load_bbox_json(config.bbox_path)
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
    dino_model = None
    dino_transform = None

    for timestamp in timestamps:
        if args.start_window < 0:
            raise ValueError("--start-window must be non-negative")
        active = []
        for part in selected_parts:
            frame = int(timestamp)
            if args.reanchor_active_parts:
                visible = frame >= part.start_frame
            elif args.start_window:
                visible = part.start_frame <= frame <= (
                    part.start_frame + args.start_window
                )
            elif args.only_starting_parts:
                visible = frame == part.start_frame
            else:
                visible = frame >= part.start_frame
            if visible:
                active.append(part)
        if not active:
            print(f"{timestamp}: no configured part exists yet; skip", flush=True)
            continue
        prompt = args.prompt or build_batch_prompt(active)
        references = (
            _mesh_reference_images(config, active)
            if args.mesh_references else {}
        )
        request_prompt = (
            "\n---\n".join(build_batch_prompt([part]) for part in active)
            if args.separate_parts else prompt
        )
        allowed = {part.name for part in active}
        pending = []
        for view in views:
            image_path = frame_path(config.frames_dir, view, timestamp)
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            request_fingerprint = _request_fingerprint(
                image_path=image_path,
                prompt=request_prompt,
                model_path=model_path,
                max_new_tokens=args.max_new_tokens,
                max_pixels=args.max_pixels,
                reference_images=references,
                mesh_minimum_similarity=mesh_minimum_similarity,
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
            pending.append((view, image_path, request_fingerprint))

        if not pending:
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

        detections: list[tuple[str, list[dict]]] = []
        if args.separate_parts:
            for _, image_path, _ in pending:
                raw_by_part = {}
                boxes = []
                for part in active:
                    part_references = (
                        {part.name: references[part.name]}
                        if part.name in references else {}
                    )
                    part_raw, part_boxes = _detect(
                        model,
                        processor,
                        image_path,
                        build_batch_prompt([part]),
                        args.max_new_tokens,
                        reference_images=part_references,
                    )
                    raw_by_part[part.name] = part_raw
                    boxes.extend(
                        box for box in part_boxes
                        if box["label"] == part.name
                    )
                detections.append((
                    json.dumps(raw_by_part, ensure_ascii=False), boxes
                ))
        else:
            for start in range(0, len(pending), args.batch_size):
                chunk = pending[start:start + args.batch_size]
                detections.extend(_detect_batch(
                    model,
                    processor,
                    [image_path for _, image_path, _ in chunk],
                    prompt,
                    args.max_new_tokens,
                ))

        for (view, image_path, request_fingerprint), (raw, boxes) in zip(
            pending, detections
        ):
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            boxes = _canonicalize_candidate_labels(
                boxes,
                allowed,
                keep_unmatched=bool(args.mesh_references and references),
            )
            mesh_assignment = {"status": "disabled"}
            if args.mesh_references and references:
                if dino_model is None:
                    dino_path = config.raw.get("dino_model") or os.environ.get(
                        "DINO_MODEL_PATH",
                        "/data_ft_9_10/wentai/projects/Hunyuan3D-Omni/.cache/"
                        "huggingface/models--facebook--dinov2-large/snapshots/"
                        "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                    )
                    if not Path(dino_path).exists():
                        raise FileNotFoundError(
                            "DINO model is required for mesh-guided seed assignment: "
                            f"{dino_path}"
                        )
                    dino_model, dino_transform = _load_dino(str(dino_path))
                boxes, mesh_assignment = _assign_candidates_from_mesh(
                    image,
                    boxes,
                    active,
                    references,
                    dino_model,
                    dino_transform,
                    minimum_similarity=mesh_minimum_similarity,
                )
            bbox_data.setdefault("frames", {}).setdefault(timestamp, {})[view] = {
                "image_path": str(image_path),
                "image_size": list(image.size),
                "parts": boxes,
                "raw_output": raw,
                "prompt": (
                    {
                        part.name: build_batch_prompt([part])
                        for part in active
                    }
                    if args.separate_parts else prompt
                ),
                "requested_parts": sorted(allowed),
                "request_fingerprint": request_fingerprint,
                "mesh_references": {
                    label: str(path) for label, path in references.items()
                },
                "mesh_assignment": mesh_assignment,
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
