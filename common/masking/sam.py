"""Shared SAM image-segmentation and video-tracking helpers."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .quality import clean_mask, largest_component


def normalized_xyxy_to_cxcywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) / 1000.0 for value in box)
    x1, x2 = sorted((np.clip(x1, 0.0, 1.0), np.clip(x2, 0.0, 1.0)))
    y1, y2 = sorted((np.clip(y1, 0.0, 1.0), np.clip(y2, 0.0, 1.0)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid normalized bbox: {box}")
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def normalized_xyxy_to_xywh(box: list[float]) -> list[float]:
    cx, cy, width, height = normalized_xyxy_to_cxcywh(box)
    return [cx - width / 2, cy - height / 2, width, height]


def bbox_iou_xywh(box_a: np.ndarray, box_b: list[float]) -> float:
    ax, ay, aw, ah = (float(value) for value in box_a)
    bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def select_qwen_box(
    bbox_data: dict[str, Any],
    timestamp: str,
    view: str,
    part: str,
    *,
    overrides: dict[str, Any] | None = None,
    fallback_labels: Iterable[str] = (),
) -> tuple[list[float], str]:
    override = (overrides or {}).get(timestamp, {}).get(view, {}).get(part)
    if override is not None:
        if len(override) != 4:
            raise ValueError(f"invalid bbox override for {timestamp}/{view}/{part}")
        return [float(value) for value in override], "config_override"
    items = bbox_data.get("frames", {}).get(timestamp, {}).get(view, {}).get("parts", [])
    allowed = {part, *fallback_labels}
    candidates = [
        (
            str(item.get("label")),
            [float(value) for value in item["bbox_2d"]],
        )
        for item in items
        if str(item.get("label")) in allowed and len(item.get("bbox_2d", [])) == 4
    ]
    if not candidates:
        raise RuntimeError(f"missing Qwen {part} bbox for {timestamp}/{view}")
    label, box = max(
        candidates,
        key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
    )
    return box, "qwen" if label == part else f"qwen_fallback:{label}"


def predict_image_mask(
    processor,
    state: dict[str, Any],
    prompts: Iterable[str],
    box: list[float],
    *,
    confidence: float,
    autocast,
    minimum_pixels: int = 500,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    processor.set_confidence_threshold(confidence)
    geometry = normalized_xyxy_to_cxcywh(box)
    best_mask, best_score, best_prompt = None, -1.0, ""
    for prompt in prompts:
        with autocast:
            work = dict(state)
            processor.reset_all_prompts(work)
            processor.set_text_prompt(prompt=str(prompt), state=work)
            output = processor.add_geometric_prompt(
                box=geometry, label=True, state=work
            )
        if output["scores"].numel() == 0:
            continue
        scores = output["scores"].float().cpu().numpy()
        masks = output["masks"].squeeze(1).cpu().numpy().astype(bool)
        index = int(np.argmax(scores))
        if float(scores[index]) > best_score:
            best_mask = clean_mask(masks[index])
            best_score = float(scores[index])
            best_prompt = str(prompt)
    if best_mask is None or int(best_mask.sum()) < minimum_pixels:
        return None, {"status": "empty", "bbox_2d": box}
    return best_mask, {
        "status": "ok",
        "score": best_score,
        "prompt": best_prompt,
        "pixels": int(best_mask.sum()),
        "bbox_2d": box,
    }


def _pick_seed_object(output: dict[str, Any], qwen_xywh: list[float]) -> int | None:
    object_ids = np.asarray(output.get("out_obj_ids", []))
    boxes = np.asarray(output.get("out_boxes_xywh", []))
    if not len(object_ids):
        return None
    if len(boxes) != len(object_ids):
        return int(object_ids[0])
    scores = [bbox_iou_xywh(box, qwen_xywh) for box in boxes]
    return int(object_ids[int(np.argmax(scores))])


def _collect(generator, object_id: int, masks: dict[int, np.ndarray]) -> None:
    for frame_index, output in generator:
        object_ids = np.asarray(output.get("out_obj_ids", []))
        binary = np.asarray(output.get("out_binary_masks", []))
        matches = np.flatnonzero(object_ids == object_id)
        if len(matches):
            masks[int(frame_index)] = largest_component(binary[int(matches[0])])


def track_video_part(
    model,
    view_dir: str,
    seed_index: int,
    qwen_box: list[float],
    prompt: str,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    state = model.init_state(
        resource_path=view_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=False,
        use_cv2=True,
    )
    qwen_xywh = normalized_xyxy_to_xywh(qwen_box)
    _, seed_output = model.add_prompt(
        inference_state=state,
        frame_idx=seed_index,
        text_str=prompt,
        boxes_xywh=[qwen_xywh],
        box_labels=[1],
    )
    object_id = _pick_seed_object(seed_output, qwen_xywh)
    if object_id is None:
        return {}, {"status": "seed_detector_empty", "tracked_frames": 0}
    masks: dict[int, np.ndarray] = {}
    object_ids = np.asarray(seed_output.get("out_obj_ids", []))
    binary = np.asarray(seed_output.get("out_binary_masks", []))
    matches = np.flatnonzero(object_ids == object_id)
    if len(matches):
        masks[seed_index] = largest_component(binary[int(matches[0])])
    _collect(
        model.propagate_in_video(state, start_frame_idx=seed_index, reverse=False),
        object_id,
        masks,
    )
    _collect(
        model.propagate_in_video(state, start_frame_idx=seed_index, reverse=True),
        object_id,
        masks,
    )
    return masks, {
        "status": "ok",
        "seed_object_id": object_id,
        "tracked_frames": len(masks),
        "seed_box_xywh": qwen_xywh,
    }
