"""Shared SAM image-segmentation and video-tracking helpers."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .quality import clean_mask, largest_component


def build_sam31_instance_tracker(checkpoint_path: str):
    """Build the standalone SAM 3.1 tracker from a merged checkpoint.

    SAM 3.1 multiplex checkpoints store the temporal tracker without its
    visual backbone because the full video model shares detector features.
    For prompt-locked instance tracking we rebuild that backbone and copy the
    exactly matching detector backbone weights into it.
    """

    import types
    import torch
    from sam3.model.data_misc import NestedTensor
    from sam3.model_builder import build_sam3_multiplex_video_model

    tracker = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        use_fa3=False,
        use_rope_real=False,
        device="cpu",
        compile=False,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]

    expected = set(tracker.state_dict())
    remapped = {}
    for key, value in checkpoint.items():
        if key.startswith("tracker.model."):
            remapped[key.removeprefix("tracker.model.")] = value
        elif key.startswith("detector.backbone."):
            candidate = "backbone." + key.removeprefix("detector.backbone.")
            if candidate in expected:
                remapped[candidate] = value

    missing = sorted(expected.difference(remapped))
    unexpected = sorted(set(remapped).difference(expected))
    if missing or unexpected:
        raise RuntimeError(
            "SAM 3.1 instance-tracker checkpoint mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    tracker.load_state_dict(remapped, strict=True)
    del checkpoint, remapped
    tracker = tracker.cuda().eval()

    def get_standalone_image_feature(self, inference_state, frame_idx, batch_size):
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            image = inference_state["images"][frame_idx].cuda().float().unsqueeze(0)
            backbone_out = self.forward_image(
                NestedTensor(tensors=image, mask=None),
                need_sam3_out=False,
                need_interactive_out=True,
                need_propagation_out=True,
            )
            inference_state["cached_features"] = {
                frame_idx: (image, backbone_out)
            }
        return image, self._prepare_backbone_features(backbone_out)

    # The upstream demo asks its tracker-only backbone for an unused SAM3
    # detection head.  Avoiding that output also avoids treating its flat
    # tensors as tracker-neck dictionaries in the upstream post-processing.
    tracker._get_image_feature = types.MethodType(
        get_standalone_image_feature, tracker
    )
    return tracker


class Sam31InstanceBoxProcessor:
    """Small processor adapter for SAM3.1's prompt-locked image predictor."""

    def __init__(self, tracker) -> None:
        import torch
        from sam3.model.utils.sam1_utils import SAM2Transforms

        if not hasattr(tracker, "device"):
            tracker.device = torch.device("cuda")
        self.model = tracker
        self.transforms = SAM2Transforms(
            resolution=tracker.image_size,
            mask_threshold=0.0,
            max_hole_area=256.0,
            max_sprinkle_area=0.0,
        )
        self.original_hw: tuple[int, int] | None = None
        self.interactive_pix_feat = None
        self.interactive_high_res_features = None

    def set_image(self, image) -> dict[str, Any]:
        import torch
        from sam3.model.data_misc import NestedTensor

        width, height = image.size
        with torch.inference_mode():
            input_image = self.transforms(image)[None].to(self.model.device)
            backbone_out = self.model.forward_image(
                NestedTensor(tensors=input_image, mask=None),
                need_sam3_out=False,
                need_interactive_out=True,
                need_propagation_out=False,
            )
            features = self.model._prepare_backbone_features(backbone_out)[
                "interactive"
            ]
            vision_feats = features["vision_feats"]
            feat_sizes = features["feat_sizes"]
            self.interactive_pix_feat = self.model._get_interactive_pix_mem(
                vision_feats, feat_sizes
            )
            self.interactive_high_res_features = [
                feature.permute(1, 2, 0).view(
                    feature.size(1), feature.size(2), *size
                )
                for feature, size in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        self.original_hw = (int(height), int(width))
        return {
            "original_width": int(width),
            "original_height": int(height),
        }

    def predict_box(self, pixel_box: np.ndarray):
        import torch

        if self.original_hw is None or self.interactive_pix_feat is None:
            raise RuntimeError("set_image must be called before predict_box")
        height, width = self.original_hw
        box = np.asarray(pixel_box, dtype=np.float32).reshape(2, 2)
        coords = box.copy()
        coords[:, 0] *= float(self.model.image_size) / width
        coords[:, 1] *= float(self.model.image_size) / height
        point_inputs = {
            "point_coords": torch.as_tensor(
                coords[None], dtype=torch.float32, device=self.model.device
            ),
            "point_labels": torch.as_tensor(
                [[2, 3]], dtype=torch.int32, device=self.model.device
            ),
        }
        with torch.inference_mode():
            multiplex_state = self.model.multiplex_controller.get_state(
                num_valid_entries=1,
                device=self.model.device,
                dtype=torch.float32,
                random=False,
                object_ids=[1],
            )
            output = self.model._forward_sam_heads(
                backbone_features=self.interactive_pix_feat,
                point_inputs=point_inputs,
                interactive_high_res_features=self.interactive_high_res_features,
                multimask_output=False,
                objects_to_interact=[0],
                multiplex_state=multiplex_state,
            )
            masks = self.transforms.postprocess_masks(
                output["high_res_masks"], self.original_hw
            ) > 0.0
        return (
            masks.squeeze(0).float().cpu().numpy(),
            output["ious"].squeeze(0).float().cpu().numpy(),
            output["low_res_masks"].squeeze(0).float().cpu().numpy(),
        )


def build_sam31_instance_box_processor(
    checkpoint_path: str,
    *,
    tracker=None,
) -> Sam31InstanceBoxProcessor:
    """Build an image box processor from the verified multiplex tracker."""
    if tracker is None:
        tracker = build_sam31_instance_tracker(checkpoint_path)
    return Sam31InstanceBoxProcessor(tracker)


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
    prompt_mode: str = "grounded_text_box",
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if prompt_mode == "instance_box":
        height = int(state["original_height"])
        width = int(state["original_width"])
        x1, y1, x2, y2 = (float(value) for value in box)
        pixel_box = np.asarray(
            [
                np.clip(x1 / 1000.0 * width, 0.0, width - 1.0),
                np.clip(y1 / 1000.0 * height, 0.0, height - 1.0),
                np.clip(x2 / 1000.0 * width, 1.0, width),
                np.clip(y2 / 1000.0 * height, 1.0, height),
            ],
            dtype=np.float32,
        )
        if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
            raise ValueError(f"invalid normalized bbox: {box}")
        if not hasattr(processor, "predict_box"):
            raise TypeError(
                "instance_box mode requires Sam31InstanceBoxProcessor"
            )
        with autocast:
            masks, scores, _logits = processor.predict_box(pixel_box)
        masks = np.asarray(masks, dtype=bool)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if not len(masks) or not len(scores):
            return None, {
                "status": "empty",
                "bbox_2d": box,
                "prompt_mode": prompt_mode,
            }
        index = int(np.argmax(scores))
        best_mask = clean_mask(np.asarray(masks[index]).squeeze())
        if int(best_mask.sum()) < minimum_pixels:
            return None, {
                "status": "empty",
                "bbox_2d": box,
                "prompt_mode": prompt_mode,
            }
        return best_mask, {
            "status": "ok",
            "score": float(scores[index]),
            "prompt": "instance_box",
            "prompt_mode": prompt_mode,
            "pixels": int(best_mask.sum()),
            "bbox_2d": box,
        }
    if prompt_mode != "grounded_text_box":
        raise ValueError(f"unsupported SAM image prompt mode: {prompt_mode}")

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
        "prompt_mode": prompt_mode,
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


def track_video_part_from_mask(
    tracker,
    view_dir: str,
    seed_index: int,
    seed_mask: np.ndarray,
    *,
    object_id: int = 1,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Track one prompt-locked instance from an explicit seed mask."""

    import torch
    import sam3.model.video_tracking_multiplex_demo as tracker_demo
    from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

    # The SAM 3.1 class overrides ``init_state`` for detector-shared feature
    # caches.  A standalone tracker must use the parent implementation that
    # loads frames and computes its own backbone features.
    # This SAM checkout has the newer demo call signature and the older frame
    # loader signature.  Adapt them locally instead of modifying the upstream
    # repository.
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
        state = VideoTrackingMultiplexDemo.init_state(
            tracker,
            video_path=view_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
            use_cv2=True,
        )
    finally:
        tracker_demo.load_video_frames = original_loader
    seed = np.asarray(seed_mask, dtype=bool)
    if not seed.any():
        return {}, {"status": "seed_mask_empty", "tracked_frames": 0}
    tracker.add_new_masks(
        state,
        frame_idx=int(seed_index),
        obj_ids=[int(object_id)],
        masks=torch.from_numpy(seed[None]),
    )
    tracker.propagate_in_video_preflight(state, run_mem_encoder=True)

    masks: dict[int, np.ndarray] = {}

    def collect(generator) -> None:
        for output in generator:
            frame_index, object_ids, _, video_masks = output[:4]
            ids = np.asarray(object_ids)
            matches = np.flatnonzero(ids == object_id)
            if not len(matches):
                continue
            logits = video_masks[int(matches[0])]
            binary = logits.squeeze().float().cpu().numpy() > 0.0
            masks[int(frame_index)] = largest_component(binary)

    frame_count = int(state["num_frames"])
    collect(tracker.propagate_in_video(
        state,
        start_frame_idx=int(seed_index),
        max_frame_num_to_track=frame_count,
        reverse=False,
        tqdm_disable=False,
    ))
    collect(tracker.propagate_in_video(
        state,
        start_frame_idx=int(seed_index),
        max_frame_num_to_track=frame_count,
        reverse=True,
        tqdm_disable=False,
    ))
    nonempty = sum(bool(mask.any()) for mask in masks.values())
    return masks, {
        "status": "ok" if nonempty == frame_count else "incomplete",
        "seed_object_id": int(object_id),
        "tracked_frames": len(masks),
        "nonempty_frames": nonempty,
        "video_frames": frame_count,
        "coverage": nonempty / max(frame_count, 1),
        "backend": "sam31_prompt_locked_instance_tracker",
    }
