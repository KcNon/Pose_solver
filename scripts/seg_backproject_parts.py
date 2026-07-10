"""Segment parts on full-res frames, resize masks to depth resolution, backproject.

Depth / camera backend is selectable (vggt | da3) via configs/pipeline.json or
--recon-backend. Outputs are split by backend under outputs/masks/<backend>/ and
outputs/parts_ply/<backend>/ so both can be compared side-by-side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.geom import backproject_view
from common.recon_loader import (
    load_recon,
    load_recon_colors,
    output_paths,
    resolve_backend,
)

VIEW_NAMES = ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6"]
OUT_MASK = os.path.join(ROOT, "outputs/masks")
OUT_PLY = os.path.join(ROOT, "outputs/parts_ply")
OUT_BBOX = os.path.join(ROOT, "outputs/bboxes")
OUT_PART_IMAGES = os.path.join(ROOT, "outputs/part_images")


def load_pipeline(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def largest_cc(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return lab == keep


def best_mask(processor, state, prompts: list[str], conf: float, bf16):
    best_m, best_s, best_p = None, -1.0, ""
    processor.set_confidence_threshold(conf)
    for prompt in prompts:
        with bf16:
            out = processor.set_text_prompt(prompt=prompt, state=dict(state))
        if out["scores"].numel() == 0:
            continue
        scores = out["scores"].float().cpu().numpy()
        masks = out["masks"].squeeze(1).float().cpu().numpy().astype(bool)
        i = int(np.argmax(scores))
        if float(scores[i]) > best_s:
            best_s = float(scores[i])
            best_m = masks[i]
            best_p = prompt
    if best_m is None:
        return None, 0.0, ""
    return largest_cc(best_m), best_s, best_p


def bbox_2d_to_cxcywh(bbox_2d: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox_2d
    x1n, x2n = sorted((x1 / 1000.0, x2 / 1000.0))
    y1n, y2n = sorted((y1 / 1000.0, y2 / 1000.0))
    w, h = x2n - x1n, y2n - y1n
    return [(x1n + x2n) / 2, (y1n + y2n) / 2, w, h]


def guided_mask(processor, state, prompts: list[str], bbox_2d: list[float],
                conf: float, bf16):
    best_m, best_s, best_p = None, -1.0, ""
    box = bbox_2d_to_cxcywh(bbox_2d)
    processor.set_confidence_threshold(conf)
    for prompt in prompts:
        with bf16:
            s = dict(state)
            processor.reset_all_prompts(s)
            processor.set_text_prompt(prompt=prompt, state=s)
            out = processor.add_geometric_prompt(box=box, label=True, state=s)
        if out["scores"].numel() == 0:
            continue
        scores = out["scores"].float().cpu().numpy()
        masks = out["masks"].squeeze(1).float().cpu().numpy().astype(bool)
        i = int(np.argmax(scores))
        if float(scores[i]) > best_s:
            best_s = float(scores[i])
            best_m = masks[i]
            best_p = prompt
    if best_m is None:
        return None, 0.0, ""
    return largest_cc(best_m), best_s, best_p


def resize_mask_to_depth(mask: np.ndarray, depth_hw: tuple[int, int]) -> np.ndarray:
    h, w = depth_hw
    small = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return small.astype(bool)


def part_on_black(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep masked pixels; set background to black."""
    out = np.zeros_like(frame_bgr)
    m = mask.astype(bool)
    out[m] = frame_bgr[m]
    return out


def load_parts_config(ts: str) -> tuple[list[str], dict[str, list[str]]]:
    cfg_path = os.path.join(ROOT, "configs", f"parts_{ts}.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    parts = cfg["parts"]
    prompts = cfg.get("prompts", {})
    default = {
        "lid": ["lid"],
        "body": ["rice cooker body", "cooker body without lid"],
        "inner_pot": ["inner pot", "metal bowl"],
        "cooker": ["rice cooker"],
    }
    for p in parts:
        if p not in prompts:
            prompts[p] = default.get(p, [p.replace("_", " ")])
    return parts, prompts


def load_qwen_bboxes(ts: str, vname: str, bbox_root: str) -> dict[str, list[float]]:
    path = os.path.join(bbox_root, ts, f"{vname}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[float]] = {}
    for item in data.get("parts", []):
        label = item.get("label")
        bb = item.get("bbox_2d")
        if label and bb and len(bb) == 4:
            out[str(label)] = [float(v) for v in bb]
    return out


def segment_view(processor, frame_bgr: np.ndarray, parts: list[str], prompts: dict,
                 conf: float, bf16, qwen_bboxes: dict[str, list[float]] | None = None):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    H, W = rgb.shape[:2]
    with bf16:
        state = processor.set_image(pil)

    masks: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    qwen_bboxes = qwen_bboxes or {}

    cooker, cs, cp = best_mask(processor, state, ["rice cooker"], conf, bf16)
    if cooker is None:
        cooker = np.zeros((H, W), bool)
    meta["cooker"] = {"score": cs, "prompt": cp, "px": int(cooker.sum())}

    for pname in parts:
        bbox = qwen_bboxes.get(pname)
        if bbox is not None:
            m, sc, pr = guided_mask(processor, state, prompts[pname], bbox, conf, bf16)
            mode = "qwen_text_bbox"
        else:
            m, sc, pr = best_mask(processor, state, prompts[pname], conf, bf16)
            mode = "text_only" if not qwen_bboxes else "missing_in_qwen"
        if m is None:
            m = np.zeros((H, W), bool)
        if pname in ("body", "inner_pot") and cooker.sum() > 500 and sc >= conf and m.sum() > 0:
            constrained = m & cooker
            if constrained.sum() > 0:
                m = constrained
            else:
                meta.setdefault(pname, {})["cooker_skip"] = "empty_after_intersect"
        masks[pname] = m
        meta[pname] = {
            **meta.get(pname, {}),
            "score": sc, "prompt": pr, "px": int(m.sum()),
            "mode": mode, "qwen_bbox": bbox,
        }

    if "body" in masks and masks["body"].sum() == 0 and cooker.sum() > 500:
        body = cooker.copy()
        if "lid" in masks:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            lid_d = cv2.dilate(masks["lid"].astype(np.uint8), k, 2).astype(bool)
            body &= ~lid_d
        if "inner_pot" in masks:
            body &= ~masks["inner_pot"]
        masks["body"] = largest_cc(body) if body.sum() > 0 else body
        meta["body"] = {**meta.get("body", {}), "fallback": "cooker_minus_parts", "px": int(masks["body"].sum())}

    return masks, meta, rgb


def fuse_part_cloud(depth, img, K, E, conf, part_masks, conf_thr, stride, max_pts):
    all_pts, all_cols = [], []
    for v in range(depth.shape[0]):
        m = part_masks[v].astype(bool)
        m &= np.isfinite(depth[v]) & (depth[v] > 1e-3)
        m &= conf[v] > conf_thr
        sub = np.zeros_like(m)
        sub[::stride, ::stride] = True
        m &= sub
        if m.sum() == 0:
            continue
        pts, cols = backproject_view(depth[v], K[v], E[v], mask=m, color=img[v])
        all_pts.append(pts)
        all_cols.append(cols)
    if not all_pts:
        return np.empty((0, 3), np.float32), None
    pts = np.concatenate(all_pts, 0)
    cols = np.concatenate(all_cols, 0)
    if len(pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamps", nargs="+", required=True)
    ap.add_argument("--pipeline", default=os.path.join(ROOT, "configs", "pipeline.json"))
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-pts", type=int, default=80000)
    ap.add_argument("--skip-seg", action="store_true")
    ap.add_argument("--bbox-root", default=OUT_BBOX)
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--recon-backend", choices=["vggt", "da3"], default=None,
                    help="override configs/pipeline.json recon_backend")
    args = ap.parse_args()

    cfg = load_pipeline(args.pipeline)
    backend = resolve_backend(cfg, args.recon_backend)
    frames_dir = cfg["frames_dir"]
    sam_ckpt = cfg["sam_ckpt"]
    print(f"recon backend: {backend}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=sam_ckpt, load_from_HF=False, device="cuda", eval_mode=True
    )
    processor = Sam3Processor(model, confidence_threshold=args.conf)
    bf16 = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    from common.icp import write_ply

    for ts in args.timestamps:
        parts, prompts = load_parts_config(ts)
        recon = load_recon(cfg, ts, backend=backend)
        depth = recon["depth"]
        conf = recon["conf"]
        K = recon["intrinsics"]
        E = recon["extrinsics"]
        depth_hw = recon["depth_hw"]
        conf_thr = float(np.median(conf))
        img = load_recon_colors(recon, frames_dir, ts, VIEW_NAMES)

        paths = output_paths(ROOT, backend, ts)
        odir = paths["masks"]
        ply_dir = paths["parts_ply"]
        os.makedirs(odir, exist_ok=True)
        part_img_dir = os.path.join(OUT_PART_IMAGES, backend, ts)
        os.makedirs(part_img_dir, exist_ok=True)
        os.makedirs(ply_dir, exist_ok=True)

        view_masks: dict[str, list[np.ndarray]] = {p: [] for p in parts}

        for v, vname in enumerate(VIEW_NAMES):
            frame_path = os.path.join(frames_dir, ts, f"{vname}.png")
            if not os.path.exists(frame_path):
                raise FileNotFoundError(frame_path)

            if not args.skip_seg:
                frame_bgr = cv2.imread(frame_path)
                qwen_bboxes = None if args.text_only else load_qwen_bboxes(ts, vname, args.bbox_root)
                masks, meta, rgb = segment_view(
                    processor, frame_bgr, parts, prompts, args.conf, bf16, qwen_bboxes
                )

                vis = frame_bgr.copy()
                colors = {"lid": (0, 0, 255), "body": (255, 0, 0), "inner_pot": (0, 255, 0)}
                for pname in parts:
                    m_full = masks[pname]
                    m_depth = resize_mask_to_depth(m_full, depth_hw)
                    view_masks[pname].append(m_depth)
                    cv2.imwrite(os.path.join(odir, f"{v}_{pname}.png"), (m_depth * 255).astype(np.uint8))
                    part_img = part_on_black(frame_bgr, m_full)
                    cv2.imwrite(
                        os.path.join(part_img_dir, f"{vname}_{pname}.png"),
                        part_img,
                    )
                    ov = vis.copy()
                    ov[m_full] = colors.get(pname, (255, 255, 255))
                    vis = cv2.addWeighted(vis, 0.55, ov, 0.45, 0)
                cv2.imwrite(os.path.join(odir, f"{v}_overlay.jpg"), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                info = {p: meta[p] for p in parts}
                info["cooker"] = meta.get("cooker", {})
                print(f"{ts} v{v} ({vname}): {info}")
            else:
                for pname in parts:
                    mp = os.path.join(odir, f"{v}_{pname}.png")
                    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    view_masks[pname].append(m > 127 if m is not None else np.zeros(depth_hw, bool))

        for pname in parts:
            part_masks = np.stack(view_masks[pname], 0)
            pts, cols = fuse_part_cloud(depth, img, K, E, conf, part_masks, conf_thr, args.stride, args.max_pts)
            ply_path = os.path.join(ply_dir, f"{pname}.ply")
            write_ply(ply_path, pts, cols)
            print(f"{ts} {pname}: {len(pts)} pts -> {ply_path}")

    print("done")


if __name__ == "__main__":
    main()
