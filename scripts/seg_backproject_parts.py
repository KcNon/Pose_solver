"""Segment parts on original frames, resize masks to DA3 depth resolution, backproject.

Why not segment on DA3's 504x280 images?
  SAM3 works much better on full-res source frames; DA3 images are heavily downscaled
  and lose object detail. Masks are resized to depth map size before backprojection.

Output:
  outputs/masks_da3/<ts>/<view>_{part}.png   (504x280, aligned with depth)
  outputs/masks_da3/<ts>/<view>_overlay.jpg  (full-res visualization)
  outputs/parts_ply/<ts>/<part>.ply
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

DA3 = "/data_ft_9_10/wentai/projects/vggt-omega/试标数据-6.30/2/output_test/da3_output"
FRAMES = "/data_ft_9_10/wentai/projects/vggt-omega/试标数据-6.30/2/output_test/frames"
CKPT = "/data_ft_9_10/wentai/projects/sam3/sam3.1/sam3.1_multiplex.pt"
OUT_MASK = os.path.join(ROOT, "outputs/masks_da3")
OUT_PLY = os.path.join(ROOT, "outputs/parts_ply")
VIEW_NAMES = ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6"]


def largest_cc(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return lab == keep


def best_mask(processor, state, prompts: list[str], conf: float, bf16):
    """Try multiple text prompts; return highest-scoring mask across all."""
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


def resize_mask_to_depth(mask: np.ndarray, depth_hw: tuple[int, int]) -> np.ndarray:
    h, w = depth_hw
    small = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return small.astype(bool)


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


def segment_view(processor, frame_bgr: np.ndarray, parts: list[str], prompts: dict, conf: float, bf16):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    H, W = rgb.shape[:2]
    with bf16:
        state = processor.set_image(pil)

    masks: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    # optional whole-object constraint
    cooker, cs, cp = best_mask(processor, state, ["rice cooker"], conf, bf16)
    if cooker is None:
        cooker = np.zeros((H, W), bool)
    meta["cooker"] = {"score": cs, "prompt": cp, "px": int(cooker.sum())}

    for pname in parts:
        m, sc, pr = best_mask(processor, state, prompts[pname], conf, bf16)
        if m is None:
            m = np.zeros((H, W), bool)
        # only constrain parts that should stay inside the appliance shell
        if pname in ("body", "inner_pot") and cooker.sum() > 500 and sc >= conf:
            m &= cooker
        masks[pname] = m
        meta[pname] = {"score": sc, "prompt": pr, "px": int(m.sum())}

    # body fallback: cooker minus other parts
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
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-pts", type=int, default=80000)
    ap.add_argument("--skip-seg", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=CKPT, load_from_HF=False, device="cuda", eval_mode=True
    )
    processor = Sam3Processor(model, confidence_threshold=args.conf)
    bf16 = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    from common.icp import write_ply

    for ts in args.timestamps:
        parts, prompts = load_parts_config(ts)
        npz_path = os.path.join(DA3, ts, "exports/npz/results.npz")
        d = np.load(npz_path)
        img, depth, K, E, conf = d["image"], d["depth"], d["intrinsics"], d["extrinsics"], d["conf"]
        depth_hw = depth.shape[1:3]
        conf_thr = float(np.median(conf))
        odir = os.path.join(OUT_MASK, ts)
        os.makedirs(odir, exist_ok=True)
        ply_dir = os.path.join(OUT_PLY, ts)
        os.makedirs(ply_dir, exist_ok=True)

        view_masks: dict[str, list[np.ndarray]] = {p: [] for p in parts}

        for v, vname in enumerate(VIEW_NAMES):
            frame_path = os.path.join(FRAMES, ts, f"{vname}.png")
            if not os.path.exists(frame_path):
                raise FileNotFoundError(frame_path)

            if not args.skip_seg:
                frame_bgr = cv2.imread(frame_path)
                masks, meta, rgb = segment_view(processor, frame_bgr, parts, prompts, args.conf, bf16)

                vis = frame_bgr.copy()
                colors = {"lid": (0, 0, 255), "body": (255, 0, 0), "inner_pot": (0, 255, 0)}
                for pname in parts:
                    m_full = masks[pname]
                    m_depth = resize_mask_to_depth(m_full, depth_hw)
                    view_masks[pname].append(m_depth)
                    cv2.imwrite(os.path.join(odir, f"{v}_{pname}.png"), (m_depth * 255).astype(np.uint8))
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
