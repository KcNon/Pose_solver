"""Segment lid + body on DA3 result images (aligned with depth) for given timestamps.

For each timestamp and each of the 6 views:
  lid_mask    = SAM3("lid")            (highest-score instance)
  cooker_mask = SAM3("rice cooker")    (highest-score instance)
  body_mask   = cooker_mask & ~lid_mask

Masks are saved at DA3 resolution (280x504) so they index depth directly.
Output: outputs/masks_da3/<ts>/<view>_{lid,body,cooker}.png (+ overlay jpg)
"""
import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
from PIL import Image

DA3 = "/data_ft_9_10/wentai/projects/depth-anything-3/试标数据-6.30/2/output_test/da3_output"
CKPT = "/data_ft_9_10/wentai/projects/sam3/sam3.1/sam3.1_multiplex.pt"
OUT = "/data_ft_9_10/wentai/projects/pose_solver/outputs/masks_da3"


def largest_cc(mask):
    mask = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return lab == keep


def best_mask(processor, state, prompt, conf, bf16):
    with bf16:
        out = processor.set_text_prompt(prompt=prompt, state=dict(state))
    if out["scores"].numel() == 0:
        return None, 0.0
    scores = out["scores"].float().cpu().numpy()
    masks = out["masks"].squeeze(1).float().cpu().numpy().astype(bool)
    i = int(np.argmax(scores))
    return masks[i], float(scores[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamps", nargs="+", default=["000010", "000012"])
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False,
                                   device="cuda", eval_mode=True)
    processor = Sam3Processor(model)
    bf16 = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    for ts in args.timestamps:
        d = np.load(os.path.join(DA3, ts, "exports/npz/results.npz"))
        imgs = d["image"]  # (6,H,W,3) RGB
        odir = os.path.join(OUT, ts)
        os.makedirs(odir, exist_ok=True)
        for v in range(imgs.shape[0]):
            rgb = imgs[v]
            pil = Image.fromarray(rgb)
            with bf16:
                state = processor.set_image(pil)
            processor.set_confidence_threshold(args.conf)
            lid, ls = best_mask(processor, state, "lid", args.conf, bf16)
            cooker, cs = best_mask(processor, state, "rice cooker", args.conf, bf16)
            H, W = rgb.shape[:2]
            if lid is None:
                lid = np.zeros((H, W), bool)
            else:
                lid = largest_cc(lid)
            if cooker is None:
                cooker = np.zeros((H, W), bool)
            else:
                cooker = largest_cc(cooker)
            body = cooker & (~lid)
            if body.sum() > 0:
                body = largest_cc(body)

            cv2.imwrite(os.path.join(odir, f"{v}_lid.png"), (lid * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(odir, f"{v}_body.png"), (body * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(odir, f"{v}_cooker.png"), (cooker * 255).astype(np.uint8))

            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ov = vis.copy()
            ov[lid] = (0, 0, 255)
            ov[body] = (255, 0, 0)
            vis = cv2.addWeighted(vis, 0.5, ov, 0.5, 0)
            cv2.imwrite(os.path.join(odir, f"{v}_overlay.jpg"), vis)
            print(f"{ts} v{v}: lid {int(lid.sum())}px({ls:.2f}) "
                  f"cooker {int(cooker.sum())}px({cs:.2f}) body {int(body.sum())}px")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
