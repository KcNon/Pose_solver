"""SAM3 text-prompted segmentation on single images.

Loads the SAM3 image (detector) model from the local sam3.1 checkpoint and
runs one or more open-vocabulary text prompts on an image, saving per-prompt
overlay visualizations and raw masks.

Usage:
  python segment_sam3.py --image <path> --prompts "rice cooker" "lid" \
      --out outputs/seg/test --gpu 4
"""
import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch
from PIL import Image

CKPT = "/data_ft_9_10/wentai/projects/sam3/sam3.1/sam3.1_multiplex.pt"

# distinct BGR colors for overlays
COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
]


def build_model(gpu: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=CKPT, load_from_HF=False, device="cuda", eval_mode=True
    )
    processor = Sam3Processor(model)
    return processor


def run_prompt(processor, state, prompt, conf):
    processor.set_confidence_threshold(conf)
    out = processor.set_text_prompt(prompt=prompt, state=dict(state))
    masks = out["masks"].squeeze(1).float().cpu().numpy().astype(bool)  # (N,H,W)
    boxes = out["boxes"].float().cpu().numpy()
    scores = out["scores"].float().cpu().numpy()
    return masks, boxes, scores


def overlay(img_bgr, masks, boxes, scores, color, prompt):
    vis = img_bgr.copy()
    for i, m in enumerate(masks):
        c = color
        colored = np.zeros_like(vis)
        colored[m] = c
        vis = cv2.addWeighted(vis, 1.0, colored, 0.45, 0)
        ys, xs = np.where(m)
        if len(xs):
            x0, y0 = int(xs.min()), int(ys.min())
            cv2.rectangle(vis, (x0, y0), (int(xs.max()), int(ys.max())), c, 2)
            cv2.putText(vis, f"{prompt}:{scores[i]:.2f}", (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    processor = build_model(args.gpu)

    pil = Image.open(args.image).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    bf16 = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    bf16.__enter__()
    state = processor.set_image(pil)

    combined = img_bgr.copy()
    for idx, prompt in enumerate(args.prompts):
        masks, boxes, scores = run_prompt(processor, state, prompt, args.conf)
        print(f"[{prompt}] -> {len(masks)} instances, scores={np.round(scores,3).tolist()}")
        color = COLORS[idx % len(COLORS)]
        vis = overlay(img_bgr, masks, boxes, scores, color, prompt)
        safe = prompt.replace(" ", "_").replace("/", "_")
        cv2.imwrite(os.path.join(args.out, f"{safe}.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        # accumulate into combined
        for m in masks:
            colored = np.zeros_like(combined)
            colored[m] = color
            combined = cv2.addWeighted(combined, 1.0, colored, 0.45, 0)
        np.save(os.path.join(args.out, f"{safe}_masks.npy"), masks)
    cv2.imwrite(os.path.join(args.out, "combined.jpg"), combined,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("saved to", args.out)


if __name__ == "__main__":
    main()
