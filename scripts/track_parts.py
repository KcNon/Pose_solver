"""SAM3.1 multiplex video tracking of rice-cooker parts across one view sequence.

Uses point prompts with explicit obj_id (SAM2-style interactive prompts) so each
part keeps a stable identity across the sequence. Runs the SAM3.1 multiplex video
predictor over a view's frame sequence and exports per-frame per-part binary masks
plus overlay visualizations.

Prompt config (JSON):
{
  "view": "2-1",
  "objects": {
    "1": {"name": "lid",       "frame": 19,
          "points": [[0.42,0.42]], "point_labels": [1]},
    "2": {"name": "inner_pot", "frame": 19,
          "points": [[0.75,0.30]], "point_labels": [1]},
    "3": {"name": "body",      "frame": 19,
          "points": [[0.60,0.80]], "point_labels": [1]}
  }
}
Negative points (to exclude a neighbouring part) use label 0.
"""
import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
import torch

SEQ_DIR = "/data_ft_9_10/wentai/projects/pose_solver/outputs/seq"
CKPT = "/data_ft_9_10/wentai/projects/sam3/sam3.1/sam3.1_multiplex.pt"

COLORS = {
    1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0),
    4: (0, 255, 255), 5: (255, 0, 255), 6: (255, 255, 0),
}


def collect(gen, per_frame):
    for frame_idx, out in gen:
        d = per_frame.setdefault(int(frame_idx), {})
        obj_ids = out["out_obj_ids"]
        masks = out["out_binary_masks"]
        for oid, m in zip(np.asarray(obj_ids).tolist(), masks):
            d[int(oid)] = np.asarray(m).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prob", type=float, default=0.5)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from sam3.model_builder import build_sam3_predictor

    with open(args.config) as f:
        cfg = json.load(f)
    view = cfg["view"]
    view_dir = os.path.join(SEQ_DIR, view)
    names = {int(k): v["name"] for k, v in cfg["objects"].items()}

    mask_dir = os.path.join(args.out, "masks")
    vis_dir = os.path.join(args.out, "vis")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    predictor = build_sam3_predictor(
        version="sam3.1", checkpoint_path=CKPT,
        use_fa3=False, use_rope_real=False, compile=False,
    )
    model = predictor.model

    per_frame = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = model.init_state(resource_path=view_dir, use_cv2=True)
        for oid_str, spec in cfg["objects"].items():
            oid = int(oid_str)
            pts = torch.tensor(spec["points"], dtype=torch.float32)
            lbls = torch.tensor(spec["point_labels"], dtype=torch.int32)
            model.add_prompt(
                inference_state=state, frame_idx=int(spec["frame"]), obj_id=oid,
                points=pts, point_labels=lbls, clear_old_points=True,
                rel_coordinates=True, output_prob_thresh=args.prob,
            )
            print(f"added obj {oid} ({names[oid]}) @frame {spec['frame']}")

        init_frame = int(cfg.get("init_frame",
                                 min(int(s["frame"]) for s in cfg["objects"].values())))
        collect(model.propagate_in_video(state, start_frame_idx=init_frame, reverse=False,
                                         output_prob_thresh=args.prob), per_frame)
        collect(model.propagate_in_video(state, start_frame_idx=init_frame, reverse=True,
                                         output_prob_thresh=args.prob), per_frame)

    frame_files = sorted(os.listdir(view_dir), key=lambda p: int(os.path.splitext(p)[0]))
    summary = {}
    for fidx, fname in enumerate(frame_files):
        img = cv2.imread(os.path.join(view_dir, fname))
        vis = img.copy()
        d = per_frame.get(fidx, {})
        summary[fidx] = {}
        for oid in sorted(d.keys()):
            m = d[oid]
            summary[fidx][names.get(oid, oid)] = int(m.sum())
            if m.sum() == 0:
                continue
            c = COLORS.get(oid, (200, 200, 200))
            colored = np.zeros_like(vis)
            colored[m] = c
            vis = cv2.addWeighted(vis, 1.0, colored, 0.5, 0)
            ys, xs = np.where(m)
            cv2.putText(vis, names.get(oid, str(oid)),
                        (int(xs.mean()), int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(
                mask_dir, f"{fidx:05d}_{oid}_{names.get(oid, oid)}.png"),
                (m * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(vis_dir, f"{fidx:05d}.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps(summary))
    print("done ->", args.out)


if __name__ == "__main__":
    main()
