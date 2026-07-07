"""Convert per-timestamp multi-view frames into per-view JPEG sequences.

SAM3 video loader expects a folder of "<frame_index>.jpg" files. We build one
folder per camera view, where frame index == timestamp index (000000..000019).

Output: outputs/seq/<view>/00000.jpg ... and outputs/seq/index.json
"""
import json
import os

import cv2

FRAMES_DIR = "/data_ft_9_10/wentai/projects/vggt-omega/试标数据-6.30/2/output_test/frames"
OUT_DIR = "/data_ft_9_10/wentai/projects/pose_solver/outputs/seq"
VIEWS = ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6"]


def main():
    ts_dirs = sorted(d for d in os.listdir(FRAMES_DIR) if d.isdigit())
    index = {"views": VIEWS, "timestamps": ts_dirs, "frame_to_ts": {}}
    for view in VIEWS:
        vdir = os.path.join(OUT_DIR, view)
        os.makedirs(vdir, exist_ok=True)
        for i, ts in enumerate(ts_dirs):
            src = os.path.join(FRAMES_DIR, ts, f"{view}.png")
            dst = os.path.join(vdir, f"{i:05d}.jpg")
            img = cv2.imread(src)
            if img is None:
                raise FileNotFoundError(src)
            cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            index["frame_to_ts"][str(i)] = ts
        print(f"{view}: wrote {len(ts_dirs)} frames -> {vdir}")
    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("index written")


if __name__ == "__main__":
    main()
