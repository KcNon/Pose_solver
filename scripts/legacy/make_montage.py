"""Build montage grids to quickly inspect which timestamps contain the object.

For each view (2-1..2-6) produce one grid image over all timestamps.
"""
import argparse
import os

import cv2
import numpy as np

FRAMES_DIR = "/data_ft_9_10/wentai/projects/vggt-omega/试标数据-6.30/2/output_test/frames"
OUT_DIR = "/data_ft_9_10/wentai/projects/pose_solver/outputs/vis"


def build_grid(view: str, cell_w: int = 320, cols: int = 5):
    ts_dirs = sorted(d for d in os.listdir(FRAMES_DIR) if d.isdigit())
    cells = []
    for ts in ts_dirs:
        p = os.path.join(FRAMES_DIR, ts, f"{view}.png")
        if not os.path.exists(p):
            img = np.zeros((int(cell_w * 9 / 16), cell_w, 3), np.uint8)
        else:
            img = cv2.imread(p)
            h, w = img.shape[:2]
            img = cv2.resize(img, (cell_w, int(cell_w * h / w)))
        cv2.putText(img, ts, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cells.append(img)
    ch = max(c.shape[0] for c in cells)
    cw = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, cw - c.shape[1],
                                cv2.BORDER_CONSTANT) for c in cells]
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"montage_{view}.jpg")
    cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("wrote", out, grid.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", default=["2-1", "2-4"])
    args = ap.parse_args()
    for v in args.views:
        build_grid(v)
