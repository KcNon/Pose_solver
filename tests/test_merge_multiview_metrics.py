from pathlib import Path
import tempfile
import unittest

from common.io_utils import write_json
from tools.diagnostics.merge_multiview_metrics import merge_reports


class MergeMultiviewMetricsTest(unittest.TestCase):
    def _report(self, frame, iou):
        return {
            "config": "/config.json",
            "trajectory": "/trajectory.json",
            "resolution": [10, 10],
            "frames": {
                frame: {
                    "view": {
                        "part": {
                            "silhouette_iou": iou,
                            "contour_chamfer_px": 1.0,
                            "mask_pixels": 2,
                            "rendered_pixels": 2,
                        }
                    }
                }
            },
            "summary": {"part": {}},
        }

    def test_merges_frames_and_recomputes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_json(first, self._report("000001", 0.5))
            write_json(second, self._report("000002", 1.0))
            merged = merge_reports([first, second])
        self.assertEqual(list(merged["frames"]), ["000001", "000002"])
        self.assertAlmostEqual(
            merged["summary"]["part"]["all_views"]["mean_iou"], 0.75
        )


if __name__ == "__main__":
    unittest.main()
