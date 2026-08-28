import unittest

from tools.diagnostics.compare_multiview_metrics import compact_per_frame


class CompareMultiviewMetricsTests(unittest.TestCase):
    def test_compacts_each_frame_across_views(self):
        report = {
            "frames": {
                "000010": {
                    "left": {"pump": {
                        "silhouette_iou": 0.4,
                        "contour_chamfer_px": 3.0,
                    }},
                    "right": {"pump": {
                        "silhouette_iou": 0.8,
                        "contour_chamfer_px": 1.0,
                    }},
                }
            }
        }

        rows = compact_per_frame(report, "pump")

        self.assertEqual(rows["000010"]["visible_observations"], 2)
        self.assertAlmostEqual(rows["000010"]["mean_iou"], 0.6)
        self.assertAlmostEqual(
            rows["000010"]["mean_contour_chamfer_px"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
