import unittest

import numpy as np

from common.quality_cloud import (
    ViewCloud,
    assign_cross_view_support,
    eroded_mask,
    fuse_supported_clouds,
    smooth_depth_mask,
    reprojection_depth_consistency,
)


class QualityCloudTests(unittest.TestCase):
    def test_mask_erosion_removes_boundary(self) -> None:
        mask = np.zeros((7, 7), bool)
        mask[1:6, 1:6] = True
        actual = eroded_mask(mask, 1)
        self.assertEqual(int(actual.sum()), 9)

    def test_depth_edge_filter_rejects_discontinuity(self) -> None:
        depth = np.ones((7, 7), np.float32)
        depth[:, 4:] = 1.1
        smooth = smooth_depth_mask(depth, 0.01)
        self.assertFalse(smooth[3, 3])
        self.assertFalse(smooth[3, 4])
        self.assertTrue(smooth[3, 2])

    def test_cross_view_support_and_fusion(self) -> None:
        empty_color = np.zeros((2, 3), np.uint8)
        confidence = np.ones(2, np.float32)
        clouds = [
            ViewCloud(np.array([[0, 0, 0], [1, 0, 0]], float), empty_color,
                      confidence, np.zeros(2, np.int16), 2),
            ViewCloud(np.array([[0.002, 0, 0], [2, 0, 0]], float), empty_color,
                      confidence, np.zeros(2, np.int16), 2),
        ]
        assign_cross_view_support(clouds, 0.005)
        self.assertEqual(clouds[0].support.tolist(), [1, 0])
        self.assertEqual(clouds[1].support.tolist(), [1, 0])
        points, _, stats = fuse_supported_clouds(clouds, min_support=1)
        self.assertEqual(len(points), 2)
        self.assertEqual(stats["fused_points"], 2)

    def test_reprojection_depth_consistency(self) -> None:
        points = np.array([[0.0, 0.0, 1.0]], float)
        cloud = ViewCloud(points, np.zeros((1, 3), np.uint8), np.ones(1, np.float32),
                          np.zeros(1, np.int16), 1)
        depth = np.ones((2, 3, 3), np.float32)
        K = np.repeat(np.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]]), 2, axis=0)
        E = np.repeat(np.eye(4)[None], 2, axis=0)
        masks = [np.ones((3, 3), bool), np.ones((3, 3), bool)]
        result = reprojection_depth_consistency([cloud, cloud], depth, K, E, masks,
                                                target_mask_erode=0)
        self.assertEqual(result["samples"], 2)
        self.assertAlmostEqual(result["median_m"], 0.0)
        self.assertAlmostEqual(result["inlier_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
