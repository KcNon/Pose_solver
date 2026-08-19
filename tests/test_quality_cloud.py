import unittest

import numpy as np

from common.quality_cloud import (
    ViewCloud,
    assign_cross_view_support,
    eroded_mask,
    filter_centroid_consistent_views,
    fuse_supported_clouds,
    quality_gate,
    smooth_depth_mask,
    reprojection_depth_consistency,
    supported_view_clouds,
)


class QualityCloudTests(unittest.TestCase):
    def test_centroid_filter_keeps_largest_consistent_view_group(self) -> None:
        def view(center):
            points = np.asarray(center, float)[None] + np.asarray(
                [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]]
            )
            return ViewCloud(
                points,
                np.zeros((3, 3), np.uint8),
                np.ones(3, np.float32),
                np.zeros(3, np.int16),
                3,
            )

        filtered, report = filter_centroid_consistent_views(
            [view([0.0, 0.0, 1.0]), view([0.02, 0.0, 1.0]), view([0.3, 0.0, 1.0])],
            radius_m=0.05,
            minimum_points=3,
        )
        self.assertEqual(report["selected_view_indices"], [0, 1])
        self.assertEqual([len(cloud.points) for cloud in filtered], [3, 3, 0])

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

    def test_supported_metrics_exclude_singletons_and_gate_fail_closed(self) -> None:
        colors = np.zeros((2, 3), np.uint8)
        confidence = np.ones(2, np.float32)
        clouds = [
            ViewCloud(np.array([[0, 0, 1], [2, 0, 1]], float), colors,
                      confidence, np.array([1, 0], np.int16), 2),
            ViewCloud(np.array([[0.001, 0, 1], [3, 0, 1]], float), colors,
                      confidence, np.array([1, 0], np.int16), 2),
        ]
        filtered = supported_view_clouds(clouds, min_support=1)
        self.assertEqual([len(cloud.points) for cloud in filtered], [1, 1])
        gate = quality_gate(
            {
                "fused_points": 2,
                "views": [
                    {"candidate_points": 2, "supported_points": 1},
                    {"candidate_points": 2, "supported_points": 1},
                ],
            },
            {"median_m": 0.001, "overlap_ratio": 1.0},
            {"median_m": 0.001, "inlier_ratio": 1.0},
            minimum_fused_points=3,
            minimum_supported_views=2,
            minimum_supported_points_per_view=1,
        )
        self.assertFalse(gate["passed"])
        self.assertIn("too_few_fused_points", gate["reasons"])

    def test_explicit_single_view_fallback_skips_unavailable_pair_metrics(self) -> None:
        stats = {
            "fused_points": 500,
            "views": [
                {"candidate_points": 500, "supported_points": 500},
                {"candidate_points": 0, "supported_points": 0},
            ],
        }
        missing_pairs = {
            "median_m": None,
            "overlap_ratio": None,
            "inlier_ratio": None,
        }
        strict = quality_gate(
            stats,
            missing_pairs,
            missing_pairs,
            minimum_supported_views=1,
            minimum_supported_points_per_view=30,
        )
        self.assertFalse(strict["passed"])
        fallback = quality_gate(
            stats,
            missing_pairs,
            missing_pairs,
            minimum_supported_views=1,
            minimum_supported_points_per_view=30,
            allow_single_view=True,
        )
        self.assertTrue(fallback["passed"])
        self.assertTrue(fallback["single_view_fallback_used"])

    def test_reprojection_override_accepts_disjoint_multiview_surfaces(self) -> None:
        stats = {
            "fused_points": 500,
            "views": [
                {"candidate_points": 250, "supported_points": 250},
                {"candidate_points": 250, "supported_points": 250},
            ],
        }
        cross_view = {"median_m": 0.08, "overlap_ratio": 0.04}
        reprojection = {"median_m": 0.005, "inlier_ratio": 0.9}
        strict = quality_gate(
            stats,
            cross_view,
            reprojection,
            minimum_supported_views=2,
            minimum_supported_points_per_view=30,
        )
        self.assertFalse(strict["passed"])
        fallback = quality_gate(
            stats,
            cross_view,
            reprojection,
            minimum_supported_views=2,
            minimum_supported_points_per_view=30,
            allow_reprojection_override=True,
        )
        self.assertTrue(fallback["passed"])
        self.assertTrue(fallback["reprojection_override_used"])


if __name__ == "__main__":
    unittest.main()
