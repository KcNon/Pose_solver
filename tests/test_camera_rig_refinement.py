import unittest

import numpy as np

from common.camera_rig_refinement import (
    RigCorrespondence,
    choose_anchor,
    connected_components,
    corrected_extrinsics,
    correspondence_metrics,
    optimize_depth_corrections,
    optimize_pose_corrections,
)


class CameraRigRefinementTests(unittest.TestCase):
    def _batch(
        self,
        points_a: np.ndarray,
        points_b: np.ndarray,
        rays_a: np.ndarray | None = None,
        rays_b: np.ndarray | None = None,
    ) -> RigCorrespondence:
        rays_a = np.asarray(rays_a) if rays_a is not None else np.tile(
            [0.0, 0.0, 1.0], (len(points_a), 1)
        )
        rays_b = np.asarray(rays_b) if rays_b is not None else rays_a.copy()
        return RigCorrespondence(
            0, 1, np.asarray(points_a, float), np.asarray(points_b, float),
            rays_a, rays_b, 0,
        )

    def test_camera_graph_and_anchor(self) -> None:
        edges = {(0, 1): 20, (1, 2): 30, (3, 4): 10}
        self.assertEqual(connected_components(5, edges), [[0, 1, 2], [3, 4]])
        self.assertEqual(choose_anchor([0, 1, 2], edges), 1)

    def test_depth_correction_recovers_along_ray_offset(self) -> None:
        points = np.asarray([
            [-0.1, 0.0, 1.0],
            [0.0, 0.1, 1.0],
            [0.1, -0.1, 1.0],
            [0.2, 0.0, 1.0],
        ])
        shifted = points + np.asarray([0.0, 0.0, 0.012])
        batch = self._batch(points, shifted)
        correction, status = optimize_depth_corrections(
            [batch], view_count=2, active_views=[0, 1], anchor=0,
            prior_sigma_m=1.0, maximum_correction_m=0.03,
        )
        self.assertTrue(status["success"])
        self.assertAlmostEqual(correction[0], 0.0, places=8)
        self.assertAlmostEqual(correction[1], -0.012, places=4)

    def test_pose_correction_recovers_small_translation(self) -> None:
        rng = np.random.default_rng(2)
        points = rng.normal(size=(80, 3)) * 0.05
        shifted = points + np.asarray([0.008, -0.004, 0.003])
        batch = self._batch(points, shifted)
        deltas, status = optimize_pose_corrections(
            [batch], np.zeros(2), view_count=2, active_views=[0, 1],
            anchor=0, rotation_prior_deg=0.5, translation_prior_m=1.0,
            maximum_translation_m=0.03,
        )
        self.assertTrue(status["success"])
        np.testing.assert_allclose(
            deltas[1, :3, 3], [-0.008, 0.004, -0.003], atol=2e-4
        )
        metrics = correspondence_metrics([batch], np.zeros(2), deltas)
        self.assertLess(metrics["median_m"], 5e-4)

    def test_corrected_extrinsic_matches_cloud_delta(self) -> None:
        extrinsic = np.asarray([np.eye(4)[:3], np.eye(4)[:3]])
        deltas = np.asarray([np.eye(4), np.eye(4)])
        deltas[1, :3, 3] = [0.01, -0.02, 0.03]
        corrected = corrected_extrinsics(extrinsic, deltas)
        np.testing.assert_allclose(
            corrected[1, :3, 3], [-0.01, 0.02, -0.03], atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
