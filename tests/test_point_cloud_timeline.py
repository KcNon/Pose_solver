import unittest

import numpy as np

from tools.diagnostics.render_point_cloud_timeline import (
    projection_bounds,
    project_world_points,
    robust_pca_extents_mm,
    top_basis,
)


class PointCloudTimelineTests(unittest.TestCase):
    def test_project_world_points_uses_world_to_camera_extrinsic(self):
        points = np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
        intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
        uv, depth = project_world_points(points, intrinsic, np.eye(4))
        np.testing.assert_allclose(uv, [[50.0, 40.0], [100.0, 40.0]])
        np.testing.assert_allclose(depth, [2.0, 2.0])

    def test_projection_bounds_use_one_fixed_metric_span(self):
        points = np.asarray([
            [-1.0, -2.0, 0.0],
            [-1.0, 2.0, 0.0],
            [1.0, -2.0, 0.0],
            [1.0, 2.0, 0.0],
        ])
        bounds = projection_bounds(
            [points], top_basis(), padding=1.0, minimum_span_m=0.1
        )
        np.testing.assert_allclose(bounds.center_xy, [0.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(bounds.span_m, 4.0, places=6)

    def test_pca_extents_are_rotation_invariant(self):
        grid = np.asarray([
            [x, y, z]
            for x in np.linspace(-0.10, 0.10, 9)
            for y in np.linspace(-0.05, 0.05, 7)
            for z in np.linspace(-0.02, 0.02, 5)
        ])
        angle = np.radians(37.0)
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        first = robust_pca_extents_mm(grid)
        second = robust_pca_extents_mm(grid @ rotation.T)
        np.testing.assert_allclose(first, second, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
