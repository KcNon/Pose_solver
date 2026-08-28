from __future__ import annotations

import unittest

import cv2
import numpy as np

from common.rigid_observation import (
    halfspace_face_mask,
    pose_guided_rigid_region,
    thick_core_region,
)


class RigidObservationTests(unittest.TestCase):
    def test_pose_guide_removes_unsupported_flexible_appendage(self):
        source = np.zeros((80, 80), dtype=bool)
        source[20:50, 20:50] = True
        source[30:35, 50:78] = True
        rendered = np.zeros_like(source)
        rendered[21:49, 21:49] = True

        selected, report = pose_guided_rigid_region(
            source, rendered, dilation_radius=2
        )

        self.assertEqual(report["status"], "ok")
        self.assertTrue(selected[30, 49])
        self.assertFalse(selected[32, 70])

    def test_thick_core_removes_long_thin_appendage(self):
        mask = np.zeros((120, 160), dtype=np.uint8)
        cv2.circle(mask, (55, 55), 30, 1, -1)
        mask[53:58, 84:150] = 1

        rigid, report = thick_core_region(
            mask,
            erosion_radius=6,
            restore_radius=12,
            minimum_core_pixels=20,
        )

        self.assertTrue(rigid[55, 55])
        self.assertFalse(rigid[55, 140])
        self.assertGreater(rigid.sum(), 0.85 * (mask[:, :85] > 0).sum())
        self.assertEqual(report["status"], "ok")

    def test_thin_fragment_becomes_unavailable(self):
        mask = np.zeros((40, 80), dtype=bool)
        mask[19:22, 5:75] = True

        rigid, report = thick_core_region(
            mask,
            erosion_radius=3,
            restore_radius=6,
        )

        self.assertFalse(np.any(rigid))
        self.assertEqual(report["status"], "no_rigid_core")

    def test_halfspace_uses_face_centroids(self):
        vertices = np.array([
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        faces = np.array([[0, 1, 2], [3, 4, 5]])

        keep = halfspace_face_mask(
            vertices, faces, axis="x", minimum=0.0
        )

        np.testing.assert_array_equal(keep, [False, True])


if __name__ == "__main__":
    unittest.main()
