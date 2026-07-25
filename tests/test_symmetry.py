from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common.pose_transforms import axis_rotation
from common.symmetry import (
    SymmetrySpec,
    axis_direction_error_deg,
    resolve_symmetric_pose,
    symmetry_candidates,
    symmetry_rotation_distance_deg,
    symmetry_spec_from_state,
)


def _pose(euler_xyz_deg: tuple[float, float, float]) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_euler(
        "xyz", euler_xyz_deg, degrees=True
    ).as_matrix()
    value[:3, 3] = [0.2, -0.1, 0.4]
    return value


class SymmetryTests(unittest.TestCase):
    def test_continuous_axial_resolution_is_analytic(self) -> None:
        reference = _pose((17.0, -23.0, 41.0))
        measured = reference @ axis_rotation([0.0, 1.0, 0.0], np.deg2rad(137.3))
        symmetry = SymmetrySpec(
            axis_raw=(0.0, 1.0, 0.0),
            equivalence="continuous_axial",
        )

        result = resolve_symmetric_pose(measured, reference, symmetry)

        np.testing.assert_allclose(result.pose, reference, atol=1e-10)
        np.testing.assert_allclose(
            result.pose[:3, 3], measured[:3, 3], atol=0.0
        )
        self.assertLess(result.continuity_error_deg, 1e-8)

    def test_cyclic_symmetry_selects_nearest_equivalent_pose(self) -> None:
        reference = np.eye(4, dtype=np.float64)
        measured = axis_rotation([0.0, 0.0, 1.0], np.deg2rad(100.0))
        symmetry = SymmetrySpec(
            axis_raw=(0.0, 0.0, 1.0),
            equivalence="cyclic",
            discrete_order=4,
        )

        result = resolve_symmetric_pose(measured, reference, symmetry)

        self.assertAlmostEqual(result.continuity_error_deg, 10.0, places=8)
        self.assertAlmostEqual(result.axial_angle_deg, 270.0)
        self.assertEqual(len(symmetry_candidates(symmetry)), 4)

    def test_axis_flip_is_not_physical_equivalence(self) -> None:
        reference = np.eye(4, dtype=np.float64)
        symmetry = SymmetrySpec(
            axis_raw=(0.0, 1.0, 0.0),
            equivalence="none",
            observation_ambiguities=("axis_flip",),
        )
        flipped = next(
            candidate["local_transform"]
            for candidate in symmetry_candidates(symmetry)
            if candidate["axis_flipped"]
        )

        physical_only = resolve_symmetric_pose(
            flipped,
            reference,
            symmetry,
            include_observation_ambiguities=False,
        )
        ambiguity_resolved = resolve_symmetric_pose(
            flipped,
            reference,
            symmetry,
            include_observation_ambiguities=True,
        )

        self.assertAlmostEqual(physical_only.continuity_error_deg, 180.0)
        self.assertFalse(physical_only.axis_flipped)
        self.assertLess(ambiguity_resolved.continuity_error_deg, 1e-8)
        self.assertTrue(ambiguity_resolved.axis_flipped)
        self.assertAlmostEqual(
            symmetry_rotation_distance_deg(flipped, reference, symmetry),
            180.0,
        )

    def test_axial_equivalence_does_not_hide_axis_flip(self) -> None:
        reference = np.eye(4, dtype=np.float64)
        symmetry = SymmetrySpec(
            axis_raw=(0.0, 1.0, 0.0),
            equivalence="continuous_axial",
            observation_ambiguities=("axis_flip",),
            candidate_step_deg=360.0,
        )
        candidates = symmetry_candidates(symmetry)
        flipped = next(
            candidate["local_transform"]
            for candidate in candidates
            if candidate["axis_flipped"]
        )

        self.assertEqual(len(candidates), 2)
        self.assertAlmostEqual(
            symmetry_rotation_distance_deg(reference, flipped, symmetry),
            180.0,
        )

    def test_axis_metric_ignores_rotation_around_declared_axis(self) -> None:
        reference = _pose((5.0, 20.0, -15.0))
        spun = reference @ axis_rotation(
            [0.0, 1.0, 0.0], np.deg2rad(121.0)
        )
        self.assertLess(
            axis_direction_error_deg(
                reference, spun, np.asarray([0.0, 1.0, 0.0])
            ),
            1e-8,
        )

    def test_new_and_legacy_config_schemas_are_supported(self) -> None:
        current = symmetry_spec_from_state(
            {
                "symmetry": {
                    "axis_raw": [0.0, 2.0, 0.0],
                    "equivalence": "cyclic",
                    "discrete_order": 6,
                    "observation_ambiguities": ["axis_flip"],
                }
            }
        )
        legacy = symmetry_spec_from_state(
            {
                "appearance": {
                    "symmetry_axis_raw": [0.0, 1.0, 0.0],
                    "candidate_mode": "axis_flip",
                }
            }
        )

        self.assertEqual(current.axis_raw, (0.0, 1.0, 0.0))
        self.assertEqual(current.discrete_order, 6)
        self.assertEqual(legacy.equivalence, "none")
        self.assertEqual(legacy.observation_ambiguities, ("axis_flip",))

    def test_invalid_symmetry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SymmetrySpec(equivalence="continuous_axial")
        with self.assertRaises(ValueError):
            SymmetrySpec(
                axis_raw=(0.0, 1.0, 0.0),
                equivalence="cyclic",
                discrete_order=1,
            )


if __name__ == "__main__":
    unittest.main()
