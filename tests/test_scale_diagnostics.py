import unittest

import numpy as np
import trimesh

from common.scale_diagnostics import (
    pareto_indices,
    select_anchor_scale,
    select_visual_gated_candidate,
)
from common.silhouette_scale_calibration import (
    _mesh_bottom_gap,
    _preserve_support_contact,
    select_scale_candidate_index,
)


class ScaleDiagnosticsTests(unittest.TestCase):
    def test_scale_candidate_prefers_physical_area_inside_visual_tie(self):
        rows = [
            {
                "scale_factor": 1.00,
                "optimize_loss": 0.44,
                "optimize_rendered_to_target_median": 0.88,
            },
            {
                "scale_factor": 1.10,
                "optimize_loss": 0.405,
                "optimize_rendered_to_target_median": 0.99,
            },
            {
                "scale_factor": 1.15,
                "optimize_loss": 0.400,
                "optimize_rendered_to_target_median": 1.16,
            },
        ]

        selected, report = select_scale_candidate_index(
            rows, visual_loss_tie_tolerance=0.01
        )

        self.assertEqual(selected, 1)
        self.assertEqual(report["visually_equivalent_indices"], [1, 2])

    def test_scale_candidate_uses_visual_optimum_outside_tie(self):
        rows = [
            {
                "scale_factor": 1.00,
                "optimize_loss": 0.50,
                "optimize_rendered_to_target_median": 1.00,
            },
            {
                "scale_factor": 1.10,
                "optimize_loss": 0.40,
                "optimize_rendered_to_target_median": 1.10,
            },
        ]

        selected, _ = select_scale_candidate_index(
            rows, visual_loss_tie_tolerance=0.01
        )

        self.assertEqual(selected, 1)

    def test_scale_change_preserves_support_plane_contact(self):
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 2.0])
        plane = {
            "normal_world": [0.0, 0.0, 1.0],
            "point_world": [0.0, 0.0, 0.0],
        }
        baseline = np.eye(4)
        baseline[:3, :3] *= 0.1
        baseline[2, 3] = 0.1
        target_gap = _mesh_bottom_gap(mesh, baseline, plane)
        candidate = baseline.copy()
        candidate[:3, :3] *= 2.0

        adjusted = _preserve_support_contact(
            mesh, candidate, plane, target_gap
        )

        self.assertAlmostEqual(
            _mesh_bottom_gap(mesh, adjusted, plane), target_gap
        )
        np.testing.assert_allclose(adjusted[:2, 3], baseline[:2, 3])

    def test_anchor_scale_uses_residual_weighted_median(self):
        fits = [
            {"scale": 3.12, "fit_rmse_m": 0.038},
            {"scale": 0.37, "fit_rmse_m": 0.0076},
            {"scale": 2.39, "fit_rmse_m": 0.025},
            {"scale": 0.93, "fit_rmse_m": 0.017},
        ]
        scale, report = select_anchor_scale(fits)
        self.assertEqual(scale, 0.37)
        self.assertEqual(report["selected_fit_index"], 1)
        self.assertGreater(report["scale_ratio_max_to_min"], 8.0)

    def test_anchor_scale_rejects_non_finite_candidates(self):
        scale, report = select_anchor_scale([
            {"scale": float("nan"), "fit_rmse_m": 0.01},
            {"scale": 0.5, "fit_rmse_m": 0.02},
        ])
        self.assertEqual(scale, 0.5)
        self.assertEqual(report["candidate_count"], 1)

    def test_pareto_front_excludes_dominated_candidates(self):
        rows = [
            {"visual_loss": 0.10, "max_penetration_m": 0.010},
            {"visual_loss": 0.11, "max_penetration_m": 0.005},
            {"visual_loss": 0.12, "max_penetration_m": 0.012},
        ]
        self.assertEqual(
            pareto_indices(rows, ("visual_loss", "max_penetration_m")),
            [0, 1],
        )

    def test_visual_gate_prefers_lowest_penetration(self):
        rows = [
            {
                "scale_factor": 0.96,
                "visual_loss": 0.125,
                "max_penetration_m": 0.001,
            },
            {
                "scale_factor": 1.0,
                "visual_loss": 0.100,
                "max_penetration_m": 0.010,
            },
            {
                "scale_factor": 1.04,
                "visual_loss": 0.140,
                "max_penetration_m": 0.015,
            },
        ]
        selected, report = select_visual_gated_candidate(
            rows, maximum_visual_loss_degradation=0.03
        )
        self.assertEqual(selected, 0)
        self.assertEqual(report["eligible_indices"], [0, 1])
        self.assertFalse(report["trajectory_mutated"])


if __name__ == "__main__":
    unittest.main()
