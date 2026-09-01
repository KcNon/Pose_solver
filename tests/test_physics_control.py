from __future__ import annotations

import unittest

import numpy as np

from common.physics_control import (
    assembly_target_translation,
    dynamic_collision_approximation,
    elastic_tube_wrench,
    place_release_settings,
    physics_pose_refinement_settings,
    rigid_body_controller_parameters,
    score_physics_pose_candidate,
    select_control_profile,
    settled_contact_settings,
    sustained_contact_summary,
    transformed_bounds_minimum_z,
)


class PhysicsControlTests(unittest.TestCase):
    def test_physics_pose_refinement_defaults_are_bounded(self) -> None:
        settings = physics_pose_refinement_settings({})
        self.assertFalse(settings["enabled"])
        self.assertEqual(len(settings["candidates"]), 9)
        self.assertEqual(settings["apply_frame_range"], [0, 0])

    def test_elastic_tube_leaves_axial_translation_and_yaw_unconstrained(self) -> None:
        settings = physics_pose_refinement_settings({})["tube"]
        settings["mount_point_part_m"] = [0.01, 0.0, 0.0]
        settings["direction_part"] = [0.0, 0.0, -1.0]
        wrench = elastic_tube_wrench(
            position_world=np.zeros(3),
            rotation_world_from_part=np.eye(3),
            linear_velocity_world=np.asarray([0.0, 0.0, 0.2]),
            angular_velocity_world=np.asarray([0.0, 0.0, 1.0]),
            body_axis_origin_world=np.zeros(3),
            body_axis_world=np.asarray([0.0, 0.0, 1.0]),
            tube=settings,
        )
        self.assertLess(wrench["force_world_n"][0], 0.0)
        self.assertAlmostEqual(wrench["force_world_n"][2], 0.0)
        np.testing.assert_allclose(wrench["torque_world_nm"], 0.0, atol=1e-12)

    def test_physics_candidate_score_rejects_visually_large_correction(self) -> None:
        settings = physics_pose_refinement_settings({})
        result = score_physics_pose_candidate(
            {
                "translation_error_m": 0.02,
                "tilt_error_deg": 1.0,
                "final_linear_speed_mps": 0.001,
                "final_angular_speed_radps": 0.01,
                "maximum_contact_penetration_m": 0.0001,
                "contact_observed": True,
                "final_tube_energy_j": 0.0,
            },
            settings,
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["gates"]["visual_translation"])

    def test_place_release_settings_have_bounded_default_trials(self) -> None:
        settings = place_release_settings({})
        self.assertEqual(settings["initial_height_m"], 0.003)
        self.assertEqual(settings["settle_seconds"], 5.0)
        self.assertEqual(len(settings["trials"]), 5)
        self.assertEqual(settings["trials"][0]["name"], "centered")

    def test_place_release_settings_validate_contact_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "contact window"):
            place_release_settings(
                {
                    "place_release": {
                        "settle_seconds": 1.0,
                        "contact_window_seconds": 2.0,
                    }
                }
            )

    def test_assembly_target_translation_ramps_in_reference_frame(self) -> None:
        rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        simulation = {
            "assembly_target_corrections": {
                "nozzle": {
                    "translation_reference_m": [0.0, 0.01, 0.0],
                    "ramp_frame_range": [10, 20],
                    "source": "contact_calibration",
                }
            }
        }
        before = assembly_target_translation(
            simulation, part="nozzle", frame_id=9, reference_rotation=rotation
        )
        middle = assembly_target_translation(
            simulation, part="nozzle", frame_id=15, reference_rotation=rotation
        )
        after = assembly_target_translation(
            simulation, part="nozzle", frame_id=20, reference_rotation=rotation
        )
        np.testing.assert_allclose(before["translation_world_m"], [0, 0, 0])
        np.testing.assert_allclose(
            middle["translation_world_m"], [0, 0, 0.005]
        )
        np.testing.assert_allclose(
            after["translation_world_m"], [0, 0, 0.01]
        )
        self.assertEqual(after["source"], "contact_calibration")

    def test_sustained_contact_requires_recent_dense_contact(self) -> None:
        settings = {
            "contact_window_seconds": 0.5,
            "minimum_contact_fraction": 0.5,
            "maximum_contact_gap_seconds": 0.1,
        }
        passed = sustained_contact_summary(
            [False, True] * 5,
            physics_dt=0.05,
            settings=settings,
        )
        failed = sustained_contact_summary(
            [True] * 8 + [False] * 2,
            physics_dt=0.1,
            settings=settings,
        )
        self.assertTrue(passed["sustained"])
        self.assertFalse(failed["sustained"])
        self.assertTrue(failed["observed"])
        self.assertAlmostEqual(failed["longest_gap_seconds"], 0.2)

    def test_transformed_bounds_minimum_z_uses_all_corners(self) -> None:
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 0.4]
        transform[:3, :3] = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        self.assertAlmostEqual(
            transformed_bounds_minimum_z(
                [[-1.0, -2.0, -3.0], [1.0, 4.0, 5.0]],
                transform,
            ),
            -1.6,
        )

    def test_dynamic_collision_approximation_is_configurable(self) -> None:
        self.assertEqual(dynamic_collision_approximation({}), "convexDecomposition")
        self.assertEqual(
            dynamic_collision_approximation(
                {"dynamic_collision_approximation": "sdf"}
            ),
            "sdf",
        )
        with self.assertRaisesRegex(ValueError, "dynamic_collision_approximation"):
            dynamic_collision_approximation(
                {"dynamic_collision_approximation": "none"}
            )

    def test_controller_parameters_are_derived_from_mass_and_extents(self) -> None:
        parameters = rigid_body_controller_parameters(
            {"mass_kg": 0.5, "canonical_extents_m": [0.2, 0.1, 0.3]}
        )
        self.assertAlmostEqual(parameters["mass_kg"], 0.5)
        self.assertAlmostEqual(parameters["inertia_scale"], 0.5 * 0.14 / 18.0)
        self.assertAlmostEqual(parameters["force_limit_n"], 12.0)
        self.assertGreater(parameters["torque_limit_nm"], 0.05)

    def test_controller_parameter_overrides_are_supported(self) -> None:
        parameters = rigid_body_controller_parameters(
            {"mass_kg": 1.0, "canonical_extents_m": [0.1, 0.2, 0.3]},
            {"force_limit_n": 9.0, "torque_limit_nm": 0.2},
        )
        self.assertEqual(parameters["force_limit_n"], 9.0)
        self.assertEqual(parameters["torque_limit_nm"], 0.2)

    def test_controller_parameters_reject_invalid_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical_extents_m"):
            rigid_body_controller_parameters(
                {"mass_kg": 1.0, "canonical_extents_m": [0.1, 0.2, 0.0]}
            )

    def test_tracking_is_used_before_contact_is_latched(self) -> None:
        settings = settled_contact_settings(
            {
                "settled_contact_control": {
                    "enabled": True,
                    "states": ["static"],
                    "frequency_radps": 8.0,
                    "damping_ratio": 2.0,
                }
            }
        )
        profile = select_control_profile(
            state="static",
            contact_latched=False,
            position_error_m=0.004,
            tracking_frequency_radps=16.0,
            settled_settings=settings,
        )
        self.assertEqual(profile["mode"], "tracking")
        self.assertEqual(profile["frequency_radps"], 16.0)
        self.assertEqual(profile["damping_ratio"], 1.0)

    def test_latched_static_contact_uses_compliant_profile(self) -> None:
        settings = settled_contact_settings(
            {
                "settled_contact_control": {
                    "enabled": True,
                    "states": ["static"],
                    "frequency_radps": 7.0,
                    "damping_ratio": 2.5,
                }
            }
        )
        profile = select_control_profile(
            state="static",
            contact_latched=True,
            position_error_m=0.004,
            tracking_frequency_radps=16.0,
            settled_settings=settings,
        )
        self.assertEqual(profile["mode"], "settled_contact")
        self.assertEqual(profile["frequency_radps"], 7.0)
        self.assertEqual(profile["damping_ratio"], 2.5)

    def test_non_static_state_does_not_use_latched_profile(self) -> None:
        settings = settled_contact_settings(
            {"settled_contact_control": {"enabled": True}}
        )
        profile = select_control_profile(
            state="moving",
            contact_latched=True,
            position_error_m=0.004,
            tracking_frequency_radps=16.0,
            settled_settings=settings,
        )
        self.assertEqual(profile["mode"], "tracking")

    def test_large_pose_error_does_not_use_compliant_profile(self) -> None:
        settings = settled_contact_settings(
            {
                "settled_contact_control": {
                    "enabled": True,
                    "maximum_position_error_m": 0.01,
                }
            }
        )
        profile = select_control_profile(
            state="static",
            contact_latched=True,
            position_error_m=0.06,
            tracking_frequency_radps=16.0,
            settled_settings=settings,
        )
        self.assertEqual(profile["mode"], "tracking")

    def test_invalid_offsets_are_rejected_by_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "frequency_radps"):
            settled_contact_settings(
                {
                    "settled_contact_control": {
                        "enabled": True,
                        "frequency_radps": 0.0,
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
