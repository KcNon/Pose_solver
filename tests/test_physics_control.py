from __future__ import annotations

import unittest

import numpy as np

from common.physics_control import (
    dynamic_collision_approximation,
    rigid_body_controller_parameters,
    select_control_profile,
    settled_contact_settings,
    transformed_bounds_minimum_z,
)


class PhysicsControlTests(unittest.TestCase):
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
