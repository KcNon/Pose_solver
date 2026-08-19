"""Pure helpers for selecting force-control profiles during contact."""
from __future__ import annotations

from typing import Any

import numpy as np


DYNAMIC_COLLISION_APPROXIMATIONS = {
    "convexDecomposition",
    "convexHull",
    "sdf",
    "sphereFill",
}


def dynamic_collision_approximation(simulation: dict[str, Any]) -> str:
    """Return a PhysX-compatible approximation for a dynamic mesh.

    Dense reconstructed meshes are frequently concave or hollow. SDF keeps
    those details, while the legacy convex-decomposition default can close a
    cavity and report a large false penetration. Explicit legacy configs can
    continue to request convex decomposition.
    """
    approximation = str(
        simulation.get(
            "dynamic_collision_approximation",
            "convexDecomposition",
        )
    )
    if approximation not in DYNAMIC_COLLISION_APPROXIMATIONS:
        raise ValueError(
            "simulation.dynamic_collision_approximation must be one of "
            f"{sorted(DYNAMIC_COLLISION_APPROXIMATIONS)}, got "
            f"{approximation!r}"
        )
    return approximation


def transformed_bounds_minimum_z(
    bounds: list[list[float]] | np.ndarray,
    transform: list[list[float]] | np.ndarray,
) -> float:
    """Return the world minimum Z of a transformed axis-aligned box."""
    bounds_array = np.asarray(bounds, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if bounds_array.shape != (2, 3) or matrix.shape != (4, 4):
        raise ValueError("bounds must be 2x3 and transform must be 4x4")
    corners = np.asarray(
        [
            [x, y, z]
            for x in bounds_array[:, 0]
            for y in bounds_array[:, 1]
            for z in bounds_array[:, 2]
        ],
        dtype=np.float64,
    )
    world = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return float(np.min(world[:, 2]))


def rigid_body_controller_parameters(
    part_info: dict[str, Any],
    override: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Derive bounded-force controller parameters from exported geometry.

    Isaac videos previously carried parameters keyed by the rice-cooker part
    names.  A dataset-level pipeline cannot know those names, so use the
    configured mass and canonical mesh extents to estimate a conservative
    scalar inertia.  Every value remains explicitly overrideable from the
    simulation config for a validated physical asset.
    """

    mass = float(part_info.get("mass_kg", 1.0))
    extents = np.asarray(part_info.get("canonical_extents_m", []), dtype=float)
    if mass <= 0.0:
        raise ValueError("part mass_kg must be positive")
    if extents.shape != (3,) or not np.isfinite(extents).all() or np.any(extents <= 0.0):
        raise ValueError("canonical_extents_m must contain three positive values")
    # Mean principal inertia of an axis-aligned box with these extents.
    inertia = mass * float(np.square(extents).sum()) / 18.0
    defaults = {
        "inertia_scale": max(inertia, 1e-5),
        "force_limit_n": max(5.0, 24.0 * mass),
        "torque_limit_nm": max(0.05, 120.0 * inertia),
    }
    configured = override or {}
    result = {
        key: float(configured.get(key, value))
        for key, value in defaults.items()
    }
    if any(value <= 0.0 for value in result.values()):
        raise ValueError("controller parameters must be positive")
    return {**result, "mass_kg": mass}


def settled_contact_settings(simulation: dict[str, Any]) -> dict[str, Any]:
    """Return validated settings for contact-aware settled-state control."""
    raw = simulation.get("settled_contact_control", {})
    enabled = bool(raw.get("enabled", False))
    states = tuple(str(value) for value in raw.get("states", ["static"]))
    frequency = float(raw.get("frequency_radps", 8.0))
    damping_ratio = float(raw.get("damping_ratio", 2.0))
    maximum_position_error = float(
        raw.get("maximum_position_error_m", 0.01)
    )
    if not states:
        raise ValueError("settled_contact_control.states must not be empty")
    if frequency <= 0.0:
        raise ValueError(
            "settled_contact_control.frequency_radps must be positive"
        )
    if damping_ratio <= 0.0:
        raise ValueError(
            "settled_contact_control.damping_ratio must be positive"
        )
    if maximum_position_error <= 0.0:
        raise ValueError(
            "settled_contact_control.maximum_position_error_m "
            "must be positive"
        )
    return {
        "enabled": enabled,
        "states": states,
        "frequency_radps": frequency,
        "damping_ratio": damping_ratio,
        "maximum_position_error_m": maximum_position_error,
    }


def select_control_profile(
    *,
    state: str,
    contact_latched: bool,
    position_error_m: float,
    tracking_frequency_radps: float,
    settled_settings: dict[str, Any],
) -> dict[str, Any]:
    """Select tracking or compliant settled-contact controller parameters."""
    if (
        settled_settings["enabled"]
        and contact_latched
        and state in settled_settings["states"]
        and position_error_m
        <= float(settled_settings["maximum_position_error_m"])
    ):
        return {
            "mode": "settled_contact",
            "frequency_radps": float(settled_settings["frequency_radps"]),
            "damping_ratio": float(settled_settings["damping_ratio"]),
        }
    return {
        "mode": "tracking",
        "frequency_radps": float(tracking_frequency_radps),
        "damping_ratio": 1.0,
    }
