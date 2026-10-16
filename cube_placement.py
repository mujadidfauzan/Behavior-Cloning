"""Cube placement shared by demonstration collection and the scripted viewer."""

from itertools import combinations

import numpy as np
from robosuite.utils import RandomizationError
from robosuite.utils.placement_samplers import UniformRandomSampler


CUBE_X_RANGE = (-0.10, 0.10)
CUBE_Y_RANGE = (-0.20, 0.20)
MIN_CUBE_DISTANCE = 0.10
DEFAULT_OBJECT_YAW_RANGE_DEG = 45.0


def object_rotation_range(random_object_rotation, yaw_range_deg):
    """Return fixed yaw or a symmetric yaw interval in radians."""
    if not np.isfinite(yaw_range_deg) or not 0 < yaw_range_deg <= 180:
        raise ValueError("object yaw range must be greater than 0 and at most 180 degrees")
    if not random_object_rotation:
        return 0.0
    limit = float(np.deg2rad(yaw_range_deg))
    return (-limit, limit)


class MinimumDistanceSampler(UniformRandomSampler):
    """Reject layouts whose object centers are too close in the XY plane."""

    def __init__(self, *args, minimum_distance, max_attempts=300, **kwargs):
        if minimum_distance < 0:
            raise ValueError("minimum_distance must be non-negative")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.minimum_distance = float(minimum_distance)
        self.max_attempts = max_attempts
        super().__init__(*args, **kwargs)

    def sample(self, fixtures=None, reference=None, on_top=True):
        for _ in range(self.max_attempts):
            placements = super().sample(
                fixtures=fixtures, reference=reference, on_top=on_top
            )
            positions = [
                np.asarray(position)[:2] for position, _, _ in placements.values()
            ]
            if all(
                np.linalg.norm(first - second) >= self.minimum_distance
                for first, second in combinations(positions, 2)
            ):
                return placements

        raise RandomizationError(
            f"Could not place cubes at least {self.minimum_distance:.3f} m apart "
            f"after {self.max_attempts} attempts"
        )
