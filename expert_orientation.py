"""Orientation targets and delta commands for the OSC_POSE scripted expert."""

import numpy as np
from robosuite.utils import transform_utils as T
from robosuite.utils.control_utils import orientation_error


ORIENTATION_TOLERANCE = np.deg2rad(3.0)


def grasp_orientation(obs):
    """Return a grasp rotation matrix from the EEF/cube xyzw quaternions."""
    eef_euler = T.mat2euler(T.quat2mat(obs["robot0_eef_quat"]))
    cube_euler = T.mat2euler(T.quat2mat(obs["cube_quat"]))
    return T.euler2mat(np.array([eef_euler[0], eef_euler[1], cube_euler[2]]))


def orientation_distance(obs, target):
    """Measure the remaining rotation angle in radians."""
    if target is None:
        return 0.0
    current = T.quat2mat(obs["robot0_eef_quat"])
    cosine = (np.trace(target @ current.T) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def orientation_command(obs, target):
    """Return a bounded world-frame delta rotation command for OSC_POSE."""
    if target is None:
        return np.zeros(3)
    current = T.quat2mat(obs["robot0_eef_quat"])
    return np.clip(2.0 * orientation_error(target, current), -0.5, 0.5)
