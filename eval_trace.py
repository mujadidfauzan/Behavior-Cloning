"""Per-step BC diagnostics, separate from the policy inputs.

"before" is the observation used to choose an action; "after" is the result of
env.step(action). Gripper qpos values are raw finger joint positions. The
optional grasp-contact flag comes from the environment's two-finger grasp check.
An upward command means positive controller Z above the chosen threshold; the
separate EEF and cube rise events require 1 cm of actual motion.
"""

import json

import numpy as np


RISE_THRESHOLD_M = 0.01

STEP_FIELDS = (
    "seed", "episode", "global_episode", "step", "time_s",
    "eef_x_before_m", "eef_y_before_m", "eef_z_before_m",
    "cube_x_before_m", "cube_y_before_m", "cube_z_before_m",
    "eef_x_after_m", "eef_y_after_m", "eef_z_after_m",
    "cube_x_after_m", "cube_y_after_m", "cube_z_after_m",
    "eef_cube_dx_before_m", "eef_cube_dy_before_m", "eef_cube_dz_before_m",
    "eef_cube_xy_before_m", "eef_cube_distance_before_m",
    "eef_cube_dx_after_m", "eef_cube_dy_after_m", "eef_cube_dz_after_m",
    "eef_cube_xy_after_m", "eef_cube_distance_after_m",
    "action_x", "action_y", "action_z",
    "action_rx", "action_ry", "action_rz",
    "gripper_command", "gripper_logit",
    "gripper_qpos_before", "gripper_qpos_after",
    "gripper_joint_abs_sum_before", "gripper_joint_abs_sum_after",
    "grasp_contact_after",
    "eef_dx_m", "eef_dy_m", "eef_dz_m", "eef_displacement_m",
    "cube_dz_m", "cube_lift_m",
    "close_command_edge", "closed_command_run_steps",
    "first_upward_command_after_close", "first_eef_rise_after_close",
    "first_cube_rise",
)


def _xyz_fields(name, phase, xyz):
    return dict(zip(
        (f"{name}_{axis}_{phase}_m" for axis in "xyz"), map(float, xyz)
    ))


def _relative_fields(suffix, eef, cube):
    relative = eef - cube
    return {
        f"eef_cube_d{axis}_{suffix}_m": float(relative[index])
        for index, axis in enumerate("xyz")
    } | {
        f"eef_cube_xy_{suffix}_m": float(np.linalg.norm(relative[:2])),
        f"eef_cube_distance_{suffix}_m": float(np.linalg.norm(relative)),
    }


class EpisodeTrace:
    """Track event timing without storing an episode's frames in memory."""

    def __init__(self, initial_cube_z, upward_action_threshold):
        self.initial_cube_z = float(initial_cube_z)
        self.upward_action_threshold = float(upward_action_threshold)
        self.previous_close_command = False
        self.closed_command_run_steps = 0
        self.max_closed_command_run_steps = 0
        self.close_transitions = 0
        self.first_close_step = None
        self.first_close_xy_error_m = None
        self.first_close_z_error_m = None
        self.first_close_eef_z = None
        self.first_upward_command_after_close_step = None
        self.closed_steps_before_first_upward = None
        self.first_eef_rise_after_close_step = None
        self.first_cube_rise_step = None
        self.first_grasp_contact_step = None
        self.grasp_contact_steps = 0

    def record(self, seed, episode, global_episode, step, freq,
               obs_before, action, obs_after, gripper_logit,
               grasp_contact_after=None):
        """Describe one action and its result; step numbering starts at one."""
        eef_before = np.asarray(obs_before["robot0_eef_pos"], dtype=np.float64)
        cube_before = np.asarray(obs_before["cube_pos"], dtype=np.float64)
        eef_after = np.asarray(obs_after["robot0_eef_pos"], dtype=np.float64)
        cube_after = np.asarray(obs_after["cube_pos"], dtype=np.float64)
        qpos_before = np.asarray(obs_before["robot0_gripper_qpos"], dtype=np.float64).ravel()
        qpos_after = np.asarray(obs_after["robot0_gripper_qpos"], dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64)

        close_command = bool(action[6] > 0)
        closed_run_before_action = self.closed_command_run_steps
        close_command_edge = close_command and not self.previous_close_command
        self.previous_close_command = close_command
        if close_command:
            self.closed_command_run_steps += 1
            self.max_closed_command_run_steps = max(
                self.max_closed_command_run_steps, self.closed_command_run_steps
            )
        else:
            self.closed_command_run_steps = 0
        if close_command_edge:
            self.close_transitions += 1
            if self.first_close_step is None:
                self.first_close_step = step
                relative = eef_before - cube_before
                self.first_close_xy_error_m = float(np.linalg.norm(relative[:2]))
                self.first_close_z_error_m = float(relative[2])
                self.first_close_eef_z = float(eef_before[2])

        first_upward = (
            self.first_close_step is not None
            and self.first_upward_command_after_close_step is None
            and action[2] > self.upward_action_threshold
        )
        if first_upward:
            self.first_upward_command_after_close_step = step
            # Count completed close-command steps before this upward command.
            self.closed_steps_before_first_upward = closed_run_before_action

        first_eef_rise = (
            self.first_close_step is not None
            and self.first_eef_rise_after_close_step is None
            and eef_after[2] - self.first_close_eef_z >= RISE_THRESHOLD_M
        )
        if first_eef_rise:
            self.first_eef_rise_after_close_step = step
        cube_lift = float(cube_after[2] - self.initial_cube_z)
        first_cube_rise = (
            self.first_cube_rise_step is None and cube_lift >= RISE_THRESHOLD_M
        )
        if first_cube_rise:
            self.first_cube_rise_step = step
        if grasp_contact_after:
            self.grasp_contact_steps += 1
            if self.first_grasp_contact_step is None:
                self.first_grasp_contact_step = step

        eef_delta = eef_after - eef_before
        row = {
            "seed": seed, "episode": episode, "global_episode": global_episode,
            "step": step, "time_s": step / freq,
            "action_x": float(action[0]), "action_y": float(action[1]),
            "action_z": float(action[2]),
            "action_rx": float(action[3]), "action_ry": float(action[4]),
            "action_rz": float(action[5]),
            "gripper_command": float(action[6]),
            "gripper_logit": float(gripper_logit),
            "gripper_qpos_before": json.dumps(qpos_before.tolist()),
            "gripper_qpos_after": json.dumps(qpos_after.tolist()),
            # Panda finger joints have opposite signs; their signed sum cancels.
            "gripper_joint_abs_sum_before": float(np.abs(qpos_before).sum()),
            "gripper_joint_abs_sum_after": float(np.abs(qpos_after).sum()),
            "grasp_contact_after": (
                int(grasp_contact_after) if grasp_contact_after is not None else ""
            ),
            "eef_dx_m": float(eef_delta[0]),
            "eef_dy_m": float(eef_delta[1]),
            "eef_dz_m": float(eef_delta[2]),
            "eef_displacement_m": float(np.linalg.norm(eef_delta)),
            "cube_dz_m": float(cube_after[2] - cube_before[2]),
            "cube_lift_m": cube_lift,
            "close_command_edge": int(close_command_edge),
            "closed_command_run_steps": self.closed_command_run_steps,
            "first_upward_command_after_close": int(first_upward),
            "first_eef_rise_after_close": int(first_eef_rise),
            "first_cube_rise": int(first_cube_rise),
        }
        row.update(_xyz_fields("eef", "before", eef_before))
        row.update(_xyz_fields("cube", "before", cube_before))
        row.update(_xyz_fields("eef", "after", eef_after))
        row.update(_xyz_fields("cube", "after", cube_after))
        row.update(_relative_fields("before", eef_before, cube_before))
        row.update(_relative_fields("after", eef_after, cube_after))
        return row

    def summary(self):
        def delay(later):
            if self.first_close_step is None or later is None:
                return ""
            return later - self.first_close_step

        return {
            "first_close_step": self.first_close_step or "",
            "first_close_xy_error_m": self.first_close_xy_error_m
            if self.first_close_xy_error_m is not None else "",
            "first_close_z_error_m": self.first_close_z_error_m
            if self.first_close_z_error_m is not None else "",
            "first_upward_command_after_close_step":
                self.first_upward_command_after_close_step or "",
            "closed_steps_before_first_upward": self.closed_steps_before_first_upward
            if self.closed_steps_before_first_upward is not None else "",
            "first_close_to_upward_command_steps": delay(
                self.first_upward_command_after_close_step
            ),
            "first_eef_rise_after_close_step":
                self.first_eef_rise_after_close_step or "",
            "first_close_to_eef_rise_steps": delay(
                self.first_eef_rise_after_close_step
            ),
            "first_cube_rise_step": self.first_cube_rise_step or "",
            "first_close_to_cube_rise_steps": delay(self.first_cube_rise_step),
            "first_grasp_contact_step": self.first_grasp_contact_step or "",
            "first_close_to_grasp_contact_steps": delay(
                self.first_grasp_contact_step
            ),
            "grasp_contact_steps": self.grasp_contact_steps,
            "first_upward_command_to_cube_rise_steps": (
                self.first_cube_rise_step - self.first_upward_command_after_close_step
                if self.first_cube_rise_step is not None
                and self.first_upward_command_after_close_step is not None else ""
            ),
            "close_command_transitions": self.close_transitions,
            "max_closed_command_run_steps": self.max_closed_command_run_steps,
        }
