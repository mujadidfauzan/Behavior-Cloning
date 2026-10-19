"""Check that rollout traces distinguish close commands from actual motion."""

import csv
import io
import unittest

import numpy as np

from eval_trace import EpisodeTrace, STEP_FIELDS


def observation(eef, cube, gripper=(0.04, -0.04)):
    return {
        "robot0_eef_pos": np.asarray(eef, dtype=float),
        "cube_pos": np.asarray(cube, dtype=float),
        "robot0_gripper_qpos": np.asarray(gripper, dtype=float),
    }


class EpisodeTraceTests(unittest.TestCase):
    def test_close_hold_and_lift_are_distinct_events(self):
        cube = (0.0, 0.0, 0.82)
        states = [
            observation((0.02, 0.0, 0.90), cube),
            observation((0.004, 0.003, 0.825), cube),
            observation((0.004, 0.003, 0.825), cube, (0.02, -0.02)),
            observation((0.004, 0.003, 0.825), cube, (0.01, -0.01)),
            observation((0.004, 0.003, 0.829), cube, (0.01, -0.01)),
            observation((0.004, 0.003, 0.836), (0.0, 0.0, 0.832), (0.01, -0.01)),
        ]
        actions = [
            (0, 0, 0, 0, 0, 0, -1),
            (0, 0, 0, 0, 0, 0, 1),
            (0, 0, 0, 0, 0, 0, 1),
            (0, 0, 0.2, 0, 0, 0, 1),
            (0, 0, 0.2, 0, 0, 0, 1),
        ]
        trace = EpisodeTrace(initial_cube_z=0.82, upward_action_threshold=0.01)
        rows = [
            trace.record(1000, 1, 1, step, 20, states[step - 1], actions[step - 1],
                         states[step], gripper_logit=0.5,
                         grasp_contact_after=step >= 3)
            for step in range(1, 6)
        ]

        self.assertEqual(set(rows[0]), set(STEP_FIELDS))
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=STEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(output.getvalue())))), 5)

        self.assertEqual(rows[1]["close_command_edge"], 1)
        self.assertAlmostEqual(rows[1]["eef_cube_xy_before_m"], 0.005)
        self.assertAlmostEqual(rows[1]["eef_cube_dz_before_m"], 0.005)
        self.assertAlmostEqual(rows[1]["gripper_joint_abs_sum_before"], 0.08)
        self.assertAlmostEqual(rows[1]["gripper_joint_abs_sum_after"], 0.04)
        self.assertEqual(rows[2]["closed_command_run_steps"], 2)
        self.assertEqual(rows[2]["grasp_contact_after"], 1)
        self.assertEqual(rows[3]["first_upward_command_after_close"], 1)
        self.assertEqual(rows[4]["first_eef_rise_after_close"], 1)
        self.assertEqual(rows[4]["first_cube_rise"], 1)
        self.assertAlmostEqual(rows[4]["cube_lift_m"], 0.012)

        summary = trace.summary()
        self.assertEqual(summary["first_close_step"], 2)
        self.assertEqual(summary["first_upward_command_after_close_step"], 4)
        self.assertEqual(summary["closed_steps_before_first_upward"], 2)
        self.assertEqual(summary["first_close_to_upward_command_steps"], 2)
        self.assertEqual(summary["first_eef_rise_after_close_step"], 5)
        self.assertEqual(summary["first_close_to_eef_rise_steps"], 3)
        self.assertEqual(summary["first_cube_rise_step"], 5)
        self.assertEqual(summary["first_close_to_cube_rise_steps"], 3)
        self.assertEqual(summary["first_grasp_contact_step"], 3)
        self.assertEqual(summary["first_close_to_grasp_contact_steps"], 1)
        self.assertEqual(summary["grasp_contact_steps"], 3)
        self.assertEqual(summary["first_upward_command_to_cube_rise_steps"], 1)
        self.assertEqual(summary["close_command_transitions"], 1)
        self.assertEqual(summary["max_closed_command_run_steps"], 4)

    def test_no_close_does_not_report_grasp_timing(self):
        trace = EpisodeTrace(initial_cube_z=0.82, upward_action_threshold=0.01)
        before = observation((0, 0, 0.9), (0, 0, 0.82))
        after = observation((0, 0, 0.92), (0, 0, 0.82))
        row = trace.record(1000, 1, 1, 1, 20, before,
                           (0, 0, 0.2, 0, 0, 0, -1), after, -0.5)
        self.assertEqual(row["first_upward_command_after_close"], 0)
        self.assertEqual(row["first_eef_rise_after_close"], 0)
        self.assertEqual(trace.summary()["first_close_step"], "")
        self.assertEqual(trace.summary()["first_upward_command_after_close_step"], "")

    def test_reopening_and_reclosing_keeps_the_first_close_step(self):
        trace = EpisodeTrace(initial_cube_z=0.82, upward_action_threshold=0.01)
        state = observation((0, 0, 0.825), (0, 0, 0.82))
        edges = []
        for step, command in enumerate((1, 1, -1, 1), start=1):
            action = (0, 0, 0, 0, 0, 0, command)
            row = trace.record(1000, 1, 1, step, 20, state, action, state, 0)
            edges.append(row["close_command_edge"])
        self.assertEqual(edges, [1, 0, 0, 1])
        self.assertEqual(trace.summary()["first_close_step"], 1)
        self.assertEqual(trace.summary()["close_command_transitions"], 2)
        self.assertEqual(trace.summary()["max_closed_command_run_steps"], 2)


if __name__ == "__main__":
    unittest.main()
