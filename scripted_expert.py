import argparse
import time

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask

from cube_placement import (
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    DEFAULT_OBJECT_YAW_RANGE_DEG,
    MIN_CUBE_DISTANCE,
    MinimumDistanceSampler,
    object_rotation_range,
)
from expert_orientation import (
    ORIENTATION_TOLERANCE,
    grasp_orientation,
    orientation_command,
    orientation_distance,
)

# ============================================================
# CONFIGURATION
# ============================================================

CONTROL_FREQ = 20
MAX_STEPS = 500
NUM_EPISODES = 50
SEED = 42

RENDER = True
REALTIME_RENDER = True
RENDERER = "mjviewer"
RENDER_CAMERA = None

# Automatically show viewer help if supported
AUTO_SHOW_VIEWER_HELP = True

JOINT_NOISE_DEG = 3.0
JOINT_NOISE_RAD = np.deg2rad(JOINT_NOISE_DEG)

GRASP_X_OFFSET = 0.0
GRASP_Y_OFFSET = 0.0
GRASP_Z_OFFSET = 0.005

APPROACH_HEIGHT = 0.12
ALIGN_HEIGHT = 0.08
GRASP_HOLD_STEPS = 15

LIFT_HEIGHT = 0.20
LIFT_TOLERANCE = 0.02
LIFT_HOLD_STEPS = 20
LIFT_EXTRA_CLEARANCE = 0.03

GAINS = {
    "approach": (8.0, 0.8),
    "align": (5.0, 0.4),
    "descend": (4.0, 0.3),
    "grasp": (2.0, 0.15),
    "lift": (6.0, 0.5),
}

np.random.seed(SEED)


# ============================================================
# CUSTOM THREE-CUBE ENVIRONMENT
# ============================================================


class ThreeCubeLift(Lift):

    def __init__(
        self, *args, random_object_rotation=False,
        object_yaw_range_deg=DEFAULT_OBJECT_YAW_RANGE_DEG,
        control_orientation=None, **kwargs,
    ):
        self.random_object_rotation = bool(random_object_rotation)
        self.control_orientation = (
            self.random_object_rotation if control_orientation is None
            else bool(control_orientation)
        )
        if self.control_orientation and not self.random_object_rotation:
            raise ValueError("Orientation control requires randomized object yaw")
        self.object_yaw_range_deg = float(object_yaw_range_deg)
        self.object_rotation = object_rotation_range(
            self.random_object_rotation, self.object_yaw_range_deg
        )
        super().__init__(*args, **kwargs)

    def _load_model(self):

        # Initialize robot model without Lift's single-cube setup
        super(Lift, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )

        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        # Red target cube
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1.0, 0.0, 0.0, 1.0],
            rng=self.rng,
        )

        # Blue distractor
        self.blue_cube = BoxObject(
            name="blue_cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[0.1, 0.2, 1.0, 1.0],
            rng=self.rng,
        )

        # Green distractor
        self.green_cube = BoxObject(
            name="green_cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[0.1, 0.9, 0.2, 1.0],
            rng=self.rng,
        )

        cubes = [
            self.cube,
            self.blue_cube,
            self.green_cube,
        ]

        self.placement_initializer = MinimumDistanceSampler(
            name="ThreeCubeSampler",
            mujoco_objects=cubes,
            x_range=CUBE_X_RANGE,
            y_range=CUBE_Y_RANGE,
            minimum_distance=MIN_CUBE_DISTANCE,
            rotation=self.object_rotation,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):

        super()._setup_references()

        self.blue_cube_body_id = self.sim.model.body_name2id(self.blue_cube.root_body)

        self.green_cube_body_id = self.sim.model.body_name2id(self.green_cube.root_body)

    def get_distractor_positions(self):

        blue_pos = self.sim.data.body_xpos[self.blue_cube_body_id].copy()

        green_pos = self.sim.data.body_xpos[self.green_cube_body_id].copy()

        return blue_pos, green_pos


# ============================================================
# CONTROLLER
# ============================================================

def create_env(
    random_object_rotation=False,
    object_yaw_range_deg=DEFAULT_OBJECT_YAW_RANGE_DEG,
    control_orientation=None,
):
    control_orientation = (
        bool(random_object_rotation) if control_orientation is None
        else bool(control_orientation)
    )
    if control_orientation and not random_object_rotation:
        raise ValueError("Orientation control requires randomized object yaw")
    controller = load_composite_controller_config(controller="BASIC", robot="Panda")
    right = controller["body_parts"]["right"]
    right["type"] = "OSC_POSE"
    right["input_type"] = "delta"
    right["impedance_mode"] = "fixed"
    if control_orientation:
        right["input_ref_frame"] = "world"
    controller["body_parts"] = {"right": right}

    environment = ThreeCubeLift(
        robots="Panda",
        controller_configs=controller,
        random_object_rotation=random_object_rotation,
        object_yaw_range_deg=object_yaw_range_deg,
        control_orientation=control_orientation,
        initialization_noise={"magnitude": JOINT_NOISE_RAD, "type": "uniform"},
        renderer=RENDERER,
        render_camera=RENDER_CAMERA,
        has_renderer=RENDER,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=CONTROL_FREQ,
        horizon=MAX_STEPS,
        ignore_done=True,
        hard_reset=False,
        reward_shaping=False,
    )
    if environment.action_dim != 7:
        raise RuntimeError(f"Expected 7D action, got {environment.action_dim}")
    return environment


env = None


# ============================================================
# AUTOMATIC VIEWER HELP
# ============================================================


def print_viewer_controls():

    print("\n" + "=" * 65)
    print("MUJOCO VIEWER CONTROLS")
    print("=" * 65)
    print("Left mouse drag   : Rotate camera")
    print("Right mouse drag  : Pan camera")
    print("Mouse wheel       : Zoom")
    print("Double-click      : Select body (viewer-dependent)")
    print("F1                : Help (viewer-dependent)")
    print("Space             : Pause (viewer-dependent)")
    print("Esc               : Close viewer (viewer-dependent)")
    print("=" * 65)


def enable_viewer_help(environment):
    """
    Attempts to show built-in viewer help.

    This is version-dependent. The viewer may wrap another
    viewer object, and the help flag may not be available.
    """

    if not RENDER or not AUTO_SHOW_VIEWER_HELP:
        return False

    viewer = getattr(environment, "viewer", None)

    if viewer is None:
        print("[Viewer] Viewer has not been initialized.")
        return False

    candidates = [
        viewer,
        getattr(viewer, "viewer", None),
        getattr(viewer, "_viewer", None),
    ]

    for candidate in candidates:

        if candidate is None:
            continue

        # Older or custom viewer implementations
        if hasattr(candidate, "_showhelp"):
            try:
                candidate._showhelp = True
                print("[Viewer] Help panel enabled.")
                return True
            except Exception:
                pass

        if hasattr(candidate, "showhelp"):
            try:
                candidate.showhelp = True
                print("[Viewer] Help panel enabled.")
                return True
            except Exception:
                pass

    print(
        "[Viewer] Automatic built-in help is not supported "
        "by the detected viewer interface."
    )
    print("[Viewer] Controls are printed in the terminal instead.")

    return False


# ============================================================
# HELPERS
# ============================================================


def get_target(cube, height):

    return cube + np.array(
        [
            GRASP_X_OFFSET,
            GRASP_Y_OFFSET,
            height,
        ]
    )


def move_toward(obs, target, gripper, state, target_orientation=None):

    eef = obs["robot0_eef_pos"]
    error = target - eef

    gain, limit = GAINS[state]

    action = np.zeros(env.action_dim)

    action[:3] = np.clip(
        error * gain,
        -limit,
        limit,
    )

    action[3:6] = orientation_command(obs, target_orientation)
    action[6] = gripper

    low, high = env.action_spec
    return np.clip(action, low, high)


# ============================================================
# SCRIPTED EXPERT
# ============================================================


def run_episode(episode_id):

    obs = env.reset()

    initial_cube_pos = obs["cube_pos"].copy()
    initial_joint_pos = obs["robot0_joint_pos"].copy()
    target_orientation = (
        grasp_orientation(obs) if env.control_orientation else None
    )

    initial_blue_pos, initial_green_pos = env.get_distractor_positions()

    initial_cube_z = float(initial_cube_pos[2])
    target_cube_z = initial_cube_z + LIFT_HEIGHT

    print(f"\nEpisode {episode_id:02d}")
    print("  Red:  ", np.round(initial_cube_pos, 3))
    print("  Blue: ", np.round(initial_blue_pos, 3))
    print("  Green:", np.round(initial_green_pos, 3))

    state = "approach"
    state_steps = 0
    lift_hold_counter = 0
    lift_target = None

    max_lift = 0.0
    actual_lift = 0.0

    start_time = time.perf_counter()

    for step in range(MAX_STEPS):

        cube = obs["cube_pos"].copy()
        eef = obs["robot0_eef_pos"].copy()
        orientation_ok = (
            orientation_distance(obs, target_orientation) < ORIENTATION_TOLERANCE
        )

        # ----------------------------------------------------
        # APPROACH
        # ----------------------------------------------------

        if state == "approach":

            target = get_target(cube, APPROACH_HEIGHT)
            grip = -1.0

            if np.linalg.norm(eef - target) < 0.03:
                state = "align"
                state_steps = 0

        # ----------------------------------------------------
        # ALIGN
        # ----------------------------------------------------

        elif state == "align":

            target = get_target(cube, ALIGN_HEIGHT)
            grip = -1.0

            xy_error = np.linalg.norm(eef[:2] - target[:2])
            z_error = abs(eef[2] - target[2])

            if xy_error < 0.012 and z_error < 0.020 and orientation_ok:
                state = "descend"
                state_steps = 0

        # ----------------------------------------------------
        # DESCEND
        # ----------------------------------------------------

        elif state == "descend":

            target = get_target(cube, GRASP_Z_OFFSET)
            grip = -1.0

            xy_error = np.linalg.norm(eef[:2] - target[:2])
            z_error = abs(eef[2] - target[2])

            if xy_error < 0.008 and z_error < 0.012 and orientation_ok:
                state = "grasp"
                state_steps = 0

        # ----------------------------------------------------
        # GRASP
        # ----------------------------------------------------

        elif state == "grasp":

            target = eef.copy()
            grip = 1.0

            if state_steps >= GRASP_HOLD_STEPS:

                state = "lift"
                state_steps = 0

                lift_target = eef.copy()
                lift_target[2] += LIFT_HEIGHT + LIFT_EXTRA_CLEARANCE

        # ----------------------------------------------------
        # LIFT
        # ----------------------------------------------------

        else:

            target = lift_target
            grip = 1.0

        # ----------------------------------------------------
        # EXECUTE ACTION
        # ----------------------------------------------------

        action = move_toward(
            obs=obs,
            target=target,
            gripper=grip,
            state=state,
            target_orientation=target_orientation,
        )

        obs, reward, done, info = env.step(action)

        if RENDER:

            env.render()

            if REALTIME_RENDER:
                time.sleep(1.0 / CONTROL_FREQ)

        state_steps += 1

        # ----------------------------------------------------
        # MEASURE LIFT
        # ----------------------------------------------------

        current_cube_z = float(obs["cube_pos"][2])

        actual_lift = current_cube_z - initial_cube_z

        max_lift = max(max_lift, actual_lift)

        if state == "lift":

            if actual_lift >= (LIFT_HEIGHT - LIFT_TOLERANCE):
                lift_hold_counter += 1
            else:
                lift_hold_counter = 0

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if (
            state == "lift"
            and lift_hold_counter >= LIFT_HOLD_STEPS
            and env._check_success()
        ):

            elapsed = time.perf_counter() - start_time

            blue_pos, green_pos = env.get_distractor_positions()

            blue_shift = np.linalg.norm(blue_pos - initial_blue_pos)

            green_shift = np.linalg.norm(green_pos - initial_green_pos)

            print(
                f"  SUCCESS | "
                f"Steps: {step + 1} | "
                f"Lift: {actual_lift:.3f}m | "
                f"Blue shift: {blue_shift:.3f}m | "
                f"Green shift: {green_shift:.3f}m"
            )

            return {
                "success": True,
                "steps": step + 1,
                "elapsed": elapsed,
                "lift": actual_lift,
                "max_lift": max_lift,
                "blue_shift": blue_shift,
                "green_shift": green_shift,
                "initial_cube": initial_cube_pos,
                "initial_joints": initial_joint_pos,
            }

    # --------------------------------------------------------
    # FAILED EPISODE
    # --------------------------------------------------------

    elapsed = time.perf_counter() - start_time

    blue_pos, green_pos = env.get_distractor_positions()

    blue_shift = np.linalg.norm(blue_pos - initial_blue_pos)

    green_shift = np.linalg.norm(green_pos - initial_green_pos)

    print(
        f"  FAILED | "
        f"Steps: {MAX_STEPS} | "
        f"State: {state} | "
        f"Max lift: {max_lift:.3f}m"
    )

    return {
        "success": False,
        "steps": MAX_STEPS,
        "elapsed": elapsed,
        "lift": actual_lift,
        "max_lift": max_lift,
        "blue_shift": blue_shift,
        "green_shift": green_shift,
        "initial_cube": initial_cube_pos,
        "initial_joints": initial_joint_pos,
    }


# ============================================================
# EVALUATION
# ============================================================


def evaluate():

    results = []

    for episode in range(1, NUM_EPISODES + 1):
        results.append(run_episode(episode))

    successes = [r for r in results if r["success"]]

    success_rate = 100.0 * len(successes) / len(results)

    avg_steps = np.mean([r["steps"] for r in successes]) if successes else float("nan")

    avg_lift = np.mean([r["lift"] for r in successes]) if successes else float("nan")

    avg_blue_shift = np.mean([r["blue_shift"] for r in results])

    avg_green_shift = np.mean([r["green_shift"] for r in results])

    print("\n" + "=" * 65)
    print("THREE-CUBE EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Total episodes:       {len(results)}")
    print(f"Successful:           {len(successes)}")
    print(f"Failed:               {len(results) - len(successes)}")
    print(f"Success rate:         {success_rate:.2f}%")
    print(f"Avg steps/success:    {avg_steps:.2f}")
    print(f"Avg sim time:         {avg_steps / CONTROL_FREQ:.2f}s")
    print(f"Avg red cube lift:    {avg_lift:.3f}m")
    print(f"Avg blue cube shift:  {avg_blue_shift:.3f}m")
    print(f"Avg green cube shift: {avg_green_shift:.3f}m")

    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():
    global env, RENDER, NUM_EPISODES
    parser = argparse.ArgumentParser(description="Evaluate the three-cube scripted expert")
    yaw_mode = parser.add_mutually_exclusive_group()
    yaw_mode.add_argument("--random-object-rotation", action="store_true",
                          help="Randomize object yaw and control gripper orientation")
    yaw_mode.add_argument("--random-object-yaw-only", action="store_true",
                          help="Randomize object yaw while keeping gripper orientation fixed")
    parser.add_argument("--object-yaw-range-deg", type=float,
                        default=DEFAULT_OBJECT_YAW_RANGE_DEG,
                        help="Symmetric object yaw range in degrees (default: +/-45)")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=RENDER)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    random_yaw = args.random_object_rotation or args.random_object_yaw_only
    try:
        object_rotation_range(random_yaw, args.object_yaw_range_deg)
    except ValueError as error:
        parser.error(str(error))

    RENDER = args.render
    NUM_EPISODES = args.episodes
    print(f"Random object yaw: {random_yaw} | Orientation actions: {args.random_object_rotation}")
    if random_yaw:
        print(f"Object yaw range: +/-{args.object_yaw_range_deg:g} degrees")
    env = create_env(random_yaw, args.object_yaw_range_deg,
                     control_orientation=args.random_object_rotation)

    try:

        # Initialize viewer before starting the evaluation.
        if RENDER:
            print_viewer_controls()
            env.reset()
            env.render()

            if AUTO_SHOW_VIEWER_HELP:
                enable_viewer_help(env)
                env.render()

        evaluate()

    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
