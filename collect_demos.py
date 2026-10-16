import time
from pathlib import Path

import h5py
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

# Start with 10, then change to 100
NUM_DEMOS = 100
MAX_ATTEMPTS = 400

CONTROL_FREQ = 20
MAX_STEPS = 500

# Disable rendering for faster collection
RENDER = False
REALTIME_RENDER = False

# Camera observations
CAMERA_NAMES = ["agentview"]
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# Output
OUTPUT_DIR = Path("datasets/three_cube_lift")
# Each successful episode gets its own file. Existing files are preserved.

# Robot randomization
JOINT_NOISE_DEG = 3.0
JOINT_NOISE_RAD = np.deg2rad(JOINT_NOISE_DEG)

# Grasp calibration
GRASP_X_OFFSET = 0.0
GRASP_Y_OFFSET = 0.0
GRASP_Z_OFFSET = 0.005

APPROACH_HEIGHT = 0.12
ALIGN_HEIGHT = 0.08

GRASP_HOLD_STEPS = 15

# Actual cube lift requirement
LIFT_HEIGHT = 0.20
LIFT_TOLERANCE = 0.02
LIFT_HOLD_STEPS = 20

GAINS = {
    "approach": (8.0, 0.8),
    "align": (5.0, 0.4),
    "descend": (4.0, 0.3),
    "grasp": (2.0, 0.15),
    "lift": (6.0, 0.5),
}

np.random.seed(SEED)


# ============================================================
# THREE-CUBE ENVIRONMENT
# ============================================================


class ThreeCubeLift(Lift):

    def _load_model(self):

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

        self.cube = BoxObject(
            name="cube",
            size_min=[0.020] * 3,
            size_max=[0.022] * 3,
            rgba=[1.0, 0.0, 0.0, 1.0],
            rng=self.rng,
        )

        self.blue_cube = BoxObject(
            name="blue_cube",
            size_min=[0.020] * 3,
            size_max=[0.022] * 3,
            rgba=[0.1, 0.2, 1.0, 1.0],
            rng=self.rng,
        )

        self.green_cube = BoxObject(
            name="green_cube",
            size_min=[0.020] * 3,
            size_max=[0.022] * 3,
            rgba=[0.1, 0.9, 0.2, 1.0],
            rng=self.rng,
        )

        cubes = [
            self.cube,
            self.blue_cube,
            self.green_cube,
        ]

        sampler = SequentialCompositeSampler(name="ThreeCubeSampler")

        configs = [
            (
                "RedSampler",
                self.cube,
                [-0.03, 0.03],
                [-0.03, 0.03],
            ),
            (
                "BlueSampler",
                self.blue_cube,
                [-0.06, 0.06],
                [0.11, 0.17],
            ),
            (
                "GreenSampler",
                self.green_cube,
                [-0.06, 0.06],
                [-0.17, -0.11],
            ),
        ]

        for name, obj, x_range, y_range in configs:

            sampler.append_sampler(
                UniformRandomSampler(
                    name=name,
                    mujoco_objects=obj,
                    x_range=x_range,
                    y_range=y_range,
                    rotation=None,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                    rng=self.rng,
                )
            )

        self.placement_initializer = sampler

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

        return (
            self.sim.data.body_xpos[self.blue_cube_body_id].copy(),
            self.sim.data.body_xpos[self.green_cube_body_id].copy(),
        )


# ============================================================
# ENVIRONMENT CREATION
# ============================================================


def create_env():

    controller = load_composite_controller_config(
        controller="BASIC",
        robot="Panda",
    )

    right = controller["body_parts"]["right"]

    right["type"] = "OSC_POSE"
    right["input_type"] = "delta"
    right["impedance_mode"] = "fixed"

    controller["body_parts"] = {"right": right}

    env = ThreeCubeLift(
        robots="Panda",
        controller_configs=controller,
        initialization_noise={
            "magnitude": JOINT_NOISE_RAD,
            "type": "uniform",
        },
        has_renderer=RENDER,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=CAMERA_NAMES,
        camera_heights=IMAGE_HEIGHT,
        camera_widths=IMAGE_WIDTH,
        camera_depths=False,
        use_object_obs=True,
        control_freq=CONTROL_FREQ,
        horizon=MAX_STEPS,
        ignore_done=True,
        reward_shaping=False,
    )

    if env.action_dim != 7:
        raise RuntimeError(f"Expected 7D action, got {env.action_dim}")

    return env


# ============================================================
# SCRIPTED EXPERT
# ============================================================


def get_target(cube, height):

    return cube + np.array(
        [
            GRASP_X_OFFSET,
            GRASP_Y_OFFSET,
            height,
        ]
    )


def move_toward(env, obs, target, gripper, state):

    eef = obs["robot0_eef_pos"]
    error = target - eef

    gain, limit = GAINS[state]

    action = np.zeros(env.action_dim)

    action[:3] = np.clip(
        error * gain,
        -limit,
        limit,
    )

    action[3:6] = 0.0
    action[6] = gripper

    low, high = env.action_spec

    return np.clip(action, low, high)


# ============================================================
# TRAJECTORY RECORDING
# ============================================================


def record_observation(storage, obs):

    for key, value in obs.items():

        if isinstance(value, np.ndarray):
            storage.setdefault(key, []).append(value.copy())


def run_episode(env):

    obs = env.reset()

    initial_cube = obs["cube_pos"].copy()
    initial_joints = obs["robot0_joint_pos"].copy()

    initial_blue, initial_green = env.get_distractor_positions()

    print("\n" + "-" * 65, flush=True)
    print("INITIAL POSITIONS (after reset)", flush=True)
    print(f"  Red cube XYZ (m):   {np.round(initial_cube, 5)}", flush=True)
    print(f"  Blue cube XYZ (m):  {np.round(initial_blue, 5)}", flush=True)
    print(f"  Green cube XYZ (m): {np.round(initial_green, 5)}", flush=True)
    print(
        f"  Panda joints (deg): {np.round(np.rad2deg(initial_joints), 3)}", flush=True
    )
    print("-" * 65, flush=True)

    initial_cube_z = float(initial_cube[2])

    state = "approach"
    state_steps = 0

    lift_hold_counter = 0

    # Hold the end-effector/cube offset measured at lift start.
    grasp_offset_z = None

    observations = {}
    next_observations = {}

    actions = []
    rewards = []

    max_lift = 0.0
    success = False

    for step in range(MAX_STEPS):

        cube = obs["cube_pos"].copy()
        eef = obs["robot0_eef_pos"].copy()

        if state == "approach":

            target = get_target(cube, APPROACH_HEIGHT)
            grip = -1.0

            if np.linalg.norm(eef - target) < 0.03:
                state = "align"
                state_steps = 0

        elif state == "align":

            target = get_target(cube, ALIGN_HEIGHT)
            grip = -1.0

            xy_error = np.linalg.norm(eef[:2] - target[:2])
            z_error = abs(eef[2] - target[2])

            if xy_error < 0.012 and z_error < 0.020:
                state = "descend"
                state_steps = 0

        elif state == "descend":

            target = get_target(cube, GRASP_Z_OFFSET)
            grip = -1.0

            xy_error = np.linalg.norm(eef[:2] - target[:2])
            z_error = abs(eef[2] - target[2])

            if xy_error < 0.008 and z_error < 0.012:
                state = "grasp"
                state_steps = 0

        elif state == "grasp":

            target = eef.copy()
            grip = 1.0

            if state_steps >= GRASP_HOLD_STEPS:

                state = "lift"
                state_steps = 0

                grasp_offset_z = eef[2] - cube[2]

        else:

            grip = 1.0

        # In lift state, compute target based on the
        # desired actual cube height, not an arbitrary
        # fixed end-effector displacement.
        if state == "lift":

            target = eef.copy()

            target[2] = initial_cube_z + LIFT_HEIGHT + grasp_offset_z

            # Keep the red cube's horizontal position
            # stable during lifting.
            target[:2] = cube[:2] + np.array(
                [
                    GRASP_X_OFFSET,
                    GRASP_Y_OFFSET,
                ]
            )

        action = move_toward(
            env,
            obs,
            target,
            grip,
            state,
        )

        # Record the observation BEFORE the action.
        record_observation(observations, obs)

        next_obs, reward, done, info = env.step(action)

        # Record corresponding next observation.
        record_observation(
            next_observations,
            next_obs,
        )

        actions.append(action.copy())
        rewards.append(float(reward))

        obs = next_obs
        state_steps += 1

        if RENDER:
            env.render()

            if REALTIME_RENDER:
                time.sleep(1.0 / CONTROL_FREQ)

        actual_lift = float(obs["cube_pos"][2]) - initial_cube_z

        max_lift = max(max_lift, actual_lift)

        if state == "lift":

            if actual_lift >= (LIFT_HEIGHT - LIFT_TOLERANCE):
                lift_hold_counter += 1
            else:
                lift_hold_counter = 0

        if (
            state == "lift"
            and lift_hold_counter >= LIFT_HOLD_STEPS
            and env._check_success()
        ):
            success = True
            break

    final_blue, final_green = env.get_distractor_positions()

    return {
        "success": success,
        "steps": len(actions),
        "max_lift": max_lift,
        "final_lift": (float(obs["cube_pos"][2]) - initial_cube_z),
        "blue_shift": float(np.linalg.norm(final_blue - initial_blue)),
        "green_shift": float(np.linalg.norm(final_green - initial_green)),
        "initial_cube": initial_cube,
        "initial_blue": initial_blue,
        "initial_green": initial_green,
        "initial_joints": initial_joints,
        "observations": observations,
        "next_observations": next_observations,
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
    }


# ============================================================
# HDF5 STORAGE
# ============================================================


def save_observation_group(parent, name, data):

    group = parent.create_group(name)

    for key, values in data.items():

        array = np.asarray(values)

        # Image arrays are uint8; compress to save space.
        if key.endswith("_image"):

            group.create_dataset(
                key,
                data=array,
                compression="gzip",
                compression_opts=4,
            )

        else:

            group.create_dataset(
                key,
                data=array,
                compression="gzip",
            )


def save_demonstration(h5file, demo_id, result):

    group = h5file["data"].create_group(f"demo_{demo_id:05d}")

    group.attrs["success"] = True
    group.attrs["num_samples"] = result["steps"]
    group.attrs["max_lift"] = result["max_lift"]
    group.attrs["final_lift"] = result["final_lift"]
    group.attrs["blue_shift"] = result["blue_shift"]
    group.attrs["green_shift"] = result["green_shift"]

    group.create_dataset(
        "initial_cube",
        data=result["initial_cube"],
    )

    group.create_dataset("initial_blue", data=result["initial_blue"])
    group.create_dataset("initial_green", data=result["initial_green"])

    group.create_dataset(
        "initial_joints",
        data=result["initial_joints"],
    )

    group.create_dataset(
        "actions",
        data=result["actions"],
        compression="gzip",
    )

    group.create_dataset(
        "rewards",
        data=result["rewards"],
        compression="gzip",
    )

    save_observation_group(
        group,
        "obs",
        result["observations"],
    )

    save_observation_group(
        group,
        "next_obs",
        result["next_observations"],
    )

    # Explicit terminal flags for imitation-learning
    # dataset conversion.
    dones = np.zeros(result["steps"], dtype=np.bool_)
    dones[-1] = True

    group.create_dataset(
        "dones",
        data=dones,
    )

    h5file.flush()


def existing_demo_ids():
    ids = set()
    for path in OUTPUT_DIR.glob("demo_*.hdf5"):
        stem = path.stem
        if stem.startswith("demo_") and stem[5:].isdigit():
            ids.add(int(stem[5:]))
    return ids


def save_episode_file(demo_id, result, env):
    filename = OUTPUT_DIR / f"demo_{demo_id:05d}.hdf5"
    # Exclusive creation: never overwrite a previously saved episode.
    with h5py.File(filename, "x") as h5file:
        h5file.attrs["environment"] = "ThreeCubeLift"
        h5file.attrs["robot"] = "Panda"
        h5file.attrs["robosuite_version"] = getattr(suite, "__version__", "unknown")
        h5file.attrs["control_freq"] = CONTROL_FREQ
        h5file.attrs["seed"] = SEED
        h5file.attrs["joint_noise_deg"] = JOINT_NOISE_DEG
        h5file.attrs["target_lift_m"] = LIFT_HEIGHT
        h5file.attrs["lift_tolerance_m"] = LIFT_TOLERANCE
        h5file.attrs["image_height"] = IMAGE_HEIGHT
        h5file.attrs["image_width"] = IMAGE_WIDTH
        h5file.attrs["action_dim"] = env.action_dim
        h5file.create_group("data")
        save_demonstration(h5file, demo_id, result)
    return filename


# ============================================================
# COLLECTION
# ============================================================


def collect():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_ids = existing_demo_ids()
    saved = len(existing_ids)
    next_id = max(existing_ids, default=-1) + 1

    # Advance the RNG on resume, so restarting does not replay
    # exactly the same initial random sequence as a fresh run.
    np.random.seed(SEED + next_id)

    print("=" * 65)
    print("THREE-CUBE DEMONSTRATION COLLECTION")
    print("=" * 65)
    print(f"Target demonstrations: {NUM_DEMOS}")
    print(f"Existing files:        {saved}")
    print(f"Next demo ID:          {next_id:05d}")
    print(f"Maximum new attempts:  {MAX_ATTEMPTS}")
    print(f"Camera:                {CAMERA_NAMES}")
    print(f"Image resolution:      {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Control frequency:     {CONTROL_FREQ} Hz")
    print(f"Minimum cube lift:     {LIFT_HEIGHT - LIFT_TOLERANCE:.2f} m")
    print(f"Output directory:      {OUTPUT_DIR}")
    print("=" * 65, flush=True)

    if saved >= NUM_DEMOS:
        print("Target already reached. No new episodes needed.")
        return

    env = create_env()
    attempts = 0
    new_saved = 0
    total_steps = 0
    start_time = time.perf_counter()

    try:
        while saved < NUM_DEMOS and attempts < MAX_ATTEMPTS:
            attempts += 1
            print(f"\n[Attempt {attempts:04d} | saved {saved}/{NUM_DEMOS}]", flush=True)
            result = run_episode(env)

            if result["success"]:
                while (OUTPUT_DIR / f"demo_{next_id:05d}.hdf5").exists():
                    next_id += 1
                path = save_episode_file(next_id, result, env)
                saved += 1
                new_saved += 1
                total_steps += result["steps"]
                print(
                    f"[{attempts:04d}] SUCCESS | Saved {saved}/{NUM_DEMOS} | "
                    f"Steps {result['steps']} | Lift {result['final_lift']:.3f}m | "
                    f"Blue {result['blue_shift']:.3f}m | "
                    f"Green {result['green_shift']:.3f}m",
                    flush=True,
                )
                print(f"  File: {path}", flush=True)
                next_id += 1
            else:
                print(
                    f"[{attempts:04d}] FAILED | Steps {result['steps']} | "
                    f"Max lift {result['max_lift']:.3f}m",
                    flush=True,
                )
    finally:
        env.close()

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 65)
    print("COLLECTION SUMMARY")
    print("=" * 65)
    print(f"New attempts:       {attempts}")
    print(f"Newly saved demos:  {new_saved}")
    print(f"Total saved demos:  {saved}/{NUM_DEMOS}")
    print(f"Rejected episodes:  {attempts - new_saved}")
    print(
        f"Acceptance rate:    {100 * new_saved / attempts:.2f}%"
        if attempts
        else "Acceptance rate:    N/A"
    )
    print(f"New transitions:    {total_steps}")
    print(f"Elapsed time:       {elapsed:.1f}s")
    print(f"Dataset directory:  {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    collect()
