"""Rotation-mode regressions for sampling, expert actions, and BC checkpoints."""

import contextlib
import io
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from robosuite.models.objects import BoxObject
from robosuite.utils import transform_utils as T
from torch.utils.data import DataLoader

import bc_baseline as bc
import collect_demos as collector
import record_successful_bc as recorder
import scripted_expert as viewer
from cube_placement import (
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    MIN_CUBE_DISTANCE,
    MinimumDistanceSampler,
    object_rotation_range,
)
from camera_quality import suspicious_frames
from expert_orientation import grasp_orientation, orientation_command, orientation_distance


class ObjectRotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bc_rotation_test_")
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_demo(self, mode, name="demo_00000.hdf5", legacy=False,
                   eye_in_hand=False):
        path = self.directory / name
        rng = np.random.default_rng(123)
        length = 6
        actions = rng.uniform(-0.2, 0.2, (length, 7)).astype(np.float32)
        actions[:, 6] = np.where(np.arange(length) % 2, 1, -1)
        if not mode:
            actions[:, 3:6] = 0
        with h5py.File(path, "w") as file:
            if not legacy:
                file.attrs["control_orientation"] = mode
                file.attrs["random_object_rotation"] = mode
                file.attrs["object_yaw_range_deg"] = 45.0 if mode else 0.0
                file.attrs["use_eye_in_hand"] = eye_in_hand
            demo = file.create_group("data/demo_00000")
            demo.create_dataset("actions", data=actions)
            obs = demo.create_group("obs")
            agent_image = np.full((length, 128, 128, 3), 180, dtype=np.uint8)
            agent_image[:, 50:70, 50:70] = [220, 20, 20]
            obs.create_dataset("agentview_image", data=agent_image)
            if eye_in_hand:
                eye_image = np.full((length, 128, 128, 3), 200, dtype=np.uint8)
                eye_image[:, 48:72, 48:72] = [220, 20, 20]
                obs.create_dataset("robot0_eye_in_hand_image", data=eye_image)
            for key, size in (("robot0_joint_pos", 7), ("robot0_eef_pos", 3),
                              ("robot0_gripper_qpos", 2)):
                obs.create_dataset(key, data=rng.normal(size=(length, size)))
            quat = T.mat2quat(T.euler2mat(np.array([np.pi, 0.0, 0.0])))
            obs.create_dataset("robot0_eef_quat", data=np.tile(quat, (length, 1)))
        return path, actions

    def test_sampling_modes_preserve_spacing_and_boundaries(self):
        rng = np.random.default_rng(123)
        cubes = [BoxObject(name=name, size=[0.022] * 3, rng=rng)
                 for name in ("cube", "blue_cube", "green_cube")]
        for enabled in (False, True):
            sampler = MinimumDistanceSampler(
                name="test", mujoco_objects=cubes,
                x_range=CUBE_X_RANGE, y_range=CUBE_Y_RANGE,
                minimum_distance=MIN_CUBE_DISTANCE,
                rotation=object_rotation_range(enabled, 45.0), rng=rng,
            )
            yaws = []
            for _ in range(50):
                placements = list(sampler.sample().values())
                for first, second in combinations(placements, 2):
                    self.assertGreaterEqual(
                        np.linalg.norm(np.asarray(first[0])[:2] - np.asarray(second[0])[:2]),
                        MIN_CUBE_DISTANCE,
                    )
                for position, quat, cube in placements:
                    radius = cube.horizontal_radius
                    self.assertGreaterEqual(position[0] - radius, CUBE_X_RANGE[0])
                    self.assertLessEqual(position[0] + radius, CUBE_X_RANGE[1])
                    self.assertGreaterEqual(position[1] - radius, CUBE_Y_RANGE[0])
                    self.assertLessEqual(position[1] + radius, CUBE_Y_RANGE[1])
                    yaw = 2 * np.arctan2(quat[3], quat[0])
                    self.assertLessEqual(abs(yaw), np.pi / 4 + 1e-7)
                    yaws.append(yaw)
            if enabled:
                self.assertGreater(np.ptp(yaws), 1.0)
            else:
                np.testing.assert_allclose(yaws, 0)

    def test_invalid_yaw_ranges(self):
        for value in (0, -1, 181, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                object_rotation_range(True, value)

    def test_orientation_alignment_and_disabled_actions(self):
        eef = T.euler2mat(np.array([np.pi, 0.0, 0.3]))
        cube = T.euler2mat(np.array([0.0, 0.0, -0.4]))
        obs = {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": T.mat2quat(eef),
               "cube_quat": T.mat2quat(cube)}
        target = grasp_orientation(obs)
        np.testing.assert_allclose(target, T.euler2mat(np.array([np.pi, 0.0, -0.4])), atol=1e-6)
        self.assertAlmostEqual(orientation_distance(obs, target), 0.7, places=5)
        command = orientation_command(obs, target)
        self.assertGreater(np.linalg.norm(command), 0)
        next_rotation = T.quat2mat(T.axisangle2quat(command * 0.1)) @ eef
        next_obs = {**obs, "robot0_eef_quat": T.mat2quat(next_rotation)}
        self.assertLess(orientation_distance(next_obs, target), orientation_distance(obs, target))
        aligned_obs = {**obs, "robot0_eef_quat": T.mat2quat(target)}
        np.testing.assert_allclose(orientation_command(aligned_obs, target), 0, atol=1e-6)
        environment = SimpleNamespace(action_dim=7, action_spec=(-np.ones(7), np.ones(7)))
        for orientation in (None, target):
            action = collector.move_toward(environment, obs, np.zeros(3), -1, "align", orientation)
            with patch.object(viewer, "env", environment):
                other = viewer.move_toward(obs, np.zeros(3), -1, "align", orientation)
            np.testing.assert_allclose(action, other)
            if orientation is None:
                np.testing.assert_array_equal(action[3:6], 0)
            else:
                self.assertGreater(np.linalg.norm(action[3:6]), 0)

    def test_training_targets_round_trip_and_checkpoint_modes(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                path, actions = self.write_demo(enabled)
                mode, environment = bc.read_dataset_config([path])
                self.assertEqual(mode, enabled)
                self.assertEqual(environment["random_object_rotation"], enabled)
                keys = bc.STATE_KEYS + (("robot0_eef_quat",) if enabled else ())
                stats = bc.compute_stats([path], mode, keys)
                dataset = bc.EpisodeDataset([path], state_keys=keys, **stats)
                try:
                    image, state, motion, grip = dataset[1]
                    self.assertEqual(len(state), 16 if enabled else 12)
                    self.assertEqual(len(motion), 6 if enabled else 3)
                    decoded = bc.action_from_prediction(motion.numpy(), 1 if grip.item() else -1, stats)
                    np.testing.assert_allclose(decoded, actions[1], atol=1e-6)
                    model = bc.BCPolicy(len(state), predict_orientation=enabled)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    metrics = bc.epoch_pass(model, DataLoader(dataset, batch_size=4), "cpu", optimizer)
                    self.assertTrue(np.isfinite(metrics).all())
                    if enabled:
                        self.assertGreater(metrics[3], 0)
                    else:
                        self.assertEqual(metrics[3], 0)
                    buffer = io.BytesIO()
                    torch.save({"model": model.state_dict(), "predict_orientation": enabled}, buffer)
                    buffer.seek(0)
                    checkpoint = torch.load(buffer, weights_only=True)
                    restored = bc.BCPolicy(len(state), checkpoint["predict_orientation"])
                    restored.load_state_dict(checkpoint["model"])
                    prediction, logit = restored(image.unsqueeze(0), state.unsqueeze(0))
                    self.assertEqual(prediction.shape, (1, 6 if enabled else 3))
                    self.assertEqual(logit.shape, (1, 1))
                finally:
                    dataset.__del__()

    def test_legacy_dataset_and_checkpoint_remain_compatible(self):
        path, _ = self.write_demo(False, legacy=True)
        mode, environment = bc.read_dataset_config([path])
        self.assertFalse(mode)
        self.assertEqual(environment, {"random_object_rotation": True, "object_yaw_range_deg": 180.0})
        self.assertEqual(bc.checkpoint_environment_config({}), environment)
        legacy = bc.BCPolicy(12).state_dict()
        self.assertEqual(legacy["head.4.weight"].shape, (4, 128))
        bc.BCPolicy(12, predict_orientation=False).load_state_dict(legacy)

    def test_training_resume_keeps_epoch_optimizer_and_split(self):
        self.write_demo(False, "demo_00000.hdf5")
        self.write_demo(False, "demo_00001.hdf5")
        output = self.directory / "checkpoints"
        args = SimpleNamespace(
            dataset=str(self.directory), output=str(output), epochs=1,
            batch_size=128, lr=1e-4, val_fraction=0.5, seed=123,
            cpu=True, resume=None, wandb=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            bc.train(args)
        first = torch.load(output / "bc_last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(first["epoch"], 1)
        self.assertIn("optimizer", first)
        self.assertTrue((output / "bc_best.pt").is_file())
        args.resume = str(output / "bc_last.pt")
        args.epochs = 2
        args.seed = 999  # The saved split must prevail over a different CLI seed.
        with contextlib.redirect_stdout(io.StringIO()):
            bc.train(args)
        second = torch.load(output / "bc_last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(second["epoch"], 2)
        self.assertEqual(second["split"], first["split"])
        self.assertEqual(second["stats"], first["stats"])
        self.assertGreaterEqual(
            next(iter(second["optimizer"]["state"].values()))["step"], 2
        )

    def test_mixed_modes_are_rejected(self):
        fixed, _ = self.write_demo(False, "demo_00000.hdf5")
        rotated, _ = self.write_demo(True, "demo_00001.hdf5")
        with self.assertRaisesRegex(ValueError, "mixes rotation configurations"):
            bc.read_dataset_config([fixed, rotated])

    def test_collector_cli_enables_orientation_mode(self):
        output = self.directory / "rotated"
        argv = ["collect_demos.py", "--random-object-rotation", "--object-yaw-range-deg", "30",
                "--num-demos", "2", "--output", str(output)]
        with patch("sys.argv", argv), patch.object(collector, "collect") as collect, \
                patch.object(collector, "OUTPUT_DIR", collector.OUTPUT_DIR), \
                patch.object(collector, "RENDER", collector.RENDER):
            collector.main()
            self.assertEqual(collector.OUTPUT_DIR, output)
            collect.assert_called_once_with(True, 30.0, num_demos=2,
                                            max_attempts=collector.MAX_ATTEMPTS,
                                            use_eye_in_hand=False)

    def test_two_camera_dataset_and_checkpoint(self):
        path, _ = self.write_demo(True, eye_in_hand=True)
        mode, environment = bc.read_dataset_config([path])
        self.assertTrue(mode)
        self.assertTrue(environment["use_eye_in_hand"])
        keys = (bc.IMAGE_KEY, bc.EYE_IN_HAND_KEY)
        state_keys = bc.STATE_KEYS + ("robot0_eef_quat",)
        stats = bc.compute_stats([path], mode, state_keys)
        dataset = bc.EpisodeDataset([path], state_keys=state_keys,
                                    image_keys=keys, **stats)
        try:
            image, state, motion, grip = dataset[0]
            self.assertEqual(tuple(image.shape), (6, 128, 128))
            self.assertFalse(torch.equal(image[:3], image[3:]))
            model = bc.BCPolicy(len(state), predict_orientation=True,
                                num_cameras=2)
            predicted, logit = model(image.unsqueeze(0), state.unsqueeze(0))
            self.assertEqual(tuple(predicted.shape), (1, 6))
            self.assertEqual(tuple(logit.shape), (1, 1))
            restored = bc.BCPolicy(len(state), predict_orientation=True,
                                   num_cameras=2)
            restored.load_state_dict(model.state_dict())
            self.assertEqual(bc.image_keys_from_checkpoint({"image_keys": keys}), keys)
            with self.assertRaises(KeyError):
                bc.read_demo_info(path, state_keys, (bc.IMAGE_KEY, "missing_image"))
        finally:
            dataset.__del__()
        with h5py.File(path, "r+") as file:
            file["data/demo_00000/obs/robot0_eye_in_hand_image"][3] = 0
        with self.assertRaisesRegex(ValueError, "corrupted"):
            bc.read_demo_info(path, state_keys, keys)

    def test_camera_quality_rejects_corrupt_frames(self):
        good = np.full((3, 128, 128, 3), 180, dtype=np.uint8)
        good[:, 50:70, 50:70] = [220, 20, 20]
        self.assertEqual(len(suspicious_frames(good)), 0)
        bad = good.copy()
        bad[1] = 0
        bad[2] = np.random.default_rng(5).integers(
            0, 256, (128, 128, 3), dtype=np.uint8
        )
        np.testing.assert_array_equal(suspicious_frames(bad), [1, 2])

    def test_success_recorder_restores_and_executes_rotation(self):
        # Exercise inference without requiring an MP4 backend; encoding is unrelated
        # to restoring orientation settings and forwarding all seven action values.
        model = bc.BCPolicy(16, predict_orientation=True)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head[-1].bias[-1] = 1
        stats = {
            "state_mean": [0.0] * 16, "state_std": [1.0] * 16,
            "xyz_mean": [0.0] * 3, "xyz_std": [1.0] * 3,
            "rotation_mean": [0.1, 0.2, 0.3], "rotation_std": [1.0] * 3,
        }
        checkpoint = self.directory / "policy.pt"
        torch.save({
            "model": model.state_dict(), "state_dim": 16, "stats": stats,
            "predict_orientation": True,
            "state_keys": bc.STATE_KEYS + ("robot0_eef_quat",),
            "environment_config": {"random_object_rotation": True, "object_yaw_range_deg": 30.0},
        }, checkpoint)
        actions = []

        def observation(lift=0.0):
            return {
                "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
                "robot0_joint_pos": np.zeros(7), "robot0_eef_pos": np.zeros(3),
                "robot0_gripper_qpos": np.zeros(2), "robot0_eef_quat": np.array([1., 0., 0., 0.]),
                "cube_pos": np.array([0., 0., 0.82 + lift]),
            }

        def step(action):
            actions.append(action.copy())
            return observation(0.2), 0, False, {}

        environment = SimpleNamespace(
            reset=lambda: observation(), step=step, close=lambda: None,
            action_spec=(-np.ones(7), np.ones(7)), _check_success=lambda: True,
        )

        class Writer:
            def __init__(self, filename, **kwargs):
                self.path = Path(filename)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.path.write_bytes(b"test video")

            def append_data(self, frame):
                pass

        imageio = ModuleType("imageio")
        imageio.v2 = ModuleType("imageio.v2")
        imageio.v2.get_writer = Writer
        imageio.v2.imwrite = lambda path, image: Path(path).write_bytes(b"test image")
        modules = {"imageio": imageio, "imageio.v2": imageio.v2,
                   "imageio_ffmpeg": ModuleType("imageio_ffmpeg")}
        output = self.directory / "videos"
        argv = ["record_successful_bc.py", "--checkpoint", str(checkpoint),
                "--successes", "1", "--max-attempts", "1", "--max-steps", "25",
                "--output", str(output), "--cpu"]
        with patch.dict("sys.modules", modules), patch("sys.argv", argv), \
                patch.object(collector, "create_env", return_value=environment) as create, \
                patch.object(bc, "choose_third_person_camera", return_value="frontview"), \
                patch.object(bc, "render_third_person", return_value=np.zeros((128, 128, 3), dtype=np.uint8)), \
                contextlib.redirect_stdout(io.StringIO()):
            recorder.main()
        create.assert_called_once_with(random_object_rotation=True, object_yaw_range_deg=30.0)
        self.assertEqual(len(actions), collector.LIFT_HOLD_STEPS)
        np.testing.assert_allclose(np.asarray(actions)[:, 3:6], np.tile([0.1, 0.2, 0.3], (len(actions), 1)))
        self.assertTrue(json.loads((output / "summary.json").read_text())["predict_orientation"])

    def test_evaluation_records_success_and_failure_locally(self):
        model = bc.BCPolicy(12)
        checkpoint = self.directory / "policy.pt"
        torch.save({
            "model": model.state_dict(), "state_dim": 12,
            "stats": {"state_mean": [0.] * 12, "state_std": [1.] * 12,
                      "xyz_mean": [0.] * 3, "xyz_std": [1.] * 3},
            "environment_config": {"random_object_rotation": False,
                                   "object_yaw_range_deg": 45.0},
        }, checkpoint)
        episodes = [0]

        def observation(lift=0.0):
            return {"agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
                    "robot0_joint_pos": np.zeros(7), "robot0_eef_pos": np.zeros(3),
                    "robot0_gripper_qpos": np.zeros(2),
                    "cube_pos": np.array([0., 0., 0.82 + lift])}

        def reset():
            episodes[0] += 1
            return observation()

        def step(action):
            return observation(0.2 if episodes[0] == 1 else 0.0), 0, False, {}

        environment = SimpleNamespace(
            reset=reset, step=step, close=lambda: None,
            action_spec=(-np.ones(7), np.ones(7)), _check_success=lambda: True,
        )

        class Writer:
            def __init__(self, path, **kwargs):
                self.path = Path(path)
                self.frames = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.path.write_bytes(bytes([self.frames]))

            def append_data(self, frame):
                self.frames += 1

        imageio = ModuleType("imageio")
        imageio.v2 = ModuleType("imageio.v2")
        imageio.v2.get_writer = Writer
        modules = {"imageio": imageio, "imageio.v2": imageio.v2,
                   "imageio_ffmpeg": ModuleType("imageio_ffmpeg")}
        output = self.directory / "evaluation"
        argv = ["bc_baseline.py", "eval", "--checkpoint", str(checkpoint),
                "--episodes", "2", "--num-seeds", "1", "--max-steps", "20",
                "--diagnostic-interval", "0", "--record-videos",
                "--video-episodes", "2", "--video-stride", "2",
                "--eval-output", str(output), "--cpu"]
        with patch.dict("sys.modules", modules), patch("sys.argv", argv), \
                patch.object(collector, "create_env", return_value=environment), \
                patch.object(bc, "choose_third_person_camera", return_value="frontview"), \
                patch.object(bc, "render_third_person", return_value=np.zeros((128, 128, 3), dtype=np.uint8)), \
                contextlib.redirect_stdout(io.StringIO()):
            bc.main()
        summary = json.loads((output / "bc_eval_summary.json").read_text())
        self.assertEqual(summary["total_episodes"], 2)
        self.assertEqual(summary["total_successes"], 1)
        self.assertEqual(summary["local_videos_recorded"], 2)
        self.assertEqual(len(list((output / "videos").glob("*.mp4"))), 4)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
