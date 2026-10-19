#!/usr/bin/env python3
"""Train or evaluate a vision+proprioception BC baseline on per-episode HDF5 demos.

Training: python3 bc_baseline.py train --dataset datasets/three_cube_lift
Evaluation: python3 bc_baseline.py eval --checkpoint checkpoints/bc_best.pt --episodes 50
Run from the directory containing collect_demos.py for evaluation.
Orientation prediction and object yaw settings are inferred from dataset metadata
and restored from the checkpoint during evaluation.
"""

import argparse
import bisect
import copy
import csv
import json
import random
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from camera_quality import suspicious_frames
from eval_trace import RISE_THRESHOLD_M, STEP_FIELDS, EpisodeTrace


def init_wandb(args, job_type, config):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B is not installed. Run: python3 -m pip install wandb"
        ) from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        job_type=job_type,
        config=config,
        tags=["BC", "ThreeCubeLift"],
    )
    return run


def log_training_camera_samples(
    run, paths, max_images=6, image_keys=("agentview_image",)
):
    """Show held-out demonstration camera frames on the training dashboard."""
    if run is None:
        return
    import wandb

    samples = []
    for path in paths[:max_images]:
        with h5py.File(path, "r") as f:
            demo = f["data"][next(iter(f["data"]))]
            for key in image_keys:
                images = demo["obs"][key]
                index = len(images) // 2
                # Robosuite camera observations are vertically flipped for display.
                rgb = np.flipud(np.asarray(images[index])).copy()
                samples.append(
                    wandb.Image(
                        rgb, caption=f"{path.stem} | {key} | step {index} | validation"
                    )
                )
    if samples:
        run.log({"data/validation_robot_camera": samples})


def choose_third_person_camera(env, preferred):
    """Choose an available camera distinct from the policy's agentview."""
    available = list(env.sim.model.camera_names)
    available = [
        name.decode() if isinstance(name, bytes) else str(name) for name in available
    ]
    candidates = [preferred, "frontview", "sideview", "birdview", "robot0_eye_in_hand"]
    for name in candidates:
        if name in available and name != "agentview":
            return name
    raise RuntimeError(
        "No third-person camera found. Available cameras: "
        + ", ".join(available)
        + ". Supply --third-person-camera with an available camera name."
    )


def render_third_person(env, camera_name, size):
    # Explicit offscreen render; no on-screen viewer is needed.
    frame = env.sim.render(width=size, height=size, camera_name=camera_name)
    return np.flipud(np.asarray(frame)).copy()


def cube_grasp_contact(env):
    """Use Robosuite's grasp check when the collector exposes the red cube."""
    robots = getattr(env, "robots", None)
    cube = getattr(env, "cube", None)
    check_grasp = getattr(env, "_check_grasp", None)
    if not robots or cube is None or check_grasp is None:
        return None
    return bool(check_grasp(gripper=robots[0].gripper, object_geoms=cube))


def log_episode_videos(run, ep, robot_frames, third_frames, third_camera, fps):
    if run is None or not robot_frames:
        return
    import wandb

    # W&B Video expects T,C,H,W uint8 arrays.
    def video(frames, caption):
        array = np.stack(frames).transpose(0, 3, 1, 2)
        return wandb.Video(array, fps=fps, format="mp4", caption=caption)

    run.log(
        {
            "eval/robot_camera_video": video(
                robot_frames, f"Episode {ep:03d} | agentview"
            ),
            "eval/third_person_video": video(
                third_frames, f"Episode {ep:03d} | {third_camera}"
            ),
        },
        step=ep,
    )


IMAGE_KEY = "agentview_image"
EYE_IN_HAND_KEY = "robot0_eye_in_hand_image"
STATE_KEYS = ("robot0_joint_pos", "robot0_eef_pos", "robot0_gripper_qpos")


def image_keys_from_checkpoint(checkpoint):
    return tuple(
        checkpoint.get("image_keys", (checkpoint.get("image_key", IMAGE_KEY),))
    )


def image_from_observation(obs, image_keys):
    """Stack normalized RGB cameras along channels in checkpoint order."""
    return torch.cat(
        [
            torch.from_numpy(np.asarray(obs[key]).copy()).permute(2, 0, 1).float()
            / 255.0
            for key in image_keys
        ],
        dim=0,
    )


def read_dataset_config(paths):
    """Infer the action head and reset distribution from demonstration metadata."""
    configurations = set()
    for path in paths:
        with h5py.File(path, "r") as file:
            control_orientation = bool(file.attrs.get("control_orientation", False))
            # Earlier collectors randomized full yaw but commanded zero rotation.
            random_rotation = bool(file.attrs.get("random_object_rotation", True))
            yaw_range = float(file.attrs.get("object_yaw_range_deg", 180.0))
            eye_in_hand = bool(file.attrs.get("use_eye_in_hand", False))
            configurations.add(
                (
                    control_orientation,
                    random_rotation,
                    yaw_range if random_rotation else 0.0,
                    eye_in_hand,
                )
            )
    if len(configurations) != 1:
        raise ValueError(
            "Dataset mixes rotation configurations; train each dataset separately"
        )
    control_orientation, random_rotation, yaw_range, eye_in_hand = configurations.pop()
    environment = {
        "random_object_rotation": random_rotation,
        "object_yaw_range_deg": yaw_range if random_rotation else 45.0,
    }
    if control_orientation != random_rotation:
        environment["control_orientation"] = control_orientation
    if eye_in_hand:
        environment["use_eye_in_hand"] = True
    return control_orientation, environment


def action_from_prediction(motion, grip_logit, stats):
    """Denormalize XYZ and optional rotation into the environment's 7D action."""
    motion = np.asarray(motion, dtype=np.float32)
    action = np.zeros(7, dtype=np.float32)
    action[:3] = motion[:3] * np.asarray(stats["xyz_std"]) + stats["xyz_mean"]
    if len(motion) == 6:
        action[3:6] = (
            motion[3:6] * np.asarray(stats["rotation_std"]) + stats["rotation_mean"]
        )
    elif len(motion) != 3:
        raise ValueError(f"Expected 3 or 6 motion values, got {len(motion)}")
    # print(f" Grip logit: {grip_logit:.3f}", flush=True)
    action[6] = 1.0 if grip_logit > -0.5 else -1.0
    return action


def checkpoint_environment_config(checkpoint):
    # Preserve the full-yaw reset distribution of checkpoints predating this flag.
    return checkpoint.get(
        "environment_config",
        {
            "random_object_rotation": True,
            "object_yaw_range_deg": 180.0,
            "control_orientation": False,
        },
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_demo_info(path, state_keys=STATE_KEYS, image_keys=(IMAGE_KEY,)):
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys())
        if len(names) != 1:
            raise ValueError(f"{path}: expected exactly one demo group, found {names}")
        g = f["data"][names[0]]
        t = len(g["actions"])
        for key in tuple(state_keys) + tuple(image_keys):
            if key not in g["obs"]:
                raise KeyError(
                    f'{path}: missing obs/{key}; available: {list(g["obs"].keys())}'
                )
            if len(g["obs"][key]) != t:
                raise ValueError(f"{path}: inconsistent length for {key}")
        if g["actions"].shape != (t, 7):
            raise ValueError(
                f'{path}: expected actions (T,7), got {g["actions"].shape}'
            )
        for key in image_keys:
            if g["obs"][key].shape[1:] != (128, 128, 3):
                raise ValueError(f"{path}: expected 128x128 RGB image for {key}")
            if len(image_keys) == 2 and t:
                sample_indices = np.linspace(0, t - 1, min(8, t), dtype=int)
                frames = np.stack([g["obs"][key][int(i)] for i in sample_indices])
                bad = suspicious_frames(frames)
                if len(bad):
                    raise ValueError(
                        f"{path}: corrupted {key} frame near step {sample_indices[bad[0]]}"
                    )
        state_dim = sum(g["obs"][k].shape[1] for k in state_keys)
        return t, state_dim


class EpisodeDataset(Dataset):
    def __init__(
        self,
        paths,
        state_mean,
        state_std,
        xyz_mean,
        xyz_std,
        rotation_mean=None,
        rotation_std=None,
        state_keys=STATE_KEYS,
        image_keys=(IMAGE_KEY,),
    ):
        self.paths = [str(p) for p in paths]
        self.state_keys = tuple(state_keys)
        self.image_keys = tuple(image_keys)
        self.lengths = [
            read_demo_info(p, self.state_keys, self.image_keys)[0] for p in self.paths
        ]
        self.ends = np.cumsum(self.lengths).tolist()
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.tensor(state_std, dtype=torch.float32)
        self.motion_mean = torch.tensor(
            list(xyz_mean) + ([] if rotation_mean is None else list(rotation_mean)),
            dtype=torch.float32,
        )
        self.motion_std = torch.tensor(
            list(xyz_std) + ([] if rotation_std is None else list(rotation_std)),
            dtype=torch.float32,
        )
        self.handles = {}

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        file_index = bisect.bisect_right(self.ends, index)
        offset = 0 if file_index == 0 else self.ends[file_index - 1]
        path = self.paths[file_index]
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r")
        f = self.handles[path]
        g = f["data"][next(iter(f["data"].keys()))]
        i = index - offset
        image = image_from_observation(
            {key: g["obs"][key][i] for key in self.image_keys}, self.image_keys
        )
        state = np.concatenate([g["obs"][key][i] for key in self.state_keys]).astype(
            np.float32
        )
        state = (torch.from_numpy(state) - self.state_mean) / self.state_std
        action = np.asarray(g["actions"][i], dtype=np.float32)
        motion = (
            torch.from_numpy(action[: len(self.motion_mean)].copy()) - self.motion_mean
        ) / self.motion_std
        gripper = torch.tensor([float(action[6] > 0)], dtype=torch.float32)
        return image, state, motion, gripper

    def __del__(self):
        for f in getattr(self, "handles", {}).values():
            try:
                f.close()
            except Exception:
                pass


class EpisodeSequenceDataset(EpisodeDataset):
    """Causal windows from one demo at a time; padded steps carry no loss."""

    def __init__(self, paths, *args, sequence_length=16, sequence_stride=8, **kwargs):
        super().__init__(paths, *args, **kwargs)
        if sequence_length < 1 or sequence_stride < 1:
            raise ValueError("Sequence length and stride must be positive")
        self.sequence_length = sequence_length
        self.windows = [
            (file_index, start)
            for file_index, length in enumerate(self.lengths)
            for start in range(0, length, sequence_stride)
        ]
        if not self.windows:
            raise ValueError("Temporal training needs nonempty demonstrations")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        file_index, start = self.windows[index]
        path = self.paths[file_index]
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r")
        f = self.handles[path]
        g = f["data"][next(iter(f["data"].keys()))]
        end = min(start + self.sequence_length, self.lengths[file_index])
        valid = end - start
        images = torch.cat(
            [
                torch.from_numpy(np.asarray(g["obs"][key][start:end]).copy())
                .permute(0, 3, 1, 2)
                .float()
                / 255.0
                for key in self.image_keys
            ],
            dim=1,
        )
        state = np.concatenate(
            [g["obs"][key][start:end] for key in self.state_keys], axis=1
        ).astype(np.float32)
        state = (torch.from_numpy(state) - self.state_mean) / self.state_std
        action = np.asarray(g["actions"][start:end], dtype=np.float32)
        motion = (
            torch.from_numpy(action[:, : len(self.motion_mean)].copy())
            - self.motion_mean
        ) / self.motion_std
        gripper = torch.from_numpy((action[:, 6:7] > 0).astype(np.float32))
        if start == 0:
            previous = np.concatenate(([-1.0], action[:-1, 6]))
        else:
            previous = np.asarray(g["actions"][start - 1 : end - 1, 6])
        previous_gripper = torch.from_numpy(
            np.where(previous > 0, 1.0, -1.0).astype(np.float32)[:, None]
        )
        mask = torch.zeros(self.sequence_length, dtype=torch.float32)
        mask[:valid] = 1.0
        if valid < self.sequence_length:

            def pad_last(value):
                return torch.cat(
                    (
                        value,
                        value[-1:].expand(
                            self.sequence_length - valid, *value.shape[1:]
                        ),
                    ),
                    dim=0,
                )

            images, state, motion, gripper, previous_gripper = (
                pad_last(value)
                for value in (images, state, motion, gripper, previous_gripper)
            )
        return images, state, previous_gripper, motion, gripper, mask


def compute_stats(paths, predict_orientation=False, state_keys=STATE_KEYS):
    state_sum = state_sq = xyz_sum = xyz_sq = None
    motion_dim = 6 if predict_orientation else 3
    count = 0
    for path in paths:
        with h5py.File(path, "r") as f:
            g = f["data"][next(iter(f["data"].keys()))]
            state = np.concatenate(
                [g["obs"][key][:] for key in state_keys], axis=1
            ).astype(np.float64)
            xyz = g["actions"][:, :motion_dim].astype(np.float64)
            if state_sum is None:
                state_sum = np.zeros(state.shape[1])
                state_sq = np.zeros(state.shape[1])
                xyz_sum = np.zeros(motion_dim)
                xyz_sq = np.zeros(motion_dim)
            state_sum += state.sum(0)
            state_sq += (state**2).sum(0)
            xyz_sum += xyz.sum(0)
            xyz_sq += (xyz**2).sum(0)
            count += len(xyz)
    sm = state_sum / count
    ss = np.maximum(np.sqrt(np.maximum(state_sq / count - sm**2, 0)), 1e-3)
    xm = xyz_sum / count
    xs = np.maximum(np.sqrt(np.maximum(xyz_sq / count - xm**2, 0)), 1e-2)
    stats = dict(
        state_mean=sm.tolist(),
        state_std=ss.tolist(),
        xyz_mean=xm[:3].tolist(),
        xyz_std=xs[:3].tolist(),
    )
    if predict_orientation:
        stats.update(rotation_mean=xm[3:].tolist(), rotation_std=xs[3:].tolist())
    return stats


class BCPolicy(nn.Module):
    def __init__(self, state_dim, predict_orientation=False, num_cameras=1):
        super().__init__()
        self.predict_orientation = bool(predict_orientation)
        if num_cameras not in (1, 2):
            raise ValueError("BCPolicy supports one or two cameras")
        self.num_cameras = num_cameras
        self.motion_dim = 6 if self.predict_orientation else 3
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
        )
        if num_cameras == 2:
            self.eye_vision = copy.deepcopy(self.vision)
        self.head = nn.Sequential(
            nn.Linear(256 * num_cameras + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.motion_dim + 1),
        )

    def forward(self, image, state):
        if image.shape[1] != 3 * self.num_cameras:
            raise ValueError(f"Expected {3 * self.num_cameras} image channels")
        visual = self.vision(image[:, :3])
        if self.num_cameras == 2:
            visual = torch.cat([visual, self.eye_vision(image[:, 3:])], dim=1)
        out = self.head(torch.cat([visual, state], dim=1))
        return out[:, : self.motion_dim], out[:, self.motion_dim :]


class TemporalBCPolicy(BCPolicy):
    """One recurrent policy step per observation, with the previous gripper command."""

    def __init__(self, state_dim, predict_orientation=False, num_cameras=1):
        super().__init__(state_dim, predict_orientation, num_cameras)
        self.gru = nn.GRU(256 * num_cameras + state_dim + 1, 256, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.motion_dim + 1),
        )

    def forward(self, images, states, previous_gripper, hidden=None):
        batch, steps, channels, height, width = images.shape
        if channels != 3 * self.num_cameras:
            raise ValueError(f"Expected {3 * self.num_cameras} image channels")
        flat = images.reshape(batch * steps, channels, height, width)
        visual = self.vision(flat[:, :3])
        if self.num_cameras == 2:
            visual = torch.cat([visual, self.eye_vision(flat[:, 3:])], dim=1)
        features = torch.cat(
            (visual.reshape(batch, steps, -1), states, previous_gripper), dim=-1
        )
        recurrent, hidden = self.gru(features, hidden)
        out = self.head(recurrent)
        return out[..., : self.motion_dim], out[..., self.motion_dim :], hidden


def policy_from_checkpoint(checkpoint, device):
    policy_type = checkpoint.get("policy_type", "frame")
    if policy_type not in ("frame", "temporal_gru"):
        raise ValueError(f"Unknown checkpoint policy_type: {policy_type}")
    policy_class = TemporalBCPolicy if policy_type == "temporal_gru" else BCPolicy
    model = policy_class(
        checkpoint["state_dim"],
        predict_orientation=checkpoint.get("predict_orientation", False),
        num_cameras=len(image_keys_from_checkpoint(checkpoint)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict_step(model, image, state, previous_gripper=-1.0, hidden=None):
    """Run one rollout step, retaining memory only for temporal checkpoints."""
    if isinstance(model, TemporalBCPolicy):
        grip_input = torch.full(
            (len(image), 1, 1), previous_gripper, dtype=state.dtype, device=state.device
        )
        motion, grip_logit, hidden = model(
            image.unsqueeze(1), state.unsqueeze(1), grip_input, hidden
        )
        return motion[:, 0], grip_logit[:, 0], hidden
    motion, grip_logit = model(image, state)
    return motion, grip_logit, None


def epoch_pass(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_xyz = total_grip = total_rotation = n = 0
    for image, state, motion, gripper in loader:
        image, state, motion, gripper = [
            x.to(device) for x in (image, state, motion, gripper)
        ]
        with torch.set_grad_enabled(training):
            pred_motion, grip_logit = model(image, state)
            xyz_loss = nn.functional.mse_loss(pred_motion[:, :3], motion[:, :3])
            rotation_loss = (
                nn.functional.mse_loss(pred_motion[:, 3:6], motion[:, 3:6])
                if model.predict_orientation
                else pred_motion.new_zeros(())
            )
            grip_loss = nn.functional.binary_cross_entropy_with_logits(
                grip_logit, gripper
            )
            loss = xyz_loss + rotation_loss + 0.2 * grip_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        b = len(image)
        total_loss += loss.item() * b
        total_xyz += xyz_loss.item() * b
        total_grip += grip_loss.item() * b
        total_rotation += rotation_loss.item() * b
        n += b
    return total_loss / n, total_xyz / n, total_grip / n, total_rotation / n


def temporal_epoch_pass(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_xyz = total_grip = total_rotation = n = 0.0
    for images, states, previous_gripper, motion, gripper, mask in loader:
        images, states, previous_gripper, motion, gripper, mask = (
            value.to(device)
            for value in (images, states, previous_gripper, motion, gripper, mask)
        )
        valid = mask.sum()
        with torch.set_grad_enabled(training):
            predicted, grip_logit, _ = model(images, states, previous_gripper)

            def masked_mse(pred, target):
                return (((pred - target) ** 2).mean(dim=-1) * mask).sum() / valid

            xyz_loss = masked_mse(predicted[..., :3], motion[..., :3])
            rotation_loss = (
                masked_mse(predicted[..., 3:6], motion[..., 3:6])
                if model.predict_orientation
                else predicted.new_zeros(())
            )
            grip_each = nn.functional.binary_cross_entropy_with_logits(
                grip_logit, gripper, reduction="none"
            ).squeeze(-1)
            grip_loss = (grip_each * mask).sum() / valid
            loss = xyz_loss + rotation_loss + 0.2 * grip_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        weight = valid.item()
        total_loss += loss.item() * weight
        total_xyz += xyz_loss.item() * weight
        total_grip += grip_loss.item() * weight
        total_rotation += rotation_loss.item() * weight
        n += weight
    return total_loss / n, total_xyz / n, total_grip / n, total_rotation / n


def train(args):
    seed_all(args.seed)
    paths = sorted(Path(args.dataset).glob("demo_*.hdf5"))
    if len(paths) < 2:
        raise RuntimeError(
            f"Need at least two per-episode HDF5 files in {args.dataset}"
        )
    predict_orientation, environment_config = read_dataset_config(paths)
    state_keys = STATE_KEYS + (("robot0_eef_quat",) if predict_orientation else ())
    image_keys = (
        (IMAGE_KEY, EYE_IN_HAND_KEY)
        if environment_config.get("use_eye_in_hand", False)
        else (IMAGE_KEY,)
    )
    output = Path(args.output)
    if args.resume and output.resolve() != Path(args.resume).resolve().parent:
        raise ValueError("--output must be the directory containing --resume")
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    saved_policy_type = checkpoint.get("policy_type", "frame") if checkpoint else None
    if saved_policy_type not in (None, "frame", "temporal_gru"):
        raise ValueError(f"Unknown checkpoint policy_type: {saved_policy_type}")
    temporal = (
        bool(getattr(args, "temporal", False)) or saved_policy_type == "temporal_gru"
    )
    if checkpoint is not None and temporal != (saved_policy_type == "temporal_gru"):
        raise ValueError("--temporal does not match the checkpoint architecture")
    sequence_length = getattr(args, "sequence_length", None)
    sequence_stride = getattr(args, "sequence_stride", None)
    if temporal:
        saved_length = checkpoint.get("sequence_length", 16) if checkpoint else 16
        saved_stride = checkpoint.get("sequence_stride", 8) if checkpoint else 8
        if (
            checkpoint
            and sequence_length is not None
            and sequence_length != saved_length
        ):
            raise ValueError("--sequence-length differs from checkpoint")
        if (
            checkpoint
            and sequence_stride is not None
            and sequence_stride != saved_stride
        ):
            raise ValueError("--sequence-stride differs from checkpoint")
        sequence_length = saved_length if sequence_length is None else sequence_length
        sequence_stride = saved_stride if sequence_stride is None else sequence_stride
        if (
            sequence_length < 1
            or sequence_stride < 1
            or sequence_stride > sequence_length
        ):
            raise ValueError("Need 1 <= sequence-stride <= sequence-length")
    elif sequence_length is not None or sequence_stride is not None:
        raise ValueError("--sequence-length and --sequence-stride require --temporal")
    batch_size = getattr(args, "batch_size", None)
    batch_size = (8 if temporal else 128) if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if checkpoint is not None:
        if checkpoint.get("predict_orientation", False) != predict_orientation:
            raise ValueError("Checkpoint orientation mode differs from dataset")
        if checkpoint_environment_config(checkpoint) != environment_config:
            raise ValueError("Checkpoint object rotation settings differ from dataset")
        if tuple(checkpoint.get("state_keys", STATE_KEYS)) != state_keys:
            raise ValueError("Checkpoint state keys differ from dataset")
        if image_keys_from_checkpoint(checkpoint) != image_keys:
            raise ValueError("Checkpoint camera inputs differ from dataset")
        split = checkpoint.get("split")
        if split is None:
            split_path = Path(args.resume).parent / "split.json"
            if not split_path.is_file():
                raise ValueError(f"Cannot resume: missing training split {split_path}")
            split = json.loads(split_path.read_text())
        by_name = {path.name: path for path in paths}
        train_names, val_names = split["train"], split["val"]
        if (
            len(train_names) + len(val_names) != len(paths)
            or len(set(train_names + val_names)) != len(paths)
            or set(train_names + val_names) != set(by_name)
        ):
            raise ValueError("Dataset files differ from checkpoint training split")
        train_paths = [by_name[name] for name in train_names]
        val_paths = [by_name[name] for name in val_names]
    else:
        random.Random(args.seed).shuffle(paths)
        n_val = max(1, round(len(paths) * args.val_fraction))
        val_paths, train_paths = paths[:n_val], paths[n_val:]
        split = {
            "train": [p.name for p in train_paths],
            "val": [p.name for p in val_paths],
        }
    state_dims = {read_demo_info(p, state_keys, image_keys)[1] for p in paths}
    if len(state_dims) != 1:
        raise RuntimeError(f"Inconsistent state dimensions: {state_dims}")
    stats = (
        checkpoint["stats"]
        if checkpoint is not None
        else compute_stats(train_paths, predict_orientation, state_keys)
    )
    if len(stats["state_mean"]) != next(iter(state_dims)):
        raise ValueError("Checkpoint state dimension differs from dataset")
    dataset_class = EpisodeSequenceDataset if temporal else EpisodeDataset
    sequence_options = (
        {"sequence_length": sequence_length, "sequence_stride": sequence_stride}
        if temporal
        else {}
    )
    train_set = dataset_class(
        train_paths,
        state_keys=state_keys,
        image_keys=image_keys,
        **stats,
        **sequence_options,
    )
    val_set = dataset_class(
        val_paths,
        state_keys=state_keys,
        image_keys=image_keys,
        **stats,
        **sequence_options,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model_class = TemporalBCPolicy if temporal else BCPolicy
    model = model_class(
        state_dims.pop(),
        predict_orientation=predict_orientation,
        num_cameras=len(image_keys),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 1
    best = float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_val_loss"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        else:
            print(
                "Checkpoint lama: optimizer tidak tersimpan; optimizer dimulai ulang.",
                flush=True,
            )
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint is at epoch {start_epoch - 1}; --epochs must be greater"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "split.json").write_text(json.dumps(split, indent=2))
    run = init_wandb(
        args,
        "train",
        {
            "algorithm": "BC",
            "dataset": str(args.dataset),
            "train_episodes": len(train_paths),
            "val_episodes": len(val_paths),
            "train_transitions": sum(train_set.lengths),
            "val_transitions": sum(val_set.lengths),
            "train_windows": len(train_set) if temporal else None,
            "val_windows": len(val_set) if temporal else None,
            "epochs": args.epochs,
            "batch_size": batch_size,
            "policy_type": "temporal_gru" if temporal else "frame",
            "sequence_length": sequence_length,
            "sequence_stride": sequence_stride,
            "lr": args.lr,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "image_key": IMAGE_KEY,
            "image_keys": list(image_keys),
            "state_keys": list(state_keys),
            "predict_orientation": predict_orientation,
            "environment_config": environment_config,
            "action_dim": 7,
            "device": str(device),
        },
    )
    if run is not None:
        print(f"W&B dashboard: {run.url}", flush=True)
        log_training_camera_samples(run, val_paths, image_keys=image_keys)
    print(
        f"Policy: {'temporal_gru' if temporal else 'frame'} | "
        f"Orientation actions: {predict_orientation} | Cameras: {image_keys} | State keys: {state_keys}",
        flush=True,
    )
    if temporal:
        print(
            f"Sequence length: {sequence_length} | Stride: {sequence_stride} | "
            f"Batch: {batch_size} windows | Train/val windows: {len(train_set)}/{len(val_set)}",
            flush=True,
        )
    if checkpoint is not None:
        print(f"Resuming from epoch {start_epoch - 1} to {args.epochs}", flush=True)
    print(
        f"Train: {len(train_paths)} episodes/{sum(train_set.lengths)} transitions | "
        f"Validation: {len(val_paths)} episodes/{sum(val_set.lengths)} transitions | Device: {device}",
        flush=True,
    )
    pass_epoch = temporal_epoch_pass if temporal else epoch_pass
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            tr = pass_epoch(model, train_loader, device, optimizer)
            va = pass_epoch(model, val_loader, device)
            improved = va[0] < best
            if improved:
                best = va[0]
            saved = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stats": stats,
                "state_dim": len(stats["state_mean"]),
                "epoch": epoch,
                "best_val_loss": best,
                "state_keys": state_keys,
                "image_key": IMAGE_KEY,
                "image_keys": image_keys,
                "predict_orientation": predict_orientation,
                "environment_config": environment_config,
                "split": split,
                "policy_type": "temporal_gru" if temporal else "frame",
                "sequence_length": sequence_length,
                "sequence_stride": sequence_stride,
            }
            if improved:
                torch.save(saved, output / "bc_best.pt")
            latest_tmp = output / "bc_last.pt.tmp"
            torch.save(saved, latest_tmp)
            latest_tmp.replace(output / "bc_last.pt")
            print(
                f"Epoch {epoch:03d}/{args.epochs}: train={tr[0]:.5f} "
                f"val={va[0]:.5f} (xyz={va[1]:.5f}, rotation={va[3]:.5f}, "
                f"gripper={va[2]:.5f}) "
                f"best={best:.5f}",
                flush=True,
            )
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        "train/loss": tr[0],
                        "train/xyz_mse": tr[1],
                        "train/rotation_mse": tr[3],
                        "train/gripper_bce": tr[2],
                        "val/loss": va[0],
                        "val/xyz_mse": va[1],
                        "val/rotation_mse": va[3],
                        "val/gripper_bce": va[2],
                        "val/best_loss": best,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "checkpoint/improved": int(improved),
                    },
                    step=epoch,
                )
        print(f'Best checkpoint: {output / "bc_best.pt"} | val loss={best:.5f}')
        if run is not None:
            run.summary["best_val_loss"] = best
            run.summary["best_checkpoint"] = str(output / "bc_best.pt")
    finally:
        if run is not None:
            run.finish()


def evaluate(args):
    """Evaluate 20 episodes per seed for five seeds; diagnose stationary policies."""
    import importlib
    import statistics
    import time

    collector = importlib.import_module(args.collector_module)
    required = (
        "create_env",
        "CONTROL_FREQ",
        "LIFT_HEIGHT",
        "LIFT_TOLERANCE",
        "LIFT_HOLD_STEPS",
    )
    missing = [name for name in required if not hasattr(collector, name)]
    if missing:
        raise RuntimeError(f"Collector module {args.collector_module} lacks: {missing}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_keys = image_keys_from_checkpoint(checkpoint)
    model = policy_from_checkpoint(checkpoint, device)
    policy_type = checkpoint.get("policy_type", "frame")
    stats = checkpoint["stats"]
    state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
    state_std = np.asarray(stats["state_std"], dtype=np.float32)
    state_keys = tuple(checkpoint.get("state_keys", STATE_KEYS))
    image_key = image_keys[0]
    environment_config = checkpoint_environment_config(checkpoint)
    lift_threshold = float(collector.LIFT_HEIGHT - collector.LIFT_TOLERANCE)
    hold_steps = int(collector.LIFT_HOLD_STEPS)
    freq = int(collector.CONTROL_FREQ)
    seeds = [args.seed + i * args.seed_step for i in range(args.num_seeds)]
    output = Path(args.eval_output)
    output.mkdir(parents=True, exist_ok=True)
    local_video_enabled = getattr(args, "record_videos", False)
    if local_video_enabled:
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401 - ensure MP4 backend is installed
        except ImportError as exc:
            raise RuntimeError(
                "Local MP4 recording requires: python -m pip install imageio imageio-ffmpeg"
            ) from exc
        video_dir = output / "videos"
        video_dir.mkdir(exist_ok=True)
    local_videos_recorded = 0

    run = init_wandb(
        args,
        "eval",
        {
            "algorithm": "BC",
            "policy_type": policy_type,
            "checkpoint": str(args.checkpoint),
            "episodes_per_seed": args.episodes,
            "num_seeds": args.num_seeds,
            "seeds": seeds,
            "max_steps": args.max_steps,
            "device": str(device),
            "lift_threshold_m": lift_threshold,
            "lift_hold_steps": hold_steps,
            "live_render": args.render,
            "collector_module": args.collector_module,
            "predict_orientation": model.predict_orientation,
            "image_keys": list(image_keys),
            "environment_config": environment_config,
            "stagnation_window": args.stagnation_window,
            "stationary_threshold_m": args.stationary_threshold,
            "small_action_threshold": args.small_action_threshold,
            "video_episodes": args.video_episodes,
        },
    )
    if run is not None:
        print(f"W&B dashboard: {run.url}", flush=True)

    episode_rows, seed_rows = [], []
    global_episode = 0
    videos_logged = 0
    original_render = getattr(collector, "RENDER", False)
    original_realtime = getattr(collector, "REALTIME_RENDER", False)
    trace_path = output / "bc_eval_steps.csv"
    # Stream each row so an interrupted evaluation still leaves useful traces.
    trace_file = trace_path.open("w", newline="", buffering=1)
    try:
        trace_writer = csv.DictWriter(trace_file, fieldnames=STEP_FIELDS)
        trace_writer.writeheader()
        collector.RENDER = bool(args.render)
        collector.REALTIME_RENDER = False
        for seed_index, seed in enumerate(seeds, start=1):
            seed_all(seed)
            env = collector.create_env(**environment_config)
            try:
                if hasattr(env, "rng"):
                    rng = env.rng
                    if hasattr(rng, "seed"):
                        rng.seed(seed)
                    elif isinstance(rng, np.random.Generator):
                        env.rng = np.random.default_rng(seed)
                        sampler = getattr(env, "placement_initializer", None)
                        if sampler is not None:

                            def reseed_sampler(node):
                                if hasattr(node, "rng"):
                                    node.rng = env.rng
                                for child in getattr(node, "samplers", []):
                                    reseed_sampler(child)

                            reseed_sampler(sampler)
                low, high = env.action_spec
                media_enabled = run is not None and args.log_videos
                third_camera = (
                    choose_third_person_camera(env, args.third_person_camera)
                    if media_enabled or local_video_enabled
                    else None
                )
                successes = 0
                print(
                    f"\n=== Seed {seed_index}/{len(seeds)}: {seed} | {args.episodes} episodes ===",
                    flush=True,
                )
                for ep in range(1, args.episodes + 1):
                    obs = env.reset()
                    hidden = None
                    previous_gripper = -1.0
                    global_episode += 1
                    # Spread recordings across seeds, starting with episode 1 of each seed.
                    capture_wandb = (
                        media_enabled
                        and videos_logged < args.video_episodes
                        and (ep == 1 or seed_index == len(seeds))
                    )
                    capture_local = (
                        local_video_enabled
                        and local_videos_recorded < args.video_episodes
                    )
                    capture = capture_wandb or capture_local
                    robot_frames, third_frames, eye_frames = [], [], []
                    initial_z = float(obs["cube_pos"][2])
                    held = 0
                    max_lift = 0.0
                    success = False
                    stationary_steps = 0
                    max_stationary_steps = 0
                    stagnation_observed = False
                    eef_deltas = deque(maxlen=args.stagnation_window)
                    action_norms = deque(maxlen=args.stagnation_window)
                    sum_action_norm = 0.0
                    sum_eef_delta = 0.0
                    last_action_norm = 0.0
                    last_eef_delta = 0.0
                    last_gripper = -1.0
                    episode_images = {}
                    episode_trace = EpisodeTrace(initial_z, args.small_action_threshold)
                    for step in range(args.max_steps):
                        if capture and step % args.video_stride == 0:
                            robot_frames.append(
                                np.flipud(np.asarray(obs[image_key])).copy()
                            )
                            third_frames.append(
                                render_third_person(env, third_camera, args.video_size)
                            )
                            if capture_local and EYE_IN_HAND_KEY in image_keys:
                                eye_frames.append(
                                    np.flipud(np.asarray(obs[EYE_IN_HAND_KEY])).copy()
                                )
                        if (
                            media_enabled
                            and step == 0
                            and (ep == 1 or seed_index == 1 and ep == 1)
                        ):
                            episode_images["start"] = np.flipud(
                                np.asarray(obs[image_key])
                            ).copy()
                        prev_eef = np.asarray(
                            obs["robot0_eef_pos"], dtype=np.float64
                        ).copy()
                        image = (
                            image_from_observation(obs, image_keys)
                            .unsqueeze(0)
                            .to(device)
                        )
                        state = np.concatenate([obs[k] for k in state_keys]).astype(
                            np.float32
                        )
                        state = (
                            torch.from_numpy((state - state_mean) / state_std)
                            .unsqueeze(0)
                            .to(device)
                        )
                        with torch.no_grad():
                            pred_motion, grip_logit, hidden = predict_step(
                                model, image, state, previous_gripper, hidden
                            )
                        grip_logit_value = grip_logit.item()
                        action = action_from_prediction(
                            pred_motion[0].cpu().numpy(), grip_logit_value, stats
                        )
                        action = np.clip(action, low, high)
                        previous_gripper = float(action[6])
                        last_action_norm = float(np.linalg.norm(action[:3]))
                        last_gripper = float(action[6])
                        trace_obs_before = {
                            key: np.asarray(obs[key]).copy()
                            for key in (
                                "robot0_eef_pos",
                                "robot0_gripper_qpos",
                                "cube_pos",
                            )
                        }
                        obs, _, _, _ = env.step(action)
                        trace_writer.writerow(
                            episode_trace.record(
                                seed=seed,
                                episode=ep,
                                global_episode=global_episode,
                                step=step + 1,
                                freq=freq,
                                obs_before=trace_obs_before,
                                action=action,
                                obs_after=obs,
                                gripper_logit=grip_logit_value,
                                grasp_contact_after=cube_grasp_contact(env),
                            )
                        )
                        last_eef_delta = float(
                            np.linalg.norm(
                                np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                                - prev_eef
                            )
                        )
                        sum_action_norm += last_action_norm
                        sum_eef_delta += last_eef_delta
                        action_norms.append(last_action_norm)
                        eef_deltas.append(last_eef_delta)
                        if last_eef_delta < args.stationary_threshold:
                            stationary_steps += 1
                        else:
                            stationary_steps = 0
                        max_stationary_steps = max(
                            max_stationary_steps, stationary_steps
                        )
                        if len(eef_deltas) == args.stagnation_window:
                            if float(np.mean(eef_deltas)) < args.stationary_threshold:
                                stagnation_observed = True
                                if "stagnation" not in episode_images and media_enabled:
                                    episode_images["stagnation"] = np.flipud(
                                        np.asarray(obs[image_key])
                                    ).copy()
                        if args.render:
                            env.render()
                            if args.realtime:
                                time.sleep(1.0 / freq)
                        if (
                            args.diagnostic_interval > 0
                            and (step + 1) % args.diagnostic_interval == 0
                        ):
                            print(
                                f"  step={step+1:03d} | action_xyz_norm={last_action_norm:.5f} | "
                                f"eef_delta_m={last_eef_delta:.6f} | gripper={last_gripper:+.0f} | "
                                f"stationary_steps={stationary_steps}",
                                flush=True,
                            )
                        lift = float(obs["cube_pos"][2]) - initial_z
                        max_lift = max(max_lift, lift)
                        held = (
                            held + 1
                            if lift >= lift_threshold and env._check_success()
                            else 0
                        )
                        if held >= hold_steps:
                            success = True
                            break
                    steps_taken = step + 1
                    successes += int(success)
                    termination = "success" if success else "max_steps"
                    row = {
                        "seed": seed,
                        "episode": ep,
                        "global_episode": global_episode,
                        "success": int(success),
                        "termination": termination,
                        "steps": steps_taken,
                        "max_lift_m": max_lift,
                        "mean_action_xyz_norm": sum_action_norm / steps_taken,
                        "mean_eef_displacement_m_per_step": sum_eef_delta / steps_taken,
                        "last_action_xyz_norm": last_action_norm,
                        "last_eef_displacement_m": last_eef_delta,
                        "last_gripper_command": last_gripper,
                        "max_consecutive_stationary_steps": max_stationary_steps,
                        "stagnation_observed": int(stagnation_observed),
                    }
                    row.update(episode_trace.summary())
                    episode_rows.append(row)
                    if capture_local and robot_frames:
                        fps = max(1, round(freq / args.video_stride))
                        stem = f"seed_{seed}_episode_{ep:03d}_{termination}"
                        recordings = [
                            ("agentview", robot_frames),
                            (third_camera, third_frames),
                        ]
                        if eye_frames:
                            recordings.append(("robot0_eye_in_hand", eye_frames))
                        for name, frames in recordings:
                            path = video_dir / f"{stem}_{name}.mp4"
                            with imageio.get_writer(
                                str(path), fps=fps, codec="libx264", macro_block_size=1
                            ) as writer:
                                for frame in frames:
                                    writer.append_data(frame)
                        local_videos_recorded += 1
                        print(f"  Videos: {video_dir / stem}_*.mp4", flush=True)
                    if run is not None:
                        metrics = {
                            "eval/seed": seed,
                            "eval/seed_index": seed_index,
                            "eval/episode_success": int(success),
                            "eval/max_lift_m": max_lift,
                            "eval/steps": steps_taken,
                            "eval/seed_cumulative_success_rate": successes / ep,
                            "eval/overall_cumulative_success_rate": sum(
                                r["success"] for r in episode_rows
                            )
                            / global_episode,
                            "diagnostics/mean_action_xyz_norm": row[
                                "mean_action_xyz_norm"
                            ],
                            "diagnostics/mean_eef_displacement_m_per_step": row[
                                "mean_eef_displacement_m_per_step"
                            ],
                            "diagnostics/max_consecutive_stationary_steps": max_stationary_steps,
                            "diagnostics/stagnation_observed": int(stagnation_observed),
                            "diagnostics/last_gripper_command": last_gripper,
                        }
                        if media_enabled:
                            import wandb

                            # Image panels are populated during evaluation, including failures.
                            if ep == 1 or (
                                not success and "failure" not in episode_images
                            ):
                                episode_images["final"] = np.flipud(
                                    np.asarray(obs[image_key])
                                ).copy()
                                for label, frame in episode_images.items():
                                    metrics[f"eval/images/{label}"] = wandb.Image(
                                        frame,
                                        caption=f"seed={seed} episode={ep} {label} "
                                        f"{termination} lift={max_lift:.3f}m",
                                    )
                            if capture_wandb and robot_frames:
                                try:

                                    def video(frames, caption):
                                        array = np.stack(frames).transpose(0, 3, 1, 2)
                                        return wandb.Video(
                                            array,
                                            fps=max(1, round(freq / args.video_stride)),
                                            format="mp4",
                                            caption=caption,
                                        )

                                    metrics["eval/robot_camera_video"] = video(
                                        robot_frames,
                                        f"Seed {seed} episode {ep} agentview",
                                    )
                                    metrics["eval/third_person_video"] = video(
                                        third_frames,
                                        f"Seed {seed} episode {ep} {third_camera}",
                                    )
                                    videos_logged += 1
                                except Exception as exc:
                                    print(
                                        f"WARNING: video encoding failed: {exc}. "
                                        'Install with python3 -m pip install "wandb[media]"',
                                        flush=True,
                                    )
                        run.log(metrics, step=global_episode)
                    print(
                        f"Seed {seed} | Episode {ep:02d}/{args.episodes} | "
                        f'{"SUCCESS" if success else "FAIL"} | steps={steps_taken} | '
                        f"max_lift={max_lift:.3f}m | seed_success={successes}/{ep} | "
                        f'action_norm={row["mean_action_xyz_norm"]:.5f} | '
                        f'eef_delta={row["mean_eef_displacement_m_per_step"]:.6f}m/step | '
                        f"stagnation={stagnation_observed}",
                        flush=True,
                    )
                rate = successes / args.episodes
                seed_rows.append(
                    {
                        "seed": seed,
                        "successes": successes,
                        "episodes": args.episodes,
                        "success_rate": rate,
                    }
                )
                if run is not None:
                    # Use summary for seed results; avoid duplicate W&B steps.
                    run.summary[f"eval/seed_{seed}_success_rate"] = rate
                print(
                    f">>> Seed {seed}: {successes}/{args.episodes} = {rate*100:.1f}%",
                    flush=True,
                )
            finally:
                env.close()

        rates = np.array([r["success_rate"] for r in seed_rows], dtype=np.float64)
        mean_rate = float(rates.mean())
        std_rate = float(rates.std(ddof=1)) if len(rates) > 1 else 0.0
        total_success = sum(r["successes"] for r in seed_rows)
        summary = {
            "algorithm": "BC",
            "policy_type": policy_type,
            "checkpoint": str(args.checkpoint),
            "predict_orientation": model.predict_orientation,
            "image_keys": list(image_keys),
            "environment_config": environment_config,
            "seeds": seeds,
            "episodes_per_seed": args.episodes,
            "total_episodes": len(episode_rows),
            "total_successes": total_success,
            "overall_success_rate": total_success / len(episode_rows),
            "local_videos_recorded": local_videos_recorded,
            "mean_seed_success_rate": mean_rate,
            "std_seed_success_rate_sample": std_rate,
            "seed_results": seed_rows,
            "lift_threshold_m": lift_threshold,
            "lift_hold_steps": hold_steps,
            "stagnation_episodes": sum(r["stagnation_observed"] for r in episode_rows),
            "step_trace_file": trace_path.name,
            "rise_event_threshold_m": RISE_THRESHOLD_M,
            "upward_action_threshold": args.small_action_threshold,
        }
        with (output / "bc_eval_episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0]))
            writer.writeheader()
            writer.writerows(episode_rows)
        with (output / "bc_eval_seeds.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
            writer.writeheader()
            writer.writerows(seed_rows)
        (output / "bc_eval_summary.json").write_text(json.dumps(summary, indent=2))
        print("\n" + "=" * 66)
        print(f"BC EVALUATION: {args.num_seeds} seeds x {args.episodes} episodes")
        for row in seed_rows:
            print(
                f'  Seed {row["seed"]}: {row["successes"]}/{row["episodes"]} = {100*row["success_rate"]:.1f}%'
            )
        print(
            f"Mean success rate: {mean_rate*100:.2f}% +/- {std_rate*100:.2f}% (sample SD across seeds)"
        )
        print(
            f'Overall: {total_success}/{len(episode_rows)} = {100*summary["overall_success_rate"]:.2f}%'
        )
        print(f"Results: {output.resolve()}")
        print(f"Per-step trace: {trace_path.resolve()}")
        print("=" * 66, flush=True)
        if run is not None:
            run.summary["eval/mean_success_rate"] = mean_rate
            run.summary["eval/std_success_rate"] = std_rate
            run.summary["eval/overall_success_rate"] = summary["overall_success_rate"]
            run.summary["eval/total_episodes"] = len(episode_rows)
            run.summary["eval/stagnation_episodes"] = summary["stagnation_episodes"]
            run.summary["eval/seed_results"] = seed_rows
            import wandb

            run.log(
                {
                    "eval/final_mean_success_rate": mean_rate,
                    "eval/final_std_success_rate": std_rate,
                    "eval/seed_results_table": wandb.Table(
                        columns=["seed", "successes", "episodes", "success_rate"],
                        data=[
                            [
                                r["seed"],
                                r["successes"],
                                r["episodes"],
                                r["success_rate"],
                            ]
                            for r in seed_rows
                        ],
                    ),
                },
                step=global_episode + 1,
            )
    finally:
        trace_file.close()
        collector.RENDER = original_render
        collector.REALTIME_RENDER = original_realtime
        if run is not None:
            run.finish()


def main():
    parser = argparse.ArgumentParser(
        description="BC training/evaluation for ThreeCubeLift"
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", default="datasets/three_cube_lift")
    t.add_argument("--output", default="checkpoints")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Windows per batch for --temporal (default 8); frames otherwise (default 128)",
    )
    t.add_argument(
        "--temporal",
        action="store_true",
        help="Train a causal GRU on sequences of observations and previous gripper commands",
    )
    t.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Steps per temporal training window (default 16)",
    )
    t.add_argument(
        "--sequence-stride",
        type=int,
        default=None,
        help="Steps between temporal training windows (default 8)",
    )
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--val-fraction", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=123)
    t.add_argument("--cpu", action="store_true")
    t.add_argument(
        "--resume",
        default=None,
        help="Resume training from bc_last.pt (or an older bc_best.pt); --epochs is the total target epoch",
    )
    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", default="checkpoints/bc_best.pt")
    e.add_argument("--episodes", type=int, default=20, help="Episodes per seed")
    e.add_argument("--num-seeds", type=int, default=5)
    e.add_argument("--seed-step", type=int, default=1)
    e.add_argument("--eval-output", default="reports/bc_evaluation")
    e.add_argument(
        "--collector-module",
        default="collect_demos",
        help="Module containing create_env",
    )
    e.add_argument("--render", action="store_true", help="Display live MuJoCo viewer")
    e.add_argument(
        "--realtime",
        action="store_true",
        help="Throttle live viewer to control frequency",
    )
    e.add_argument("--max-steps", type=int, default=500)
    e.add_argument("--stagnation-window", type=int, default=100)
    e.add_argument(
        "--stationary-threshold",
        type=float,
        default=0.0001,
        help="EEF displacement threshold in meters per simulation step",
    )
    e.add_argument(
        "--small-action-threshold",
        type=float,
        default=0.01,
        help="Positive action Z threshold for the first upward-command trace event",
    )
    e.add_argument(
        "--diagnostic-interval",
        type=int,
        default=100,
        help="Print policy/controller diagnostics every N steps; 0 disables",
    )
    e.add_argument("--seed", type=int, default=1000)
    e.add_argument("--cpu", action="store_true")
    e.add_argument(
        "--log-videos", action="store_true", help="Log both camera views to W&B"
    )
    e.add_argument(
        "--record-videos",
        action="store_true",
        help="Save agentview and third-person MP4s locally for the first --video-episodes rollouts",
    )
    e.add_argument(
        "--video-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to record",
    )
    e.add_argument(
        "--video-stride",
        type=int,
        default=2,
        help="Record one frame every N simulation steps",
    )
    e.add_argument(
        "--video-size", type=int, default=256, help="Third-person video resolution"
    )
    e.add_argument(
        "--third-person-camera",
        default="frontview",
        help="Preferred third-person camera",
    )
    for subparser in (t, e):
        subparser.add_argument(
            "--wandb", action="store_true", help="Enable W&B logging"
        )
        subparser.add_argument("--wandb-project", default="robosuite-project")
        subparser.add_argument(
            "--wandb-entity",
            default=None,
            help="Optional W&B team or username (default: logged-in account)",
        )
        subparser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    if args.mode == "eval" and args.video_stride < 1:
        parser.error("--video-stride must be at least 1")
    if args.mode == "eval" and args.video_episodes < 0:
        parser.error("--video-episodes must be nonnegative")
    if args.mode == "eval" and (
        args.episodes < 1 or args.num_seeds < 1 or args.seed_step < 1
    ):
        parser.error("--episodes, --num-seeds and --seed-step must be positive")
    if args.mode == "eval" and (
        args.stagnation_window < 1 or args.stationary_threshold <= 0
    ):
        parser.error("--stagnation-window and --stationary-threshold must be positive")
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
