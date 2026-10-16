#!/usr/bin/env python3
"""Train or evaluate a vision+proprioception BC baseline on per-episode HDF5 demos.

Training: python3 bc_baseline.py train --dataset datasets/three_cube_lift
Evaluation: python3 bc_baseline.py eval --checkpoint checkpoints/bc_best.pt --episodes 50
Run from the directory containing collect_demos.py for evaluation.
"""

import argparse
import bisect
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


def log_training_camera_samples(run, paths, max_images=6):
    """Show held-out demonstration camera frames on the training dashboard."""
    if run is None:
        return
    import wandb

    samples = []
    for path in paths[:max_images]:
        with h5py.File(path, "r") as f:
            demo = f["data"][next(iter(f["data"]))]
            images = demo["obs"][IMAGE_KEY]
            index = len(images) // 2
            # Robosuite camera observations are vertically flipped for display.
            rgb = np.flipud(np.asarray(images[index])).copy()
            samples.append(
                wandb.Image(rgb, caption=f"{path.stem} | step {index} | validation")
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
STATE_KEYS = ("robot0_joint_pos", "robot0_eef_pos", "robot0_gripper_qpos")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_demo_info(path):
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys())
        if len(names) != 1:
            raise ValueError(f"{path}: expected exactly one demo group, found {names}")
        g = f["data"][names[0]]
        t = len(g["actions"])
        for key in STATE_KEYS + (IMAGE_KEY,):
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
        if g["obs"][IMAGE_KEY].shape[1:] != (128, 128, 3):
            raise ValueError(f"{path}: expected 128x128 RGB images")
        state_dim = sum(g["obs"][k].shape[1] for k in STATE_KEYS)
        return t, state_dim


class EpisodeDataset(Dataset):
    def __init__(self, paths, state_mean, state_std, xyz_mean, xyz_std):
        self.paths = [str(p) for p in paths]
        self.lengths = [read_demo_info(p)[0] for p in self.paths]
        self.ends = np.cumsum(self.lengths).tolist()
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.tensor(state_std, dtype=torch.float32)
        self.xyz_mean = torch.tensor(xyz_mean, dtype=torch.float32)
        self.xyz_std = torch.tensor(xyz_std, dtype=torch.float32)
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
        image = (
            torch.from_numpy(np.asarray(g["obs"][IMAGE_KEY][i]).copy())
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        state = np.concatenate([g["obs"][key][i] for key in STATE_KEYS]).astype(
            np.float32
        )
        state = (torch.from_numpy(state) - self.state_mean) / self.state_std
        action = np.asarray(g["actions"][i], dtype=np.float32)
        xyz = (torch.from_numpy(action[:3].copy()) - self.xyz_mean) / self.xyz_std
        gripper = torch.tensor([float(action[6] > 0)], dtype=torch.float32)
        return image, state, xyz, gripper

    def __del__(self):
        for f in getattr(self, "handles", {}).values():
            try:
                f.close()
            except Exception:
                pass


def compute_stats(paths):
    state_sum = state_sq = xyz_sum = xyz_sq = None
    count = 0
    for path in paths:
        with h5py.File(path, "r") as f:
            g = f["data"][next(iter(f["data"].keys()))]
            state = np.concatenate(
                [g["obs"][key][:] for key in STATE_KEYS], axis=1
            ).astype(np.float64)
            xyz = g["actions"][:, :3].astype(np.float64)
            if state_sum is None:
                state_sum = np.zeros(state.shape[1])
                state_sq = np.zeros(state.shape[1])
                xyz_sum = np.zeros(3)
                xyz_sq = np.zeros(3)
            state_sum += state.sum(0)
            state_sq += (state**2).sum(0)
            xyz_sum += xyz.sum(0)
            xyz_sq += (xyz**2).sum(0)
            count += len(xyz)
    sm = state_sum / count
    ss = np.maximum(np.sqrt(np.maximum(state_sq / count - sm**2, 0)), 1e-3)
    xm = xyz_sum / count
    xs = np.maximum(np.sqrt(np.maximum(xyz_sq / count - xm**2, 0)), 1e-2)
    return dict(
        state_mean=sm.tolist(),
        state_std=ss.tolist(),
        xyz_mean=xm.tolist(),
        xyz_std=xs.tolist(),
    )


class BCPolicy(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
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
        self.head = nn.Sequential(
            nn.Linear(256 + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, image, state):
        out = self.head(torch.cat([self.vision(image), state], dim=1))
        return out[:, :3], out[:, 3:4]


def epoch_pass(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_xyz = total_grip = n = 0
    for image, state, xyz, gripper in loader:
        image, state, xyz, gripper = [
            x.to(device) for x in (image, state, xyz, gripper)
        ]
        with torch.set_grad_enabled(training):
            pred_xyz, grip_logit = model(image, state)
            xyz_loss = nn.functional.mse_loss(pred_xyz, xyz)
            grip_loss = nn.functional.binary_cross_entropy_with_logits(
                grip_logit, gripper
            )
            loss = xyz_loss + 0.2 * grip_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        b = len(image)
        total_loss += loss.item() * b
        total_xyz += xyz_loss.item() * b
        total_grip += grip_loss.item() * b
        n += b
    return total_loss / n, total_xyz / n, total_grip / n


def train(args):
    seed_all(args.seed)
    paths = sorted(Path(args.dataset).glob("demo_*.hdf5"))
    if len(paths) < 2:
        raise RuntimeError(
            f"Need at least two per-episode HDF5 files in {args.dataset}"
        )
    random.Random(args.seed).shuffle(paths)
    n_val = max(1, round(len(paths) * args.val_fraction))
    val_paths, train_paths = paths[:n_val], paths[n_val:]
    state_dims = {read_demo_info(p)[1] for p in paths}
    if len(state_dims) != 1:
        raise RuntimeError(f"Inconsistent state dimensions: {state_dims}")
    stats = compute_stats(train_paths)
    train_set = EpisodeDataset(train_paths, **stats)
    val_set = EpisodeDataset(val_paths, **stats)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model = BCPolicy(state_dims.pop()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "split.json").write_text(
        json.dumps(
            {
                "train": [p.name for p in train_paths],
                "val": [p.name for p in val_paths],
            },
            indent=2,
        )
    )
    run = init_wandb(
        args,
        "train",
        {
            "algorithm": "BC",
            "dataset": str(args.dataset),
            "train_episodes": len(train_paths),
            "val_episodes": len(val_paths),
            "train_transitions": len(train_set),
            "val_transitions": len(val_set),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "image_key": IMAGE_KEY,
            "state_keys": list(STATE_KEYS),
            "action_dim": 7,
            "device": str(device),
        },
    )
    if run is not None:
        print(f"W&B dashboard: {run.url}", flush=True)
        log_training_camera_samples(run, val_paths)
    best = float("inf")
    print(
        f"Train: {len(train_paths)} episodes/{len(train_set)} transitions | "
        f"Validation: {len(val_paths)} episodes/{len(val_set)} transitions | Device: {device}",
        flush=True,
    )
    try:
        for epoch in range(1, args.epochs + 1):
            tr = epoch_pass(model, train_loader, device, optimizer)
            va = epoch_pass(model, val_loader, device)
            improved = va[0] < best
            if improved:
                best = va[0]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "stats": stats,
                        "state_dim": len(stats["state_mean"]),
                        "epoch": epoch,
                        "best_val_loss": best,
                        "state_keys": STATE_KEYS,
                        "image_key": IMAGE_KEY,
                    },
                    output / "bc_best.pt",
                )
            print(
                f"Epoch {epoch:03d}/{args.epochs}: train={tr[0]:.5f} "
                f"val={va[0]:.5f} (xyz={va[1]:.5f}, gripper={va[2]:.5f}) "
                f"best={best:.5f}",
                flush=True,
            )
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        "train/loss": tr[0],
                        "train/xyz_mse": tr[1],
                        "train/gripper_bce": tr[2],
                        "val/loss": va[0],
                        "val/xyz_mse": va[1],
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
    model = BCPolicy(checkpoint["state_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = checkpoint["stats"]
    state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
    state_std = np.asarray(stats["state_std"], dtype=np.float32)
    xyz_mean = np.asarray(stats["xyz_mean"], dtype=np.float32)
    xyz_std = np.asarray(stats["xyz_std"], dtype=np.float32)
    lift_threshold = float(collector.LIFT_HEIGHT - collector.LIFT_TOLERANCE)
    hold_steps = int(collector.LIFT_HOLD_STEPS)
    freq = int(collector.CONTROL_FREQ)
    seeds = [args.seed + i * args.seed_step for i in range(args.num_seeds)]
    output = Path(args.eval_output)
    output.mkdir(parents=True, exist_ok=True)

    run = init_wandb(
        args,
        "eval",
        {
            "algorithm": "BC",
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
    try:
        collector.RENDER = bool(args.render)
        collector.REALTIME_RENDER = False
        for seed_index, seed in enumerate(seeds, start=1):
            seed_all(seed)
            env = collector.create_env()
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
                    if media_enabled
                    else None
                )
                successes = 0
                print(
                    f"\n=== Seed {seed_index}/{len(seeds)}: {seed} | {args.episodes} episodes ===",
                    flush=True,
                )
                for ep in range(1, args.episodes + 1):
                    obs = env.reset()
                    global_episode += 1
                    # Spread recordings across seeds, starting with episode 1 of each seed.
                    capture = (
                        media_enabled
                        and videos_logged < args.video_episodes
                        and (ep == 1 or seed_index == len(seeds))
                    )
                    robot_frames, third_frames = [], []
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
                    for step in range(args.max_steps):
                        if capture and step % args.video_stride == 0:
                            robot_frames.append(
                                np.flipud(np.asarray(obs[IMAGE_KEY])).copy()
                            )
                            third_frames.append(
                                render_third_person(env, third_camera, args.video_size)
                            )
                        if (
                            media_enabled
                            and step == 0
                            and (ep == 1 or seed_index == 1 and ep == 1)
                        ):
                            episode_images["start"] = np.flipud(
                                np.asarray(obs[IMAGE_KEY])
                            ).copy()
                        prev_eef = np.asarray(
                            obs["robot0_eef_pos"], dtype=np.float64
                        ).copy()
                        image = (
                            torch.from_numpy(np.asarray(obs[IMAGE_KEY]).copy())
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .float()
                            .to(device)
                            / 255.0
                        )
                        state = np.concatenate([obs[k] for k in STATE_KEYS]).astype(
                            np.float32
                        )
                        state = (
                            torch.from_numpy((state - state_mean) / state_std)
                            .unsqueeze(0)
                            .to(device)
                        )
                        with torch.no_grad():
                            pred_xyz, grip_logit = model(image, state)
                        action = np.zeros(7, dtype=np.float32)
                        action[:3] = pred_xyz[0].cpu().numpy() * xyz_std + xyz_mean
                        action[6] = 1.0 if grip_logit.item() > 0 else -1.0
                        action = np.clip(action, low, high)
                        last_action_norm = float(np.linalg.norm(action[:3]))
                        last_gripper = float(action[6])
                        obs, _, _, _ = env.step(action)
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
                                        np.asarray(obs[IMAGE_KEY])
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
                    episode_rows.append(row)
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
                                    np.asarray(obs[IMAGE_KEY])
                                ).copy()
                                for label, frame in episode_images.items():
                                    metrics[f"eval/images/{label}"] = wandb.Image(
                                        frame,
                                        caption=f"seed={seed} episode={ep} {label} "
                                        f"{termination} lift={max_lift:.3f}m",
                                    )
                            if capture and robot_frames:
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
            "checkpoint": str(args.checkpoint),
            "seeds": seeds,
            "episodes_per_seed": args.episodes,
            "total_episodes": len(episode_rows),
            "total_successes": total_success,
            "overall_success_rate": total_success / len(episode_rows),
            "mean_seed_success_rate": mean_rate,
            "std_seed_success_rate_sample": std_rate,
            "seed_results": seed_rows,
            "lift_threshold_m": lift_threshold,
            "lift_hold_steps": hold_steps,
            "stagnation_episodes": sum(r["stagnation_observed"] for r in episode_rows),
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
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--val-fraction", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=123)
    t.add_argument("--cpu", action="store_true")
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
    e.add_argument("--small-action-threshold", type=float, default=0.01)
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
