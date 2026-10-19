#!/usr/bin/env python3
"""Record exactly N successful BC rollouts locally, ignoring failed attempts.

Requires bc_baseline.py and collect_demos.py in the project directory.
MP4 encoding: python3 -m pip install imageio imageio-ffmpeg
"""

import argparse
import csv
import importlib
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description="Record successful BC policy rollouts only")
    p.add_argument("--checkpoint", default="checkpoints/bc_best.pt")
    p.add_argument("--successes", type=int, default=3)
    p.add_argument("--max-attempts", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--output", default="videos/bc_successes")
    p.add_argument("--collector-module", default="collect_demos")
    p.add_argument("--third-person-camera", default="frontview")
    p.add_argument("--video-size", type=int, default=256)
    p.add_argument("--video-stride", type=int, default=2)
    p.add_argument("--render", action="store_true", help="Show live MuJoCo viewer")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if (
        min(
            args.successes,
            args.max_attempts,
            args.max_steps,
            args.video_stride,
            args.video_size,
        )
        < 1
    ):
        p.error("Counts, sizes, and stride must be positive")

    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401 - verifies MP4 backend is available
    except ImportError as exc:
        raise SystemExit(
            "Install MP4 dependencies: python3 -m pip install imageio imageio-ffmpeg"
        ) from exc

    import bc_baseline as bc

    collector = importlib.import_module(args.collector_module)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_keys = bc.image_keys_from_checkpoint(checkpoint)
    model = bc.policy_from_checkpoint(checkpoint, device)
    stats = checkpoint["stats"]
    state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
    state_std = np.asarray(stats["state_std"], dtype=np.float32)
    environment_config = bc.checkpoint_environment_config(checkpoint)
    image_key = image_keys[0]
    state_keys = tuple(
        checkpoint.get(
            "state_keys", ("robot0_joint_pos", "robot0_eef_pos", "robot0_gripper_qpos")
        )
    )
    threshold = float(collector.LIFT_HEIGHT - collector.LIFT_TOLERANCE)
    hold_steps = int(collector.LIFT_HOLD_STEPS)
    fps = max(1, round(collector.CONTROL_FREQ / args.video_stride))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Preserve the collector's default behavior unless live rendering is requested.
    old_render = getattr(collector, "RENDER", None)
    if args.render:
        collector.RENDER = True
    successes = 0
    rows = []
    try:
        for attempt in range(1, args.max_attempts + 1):
            if successes >= args.successes:
                break
            seed = args.seed + attempt - 1
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            env = collector.create_env(**environment_config)
            try:
                # Match per-attempt seed to the environment's placement RNG.
                if hasattr(env, "rng"):
                    if hasattr(env.rng, "seed"):
                        env.rng.seed(seed)
                    elif isinstance(env.rng, np.random.Generator):
                        env.rng = np.random.default_rng(seed)
                    sampler = getattr(env, "placement_initializer", None)

                    def propagate_rng(node):
                        if hasattr(node, "rng"):
                            node.rng = env.rng
                        for child in getattr(node, "samplers", []):
                            propagate_rng(child)

                    if sampler is not None:
                        propagate_rng(sampler)
                obs = env.reset()
                hidden = None
                previous_gripper = -1.0
                camera = bc.choose_third_person_camera(env, args.third_person_camera)
                low, high = env.action_spec
                initial_z = float(obs["cube_pos"][2])
                max_lift = 0.0
                held = 0
                success = False
                # Write each attempt to temporary MP4s; retain files ONLY on success.
                with tempfile.TemporaryDirectory(prefix="bc_rollout_") as tmp:
                    robot_tmp = Path(tmp) / "agentview.mp4"
                    third_tmp = Path(tmp) / "third_person.mp4"
                    with imageio.get_writer(
                        str(robot_tmp), fps=fps, codec="libx264", macro_block_size=1
                    ) as rw, imageio.get_writer(
                        str(third_tmp), fps=fps, codec="libx264", macro_block_size=1
                    ) as tw:
                        start_frame = np.flipud(np.asarray(obs[image_key])).copy()
                        for step in range(args.max_steps):
                            if step % args.video_stride == 0:
                                rw.append_data(
                                    np.flipud(np.asarray(obs[image_key])).copy()
                                )
                                tw.append_data(
                                    bc.render_third_person(env, camera, args.video_size)
                                )
                            image = bc.image_from_observation(
                                obs, image_keys
                            ).unsqueeze(0).to(device)
                            state = np.concatenate([obs[k] for k in state_keys]).astype(
                                np.float32
                            )
                            state = (
                                torch.from_numpy((state - state_mean) / state_std)
                                .unsqueeze(0)
                                .to(device)
                            )
                            with torch.no_grad():
                                motion, grip, hidden = bc.predict_step(
                                    model, image, state, previous_gripper, hidden
                                )
                            action = bc.action_from_prediction(
                                motion[0].cpu().numpy(), grip.item(), stats
                            )
                            action = np.clip(action, low, high)
                            previous_gripper = float(action[6])
                            obs, _, _, _ = env.step(action)
                            if args.render:
                                env.render()
                            lift = float(obs["cube_pos"][2]) - initial_z
                            max_lift = max(max_lift, lift)
                            held = (
                                held + 1
                                if lift >= threshold and env._check_success()
                                else 0
                            )
                            if held >= hold_steps:
                                success = True
                                # Include final success state in both videos.
                                rw.append_data(
                                    np.flipud(np.asarray(obs[image_key])).copy()
                                )
                                tw.append_data(
                                    bc.render_third_person(env, camera, args.video_size)
                                )
                                break
                    row = {
                        "attempt": attempt,
                        "seed": seed,
                        "success": int(success),
                        "steps": step + 1,
                        "max_lift_m": round(max_lift, 5),
                    }
                    rows.append(row)
                    if success:
                        successes += 1
                        prefix = f"success_{successes:02d}_seed_{seed}"
                        robot_path = output / f"{prefix}_agentview.mp4"
                        third_path = output / f"{prefix}_{camera}.mp4"
                        robot_path.write_bytes(robot_tmp.read_bytes())
                        third_path.write_bytes(third_tmp.read_bytes())
                        # Save representative frames alongside each successful video.
                        import imageio.v2 as iio

                        iio.imwrite(output / f"{prefix}_start.png", start_frame)
                        iio.imwrite(
                            output / f"{prefix}_final.png",
                            np.flipud(np.asarray(obs[image_key])).copy(),
                        )
                        print(
                            f"[{attempt}/{args.max_attempts}] SUCCESS {successes}/{args.successes} "
                            f"({step+1} steps, lift {max_lift:.3f}m) -> {robot_path}, {third_path}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{attempt}/{args.max_attempts}] FAIL (steps={step+1}, lift={max_lift:.3f}m); retrying",
                            flush=True,
                        )
            finally:
                env.close()
    finally:
        if old_render is not None:
            collector.RENDER = old_render
        with (output / "attempts.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["attempt", "seed", "success", "steps", "max_lift_m"]
            )
            writer.writeheader()
            writer.writerows(rows)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "requested_successes": args.successes,
                    "recorded_successes": successes,
                    "attempts": len(rows),
                    "attempt_success_rate": successes / len(rows) if rows else None,
                    "success_threshold_m": threshold,
                    "hold_steps": hold_steps,
                    "predict_orientation": model.predict_orientation,
                    "policy_type": checkpoint.get("policy_type", "frame"),
                    "environment_config": environment_config,
                },
                indent=2,
            )
        )
    print(
        f"Finished: recorded {successes}/{args.successes} successful rollouts in {output}",
        flush=True,
    )
    if successes < args.successes:
        raise SystemExit(
            "Maximum attempts reached before collecting enough successful videos"
        )


if __name__ == "__main__":
    main()
