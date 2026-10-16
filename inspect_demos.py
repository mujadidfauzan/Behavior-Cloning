#!/usr/bin/env python3
"""Inspect combined or per-episode Robosuite three-cube HDF5 demonstrations."""

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from camera_quality import suspicious_frames

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DATASET = "datasets/three_cube_lift"
DEFAULT_OUTPUT = "reports/demo_inspection"
EXPECTED_ACTION_DIM = 7
EXPECTED_IMAGE_SHAPE = (128, 128, 3)


def iter_demo_locations(path):
    files = sorted(path.glob("demo_*.hdf5")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No demo_*.hdf5 files found in {path}")
    for file in files:
        try:
            with h5py.File(file, "r") as h5:
                if "data" not in h5:
                    yield file, None, "Missing /data group"
                else:
                    names = sorted(n for n in h5["data"] if n.startswith("demo_"))
                    if not names:
                        yield file, None, "No demo groups found"
                    for name in names:
                        yield file, name, None
        except (OSError, ValueError) as exc:
            yield file, None, str(exc)


def inspect_demo(group, name, file, threshold):
    problems = []
    actions = group["actions"][:] if "actions" in group else np.empty((0, 0))
    t = len(actions)
    if actions.ndim != 2 or actions.shape[1] != EXPECTED_ACTION_DIM:
        problems.append(f"Action shape {actions.shape}; expected (T,7)")
    if t == 0 or not np.isfinite(actions).all():
        problems.append("Actions empty or contain NaN/Inf")
    if actions.size and (np.min(actions) < -1.001 or np.max(actions) > 1.001):
        problems.append("Action values outside expected normalized [-1,1] range")
    for key in ("rewards", "dones"):
        if key not in group or len(group[key]) != t:
            problems.append(f"{key} missing or length != {t}")
    if "dones" in group and len(group["dones"]):
        d = group["dones"][:]
        if not bool(d[-1]) or np.any(d[:-1]):
            problems.append("Terminal flags are inconsistent")
    camera = {}
    for group_name in ("obs", "next_obs"):
        if group_name not in group:
            problems.append(f"Missing {group_name}")
            continue
        for key, ds in group[group_name].items():
            if not isinstance(ds, h5py.Dataset):
                continue
            if ds.shape[0] != t:
                problems.append(f"{group_name}/{key} length {ds.shape[0]} != {t}")
            if key.endswith("_image"):
                camera[f"{group_name}/{key}"] = (tuple(ds.shape), str(ds.dtype))
                if (
                    ds.ndim != 4
                    or tuple(ds.shape[1:]) != EXPECTED_IMAGE_SHAPE
                    or ds.dtype != np.uint8
                ):
                    problems.append(
                        f"Unexpected camera format: {group_name}/{key} {ds.shape} {ds.dtype}"
                    )
                if t and ds.ndim == 4:
                    indices = sorted({0, t // 2, t - 1})
                    frames = []
                    for idx in indices:
                        frame = ds[idx]
                        frames.append(frame)
                        if not np.isfinite(frame).all() or np.ptp(frame) == 0:
                            problems.append(
                                f"Invalid or constant camera frame: {key}[{idx}]"
                            )
                    if ds.shape[-1] == 3:
                        for bad in suspicious_frames(np.stack(frames)):
                            problems.append(f"Corrupted camera frame: {key}[{indices[bad]}]")
    if not any(k.startswith("obs/") for k in camera):
        problems.append("No camera images in obs")
    final_lift = max_lift = float("nan")
    if (
        t
        and all(k in group for k in ("obs", "next_obs"))
        and "cube_pos" in group["obs"]
        and "cube_pos" in group["next_obs"]
    ):
        initial_z = float(group["obs/cube_pos"][0, 2])
        heights = group["next_obs/cube_pos"][:, 2] - initial_z
        final_lift, max_lift = float(heights[-1]), float(np.max(heights))
        if final_lift + 1e-5 < threshold:
            problems.append(f"Final lift {final_lift:.4f} m < {threshold:.4f} m")
        stored = float(group.attrs.get("final_lift", np.nan))
        if np.isfinite(stored) and abs(stored - final_lift) > 0.005:
            problems.append(
                f"Stored final lift {stored:.4f} differs from reconstructed {final_lift:.4f}"
            )
    else:
        problems.append("Cannot reconstruct cube lift")
    positions = {}
    for key in ("initial_cube", "initial_blue", "initial_green", "initial_joints"):
        if key in group:
            positions[key] = np.asarray(group[key][:], dtype=float)
        else:
            problems.append(f"Missing {key} (may be an older dataset)")
    return dict(
        name=name,
        file=str(file),
        steps=t,
        action_dim=actions.shape[1] if actions.ndim == 2 else -1,
        final_lift=final_lift,
        max_lift=max_lift,
        positions=positions,
        cameras=camera,
        issues=problems,
    )


def save_previews(path, name, output, flip=True):
    with h5py.File(path, "r") as h5:
        obs = h5["data"][name]["obs"]
        for key in obs:
            if not key.endswith("_image"):
                continue
            ds = obs[key]
            if ds.ndim != 4 or len(ds) == 0:
                continue
            indices = np.linspace(0, len(ds) - 1, 5, dtype=int)
            fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
            for ax, i in zip(axes, indices):
                frame = ds[int(i)]
                ax.imshow(np.flipud(frame) if flip else frame)
                ax.set_title(f"Step {i}")
                ax.axis("off")
            fig.suptitle(f"{name} — {key}")
            fig.tight_layout()
            preview_dir = output / "camera_previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview_dir / f"{name}_{key}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def summary_stats(values):
    a = np.asarray(values, dtype=float)
    return f"mean={np.mean(a):.4f}, min={np.min(a):.4f}, max={np.max(a):.4f}, std={np.std(a):.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Directory of per-demo HDF5 files, or a combined HDF5 file",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-lift",
        type=float,
        default=None,
        help="Override minimum lift in meters; otherwise read file metadata",
    )
    parser.add_argument("--previews", type=int, default=5)
    parser.add_argument("--no-flip", action="store_true")
    args = parser.parse_args()
    path, output = Path(args.dataset), Path(args.output)
    if not path.exists():
        parser.error(f"Dataset does not exist: {path}")
    output.mkdir(parents=True, exist_ok=True)
    results, errors, thresholds = [], [], []
    for file, name, error in iter_demo_locations(path):
        if error:
            errors.append(f"{file}: {error}")
            print("ERROR:", errors[-1])
            continue
        try:
            with h5py.File(file, "r") as h5:
                threshold = (
                    float(args.min_lift)
                    if args.min_lift is not None
                    else float(h5.attrs.get("target_lift_m", 0.20))
                    - float(h5.attrs.get("lift_tolerance_m", 0.02))
                )
                result = inspect_demo(h5["data"][name], name, file, threshold)
                result["threshold"] = threshold
                results.append(result)
                thresholds.append(threshold)
                status = "PASS" if not result["issues"] else "CHECK"
                print(
                    f"{name}: {status} | T={result['steps']} | action={result['action_dim']}D | lift={result['final_lift']:.4f}m | min={threshold:.4f}m"
                )
                for issue in result["issues"]:
                    print("   -", issue)
        except (OSError, KeyError, ValueError, IndexError) as exc:
            errors.append(f"{file}/{name}: {exc}")
            print("ERROR:", errors[-1])
    if not results:
        raise RuntimeError("No readable demonstrations found")
    for r in results[: max(args.previews, 0)]:
        try:
            save_previews(Path(r["file"]), r["name"], output, not args.no_flip)
        except Exception as exc:
            errors.append(f"Preview {r['name']}: {exc}")
    names = [r["name"] for r in results]
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(max(9, min(18, len(results) * 0.16)), 4))
    ax.plot(
        x,
        [r["steps"] for r in results],
        marker=".",
        linestyle="none" if len(results) > 50 else "-",
    )
    ax.set(xlabel="Demonstration index", ylabel="Steps", title="Trajectory lengths")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "trajectory_lengths.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, [r["final_lift"] for r in results], ".", label="Final lift")
    ax.plot(x, [r["max_lift"] for r in results], "x", label="Maximum lift")
    ax.plot(x, thresholds, "--", label="Minimum lift (metadata / override)")
    ax.set(
        xlabel="Demonstration index", ylabel="Lift (m)", title="Red cube lift heights"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "lift_heights.png", dpi=150)
    plt.close(fig)
    lines = [
        f"Dataset: {path}",
        f"Demonstrations: {len(results)}",
        f'Total transitions: {sum(r["steps"] for r in results)}',
        f'Valid demonstrations: {sum(not r["issues"] for r in results)}',
        f'Demonstrations with issues: {sum(bool(r["issues"]) for r in results)}',
        f'Trajectory steps: {summary_stats([r["steps"] for r in results])}',
        f'Final lift (m): {summary_stats([r["final_lift"] for r in results])}',
        f'Max lift (m): {summary_stats([r["max_lift"] for r in results])}',
        f"Minimum lift threshold (m): {summary_stats(thresholds)}",
        "",
        "INITIAL POSITION RANDOMIZATION",
    ]
    for key, label, unit in [
        ("initial_cube", "Red cube XYZ", "m"),
        ("initial_blue", "Blue cube XYZ", "m"),
        ("initial_green", "Green cube XYZ", "m"),
        ("initial_joints", "Panda joints", "deg"),
    ]:
        values = [r["positions"][key] for r in results if key in r["positions"]]
        if not values:
            lines.append(f"{label}: unavailable")
            continue
        arr = np.asarray(values)
        if key == "initial_joints":
            arr = np.rad2deg(arr)
        lines.append(f"{label} ({unit}), N={len(arr)}:")
        lines.append(f"  Mean: {np.array2string(arr.mean(axis=0), precision=4)}")
        lines.append(f"  Std:  {np.array2string(arr.std(axis=0), precision=4)}")
        lines.append(f"  Min:  {np.array2string(arr.min(axis=0), precision=4)}")
        lines.append(f"  Max:  {np.array2string(arr.max(axis=0), precision=4)}")
    lines += ["", "PER-EPISODE RESULTS"]
    for r in results:
        lines.append(
            f"{r['name']}: {'PASS' if not r['issues'] else 'CHECK'} | steps={r['steps']} | action={r['action_dim']}D | final_lift={r['final_lift']:.4f}m | max_lift={r['max_lift']:.4f}m"
        )
        for issue in r["issues"]:
            lines.append("  - " + issue)
    if errors:
        lines += ["", "FILE / PREVIEW ERRORS"] + errors
    report = "\n".join(lines) + "\n"
    (output / "inspection_report.txt").write_text(report)
    print("\n" + report)
    print("Reports saved to:", output)


if __name__ == "__main__":
    main()
