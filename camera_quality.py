"""Catch broken offscreen camera frames before they enter BC datasets."""

import numpy as np


def suspicious_frames(images):
    """Indices of nearly black or high-frequency corrupted RGB frames."""
    frames = np.asarray(images)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected (T,H,W,3) images, got {frames.shape}")
    if len(frames) == 0:
        return np.empty(0, dtype=int)
    horizontal_change = np.abs(np.diff(frames.astype(np.int16), axis=2)).mean(
        axis=(1, 2, 3)
    )
    black_fraction = (frames == 0).mean(axis=(1, 2, 3))
    return np.flatnonzero((horizontal_change > 20.0) | (black_fraction > 0.95))
