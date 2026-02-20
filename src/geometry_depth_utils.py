"""Shared geometry/depth utility functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def mad_keep_mask(values: np.ndarray, thresh: float) -> np.ndarray:
    """Create keep mask using robust z-score (MAD)."""
    if thresh <= 0:
        return np.ones(values.shape[0], dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-8:
        return np.ones(values.shape[0], dtype=bool)
    z_score = 0.6745 * (values - median) / mad
    return np.abs(z_score) <= thresh


def build_filter_keep_mask(
    points: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    shape: tuple[int, int],
    border_margin: int,
    depth_mad: float,
    radius_mad: float,
) -> np.ndarray:
    """Build keep mask using border clipping + depth/radius MAD filtering."""
    keep = np.ones(points.shape[0], dtype=bool)
    if border_margin > 0 and ys.shape[0] == points.shape[0]:
        height, width = shape
        keep &= (
            (ys >= border_margin)
            & (ys < height - border_margin)
            & (xs >= border_margin)
            & (xs < width - border_margin)
        )

    filtered = points[keep]
    if filtered.size == 0:
        return np.zeros(points.shape[0], dtype=bool)

    if depth_mad > 0 or radius_mad > 0:
        sub_keep = np.ones(filtered.shape[0], dtype=bool)
        if depth_mad > 0:
            sub_keep &= mad_keep_mask(filtered[:, 2], depth_mad)
        if radius_mad > 0:
            center = np.median(filtered, axis=0)
            radius = np.linalg.norm(filtered - center, axis=1)
            sub_keep &= mad_keep_mask(radius, radius_mad)
        indices = np.where(keep)[0]
        keep_final = np.zeros(points.shape[0], dtype=bool)
        keep_final[indices[sub_keep]] = True
        return keep_final

    return keep


def backproject_depth(
    depth: np.ndarray, valid_mask: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Backproject depth map to camera-frame point cloud on valid pixels."""
    ys, xs = np.where(valid_mask)
    z = depth[valid_mask].astype(np.float32)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def save_points_ply(points: np.ndarray, path: Path) -> None:
    """Save (N,3) points as an ASCII PLY file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points.astype(np.float32)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        if points.shape[0] > 0:
            np.savetxt(f, points, fmt="%.6f %.6f %.6f")
