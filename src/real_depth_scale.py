"""Real depth based masked pointcloud/stat export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from camera_intrinsics import load_intrinsics_tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build masked pointcloud/stat outputs from real depth image."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--depth-image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-full-ply", type=Path, required=True)
    parser.add_argument("--cam-k", type=Path, default=None)
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--depth-scale", type=str, default="auto")
    parser.add_argument("--border-margin", type=int, default=5)
    parser.add_argument("--depth-mad", type=float, default=2.5)
    parser.add_argument("--radius-mad", type=float, default=2.5)
    parser.add_argument("--min-points", type=int, default=500)
    return parser.parse_args()


def mad_keep_mask(values: np.ndarray, thresh: float) -> np.ndarray:
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


def compute_scale(points_np: np.ndarray) -> dict:
    mins = points_np.min(axis=0)
    maxs = points_np.max(axis=0)
    size = maxs - mins
    diag = float(np.linalg.norm(size))
    max_dim = float(size.max())
    return {
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
        "bbox_size": size.tolist(),
        "bbox_diag": diag,
        "bbox_max_dim": max_dim,
        "scale_method": "bbox_diag",
    }


def save_points_ply(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for px, py, pz in points:
            f.write(f"{px} {py} {pz}\n")


def main() -> int:
    args = parse_args()

    image_path = args.image.resolve()
    depth_path = args.depth_image.resolve()
    mask_path = args.mask.resolve()
    output_npz = args.output_npz.resolve()
    output_json = args.output_json.resolve()
    output_ply = args.output_ply.resolve()
    output_full_ply = args.output_full_ply.resolve()
    cam_k_path = args.cam_k.resolve() if args.cam_k is not None else None

    if not depth_path.exists():
        print(f"Missing depth image: {depth_path}")
        return 1
    if not mask_path.exists():
        print(f"Missing mask: {mask_path}")
        return 1
    if cam_k_path is not None and not cam_k_path.exists():
        print(f"Missing camera intrinsics: {cam_k_path}")
        return 1

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        print(f"Failed to read depth image: {depth_path}")
        return 1
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    depth = depth_raw.astype(np.float32)

    depth_scale_raw = str(args.depth_scale).strip().lower()
    if depth_scale_raw == "auto":
        if depth_raw.dtype.kind in ("u", "i") and float(np.max(depth_raw)) > 50:
            depth_scale_value = 0.001
        elif float(np.max(depth)) > 20 and float(np.max(depth)) < 10000:
            depth_scale_value = 0.001
        else:
            depth_scale_value = 1.0
    else:
        depth_scale_value = float(args.depth_scale)
    depth *= float(depth_scale_value)

    if cam_k_path is not None:
        fx, fy, cx, cy = load_intrinsics_tuple(cam_k_path)
    else:
        fx = fy = cx = cy = None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"Failed to read mask: {mask_path}")
        return 1
    if mask.ndim == 3:
        mask = mask[:, :, -1]
    mask = mask > 0

    if mask.shape != depth.shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    valid = mask & np.isfinite(depth) & (depth > 0)
    if int(valid.sum()) < args.min_pixels:
        print(f"Not enough valid pixels: {int(valid.sum())}")
        return 1

    ys, xs = np.where(valid)
    depth_masked = depth[valid]

    height, width = depth.shape
    if fx is None or fy is None:
        fx = fy = float(max(height, width))
    if cx is None or cy is None:
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0

    z = depth_masked
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    points_count_raw = int(points.shape[0])

    valid_full = np.isfinite(depth) & (depth > 0)
    ys_full, xs_full = np.where(valid_full)
    z_full = depth[valid_full]
    x_full = (xs_full - cx) * z_full / fx
    y_full = (ys_full - cy) * z_full / fy
    points_full = np.stack([x_full, y_full, z_full], axis=1).astype(np.float32)

    keep = build_filter_keep_mask(
        points,
        ys,
        xs,
        depth.shape,
        args.border_margin,
        args.depth_mad,
        args.radius_mad,
    )
    filter_applied = int(keep.sum()) >= args.min_points
    if not filter_applied:
        keep = np.ones(points.shape[0], dtype=bool)
    filtered_points = points[keep]
    filtered_depth = depth_masked[keep]
    filtered_valid = valid.copy()
    if keep.shape[0] == ys.shape[0]:
        filtered_valid[ys[~keep], xs[~keep]] = False
    if filtered_points.size == 0:
        filtered_points = points
        filtered_depth = depth_masked
        filtered_valid = valid
        filter_applied = False

    points = filtered_points
    depth_masked = filtered_depth

    stats = {
        "image": str(image_path),
        "depth_image": str(depth_path),
        "mask": str(mask_path),
        "model": "real_depth",
        "depth_scale": float(depth_scale_value),
        "camera_fx": float(fx),
        "camera_fy": float(fy),
        "camera_cx": float(cx),
        "camera_cy": float(cy),
        "points_count_raw": points_count_raw,
        "points_count": int(points.shape[0]),
        "filter_applied": bool(filter_applied),
        "filter_border_margin": int(args.border_margin),
        "filter_depth_mad": float(args.depth_mad),
        "filter_radius_mad": float(args.radius_mad),
        "filter_min_points": int(args.min_points),
        "depth_mean": float(depth_masked.mean()),
        "depth_median": float(np.median(depth_masked)),
        "depth_min": float(depth_masked.min()),
        "depth_max": float(depth_masked.max()),
    }
    stats.update(compute_scale(points))

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    np.savez_compressed(
        output_npz,
        mask=mask.astype(np.uint8),
        valid_mask=filtered_valid.astype(np.uint8),
        points_masked=points.astype(np.float32),
        depth_masked=depth_masked.astype(np.float32),
        valid_full=valid_full.astype(np.uint8),
        points_full=points_full.astype(np.float32),
        depth_full=depth.astype(np.float32),
    )

    save_points_ply(points, output_ply)
    save_points_ply(points_full, output_full_ply)

    print(f"Saved real depth stats: {output_json}")
    print(f"Saved real depth npz: {output_npz}")
    print(f"Saved real depth ply: {output_ply}")
    print(f"Saved real depth full ply: {output_full_ply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
