"""SAM3D camera/scale usage visualization.

Outputs multiple artifacts to inspect how SAM3D pose/scale values align with MoGe.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3D camera/scale viz outputs")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mask-stem", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--camera-focal", type=float, default=None)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip launching the interactive viewer.",
    )
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser window.",
    )
    parser.add_argument(
        "--analysis-seed",
        type=int,
        default=None,
        help="Seed for SAM3D analysis fallback (only used when pose/pointmap are missing).",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


def find_image_path(output_root: Path) -> Optional[Path]:
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        candidates = sorted(output_root.glob(ext))
        if candidates:
            return candidates[0]
    moge_dir = output_root / "moge_scale"
    for json_path in sorted(moge_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        image_path = payload.get("image")
        if image_path:
            candidate = Path(image_path)
            if candidate.exists():
                return candidate
    return None


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask * 255] * 3, axis=2).astype(np.uint8)


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    values = depth[valid]
    vmin, vmax = np.percentile(values, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (depth - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    if plt is not None:
        try:
            import matplotlib

            cmap = matplotlib.colormaps.get_cmap("magma")
        except Exception:
            import matplotlib.cm as cm

            cmap = cm.get_cmap("magma")
        rgba = cmap(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        rgb[~valid] = 0
        return rgb
    gray = (norm * 255).astype(np.uint8)
    gray[~valid] = 0
    return np.stack([gray] * 3, axis=2)


def find_moge_npz(moge_dir: Path, mask_stem: str) -> Optional[Path]:
    candidates = sorted(moge_dir.glob(f"*_{mask_stem}.npz"))
    if not candidates:
        candidates = sorted(moge_dir.glob(f"{mask_stem}.npz"))
    return candidates[0] if candidates else None


def load_moge_depth(npz_path: Path) -> Optional[np.ndarray]:
    if npz_path is None or not npz_path.exists():
        return None
    data = np.load(npz_path)
    valid_mask = data.get("valid_mask")
    depth_masked = data.get("depth_masked")
    if valid_mask is None or depth_masked is None:
        return None
    valid_mask = valid_mask.astype(bool)
    depth_masked = depth_masked.astype(np.float32)
    depth_full = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    if depth_masked.shape[0] != int(valid_mask.sum()):
        return None
    depth_full[valid_mask] = depth_masked
    return depth_full


def load_moge_points(npz_path: Path) -> Optional[np.ndarray]:
    if npz_path is None or not npz_path.exists():
        return None
    data = np.load(npz_path)
    points = data.get("points_masked")
    if points is not None:
        points = points.astype(np.float32)
        points = points[np.isfinite(points).all(axis=1)]
        return points
    valid_mask = data.get("valid_mask")
    depth_masked = data.get("depth_masked")
    if valid_mask is None or depth_masked is None:
        return None
    valid_mask = valid_mask.astype(bool)
    depth_masked = depth_masked.astype(np.float32)
    if depth_masked.shape[0] != int(valid_mask.sum()):
        return None
    ys, xs = np.where(valid_mask)
    height, width = valid_mask.shape
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    focal = float(max(height, width))
    z = depth_masked
    x = (xs - cx) * z / focal
    y = -(ys - cy) * z / focal
    points = np.stack([x, y, z], axis=1)
    points = points[np.isfinite(points).all(axis=1)]
    return points


def load_ply_points(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("plyfile is required to parse PLY.") from exc

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise RuntimeError("PLY file missing vertex data.")
    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()
    points = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).astype(np.float32)

    colors = None
    if all(name in names for name in ("red", "green", "blue")):
        colors = np.stack(
            (vertex["red"], vertex["green"], vertex["blue"]),
            axis=1,
        ).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
    elif all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        colors = np.stack(
            (vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]),
            axis=1,
        ).astype(np.float32)
        colors = np.clip(colors + 0.5, 0.0, 1.0)
    return points, colors


def write_ply_points(path: Path, points: np.ndarray, colors: Optional[np.ndarray]) -> None:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise RuntimeError("plyfile is required to write PLY.") from exc

    points = points.astype(np.float32)
    if colors is None:
        verts = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        verts["x"] = points[:, 0]
        verts["y"] = points[:, 1]
        verts["z"] = points[:, 2]
    else:
        colors = np.clip(colors, 0.0, 1.0)
        colors_u8 = (colors * 255).astype(np.uint8)
        verts = np.empty(
            points.shape[0],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        verts["x"] = points[:, 0]
        verts["y"] = points[:, 1]
        verts["z"] = points[:, 2]
        verts["red"] = colors_u8[:, 0]
        verts["green"] = colors_u8[:, 1]
        verts["blue"] = colors_u8[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))


def render_pointcloud_png(
    path: Path,
    points: np.ndarray,
    colors: Optional[np.ndarray],
    max_points: int,
    title: str,
) -> None:
    if plt is None:
        return
    if points is None or points.size == 0:
        return
    points = points.astype(np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        return
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2])
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_pointcloud_figure(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    max_points: int,
    title: str,
) -> tuple[object, Optional[str]]:
    if points is None or points.size == 0:
        return None, "empty"
    points = points.astype(np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        return None, "invalid"
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    try:
        import plotly.graph_objects as go

        marker = dict(size=2, opacity=0.8)
        if colors is not None:
            marker["color"] = colors
        else:
            marker["color"] = points[:, 2]
            marker["colorscale"] = "Viridis"
            marker["showscale"] = True
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=marker,
                )
            ]
        )
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig, None
    except Exception as exc:
        return None, f"{title}: {exc}"


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    # quat: (4,) in (w, x, y, z)
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def get_camera_to_pytorch3d_rotation() -> Optional[np.ndarray]:
    try:
        import torch
        from pytorch3d.renderer import look_at_view_transform

        rot, _ = look_at_view_transform(
            eye=np.array([[0, 0, -1]]),
            at=np.array([[0, 0, 0]]),
            up=np.array([[0, -1, 0]]),
            device="cpu",
        )
        return rot[0].cpu().numpy().astype(np.float32)
    except Exception:
        return None


def apply_rotation(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    if points is None or points.size == 0:
        return points
    return (rotation @ points.T).T


def compute_ssi_scale_shift(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points is None or points.size == 0:
        return np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    z_vals = points[:, 2]
    shift_z = np.nanmedian(z_vals)
    shift = np.array([0.0, 0.0, shift_z], dtype=np.float32)
    shifted = points - shift.reshape(1, 3)
    scale = np.nanmean(np.abs(shifted))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    scale_vec = np.array([scale, scale, scale], dtype=np.float32)
    return scale_vec, shift


def compute_object_scale_shift(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points is None or points.size == 0:
        return np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    shift = np.nanmedian(points, axis=0).astype(np.float32)
    centered = points - shift.reshape(1, 3)
    max_dims = np.nanmax(np.abs(centered), axis=1)
    scale = np.nanmedian(max_dims)
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    scale_vec = np.array([scale, scale, scale], dtype=np.float32)
    return scale_vec, shift


def align_points_ssi(
    points: np.ndarray,
    source_scale: np.ndarray,
    source_shift: np.ndarray,
    target_scale: np.ndarray,
    target_shift: np.ndarray,
) -> np.ndarray:
    if points is None or points.size == 0:
        return points
    normalized = (points - source_shift.reshape(1, 3)) / source_scale.reshape(1, 3)
    return normalized * target_scale.reshape(1, 3) + target_shift.reshape(1, 3)


def parse_pose_txt(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    numbers = [
        float(x)
        for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
    ]
    if len(numbers) < 10:
        raise ValueError("Pose file missing values.")
    quat = np.array(numbers[0:4], dtype=np.float32)
    trans = np.array(numbers[4:7], dtype=np.float32)
    scale = np.array(numbers[7:10], dtype=np.float32)
    return quat, trans, scale


def apply_transform(
    points: np.ndarray,
    scale: np.ndarray,
    quat: np.ndarray,
    trans: np.ndarray,
    *,
    inverse: bool = False,
    trans_scale: float = 1.0,
) -> np.ndarray:
    trans = trans * float(trans_scale)
    R = quaternion_to_matrix(quat)
    if inverse:
        inv_scale = 1.0 / np.maximum(scale, 1e-8)
        centered = points - trans.reshape(1, 3)
        rotated = (R.T @ centered.T).T
        return rotated * inv_scale.reshape(1, 3)
    scaled = points * scale.reshape(1, 3)
    rotated = (R @ scaled.T).T
    return rotated + trans.reshape(1, 3)


def apply_transform_matrix(
    points: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    trans: np.ndarray,
) -> np.ndarray:
    if points is None or points.size == 0:
        return points
    scaled = points * scale.reshape(1, 3)
    rotated = (rotation @ scaled.T).T
    return rotated + trans.reshape(1, 3)


def apply_transform_halo(
    points: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    trans: np.ndarray,
    rot_mesh: np.ndarray,
    rot_pm: np.ndarray,
) -> np.ndarray:
    if points is None or points.size == 0:
        return points
    linear = rotation @ np.diag(scale.astype(np.float32))
    halo_linear = rot_mesh @ linear @ rot_pm
    halo_trans = rot_mesh @ trans.reshape(3)
    return (halo_linear @ points.T).T + halo_trans.reshape(1, 3)


def normalize_points(points: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    if points is None or points.size == 0:
        return points
    return (points - shift.reshape(1, 3)) / scale.reshape(1, 3)


def ensure_analysis_outputs(
    output_root: Path,
    image_path: Path,
    mask_path: Path,
    analysis_dir: Path,
    seed: Optional[int],
) -> bool:
    pose_txt = analysis_dir / f"{mask_path.stem}_pose.txt"
    pointmap_ply = analysis_dir / f"{mask_path.stem}_pointmap_cloud.ply"
    if pose_txt.exists() and pointmap_ply.exists():
        return True
    repo_root = Path(__file__).resolve().parents[1]
    analyze_script = repo_root / "src" / "sam3d_inference_analyze.py"
    if not analyze_script.exists():
        print(f"Missing analysis script: {analyze_script}")
        return False
    cmd = [
        sys.executable,
        str(analyze_script),
        "--output-root",
        str(output_root),
        "--image",
        str(image_path),
        "--mask",
        str(mask_path),
        "--analysis-dir",
        str(analysis_dir),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    print("Generating missing SAM3D analysis outputs...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("Failed to generate analysis outputs.")
        return False
    return pose_txt.exists()


def reorder_quat(quat: np.ndarray, order: str) -> np.ndarray:
    if order == "xyzw":
        return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)
    return quat


def flip_axis(points: np.ndarray, axis: str) -> np.ndarray:
    out = points.copy()
    if axis == "y":
        out[:, 1] *= -1.0
    elif axis == "z":
        out[:, 2] *= -1.0
    return out


def swap_axes(points: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    out = points.copy()
    a, b = axes
    out[:, [a, b]] = out[:, [b, a]]
    return out


def project_depth(points: np.ndarray, height: int, width: int, focal: Optional[float]) -> np.ndarray:
    if focal is None:
        focal = float(max(height, width))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    u = np.round((x * focal / z) + cx).astype(np.int32)
    v = np.round((-y * focal / z) + cy).astype(np.int32)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v, u), z)
    depth[~np.isfinite(depth)] = np.nan
    depth[depth == np.inf] = np.nan
    return depth


def main() -> int:
    args = parse_args()
    output_root = resolve_path(args.output_root, Path.cwd())
    if not output_root.exists():
        print(f"Missing output root: {output_root}")
        return 1

    image_path = find_image_path(output_root)
    if image_path is None:
        print("Missing image in output root. Provide image inside output root.")
        return 1

    mask_dir = output_root / "sam2_masks"
    masks = sorted(mask_dir.glob("*.png"))
    if not masks:
        print(f"No masks found under: {mask_dir}")
        return 1

    mask_path = None
    if args.mask_stem:
        candidate = mask_dir / f"{args.mask_stem}.png"
        if candidate.exists():
            mask_path = candidate
    if mask_path is None:
        mask_path = masks[0]

    mask_stem = mask_path.stem
    moge_npz = find_moge_npz(output_root / "moge_scale", mask_stem)
    sam3d_ply = output_root / "sam3d" / f"{mask_stem}.ply"
    scale_txt = output_root / "sam3d_scale" / f"{mask_stem}_scale.txt"
    analysis_dir = output_root / "sam3d_analysis"
    pose_txt = analysis_dir / f"{mask_stem}_pose.txt"
    pointmap_ply = analysis_dir / f"{mask_stem}_pointmap_cloud.ply"

    if not sam3d_ply.exists():
        print(f"Missing SAM3D PLY: {sam3d_ply}")
        return 1
    if not pose_txt.exists() or not pointmap_ply.exists():
        ok = ensure_analysis_outputs(
            output_root=output_root,
            image_path=image_path,
            mask_path=mask_path,
            analysis_dir=analysis_dir,
            seed=args.analysis_seed,
        )
        if not ok or not pose_txt.exists():
            print(f"Missing pose file: {pose_txt}")
            return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = output_root / "sam3d_camera_scale_viz" / mask_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    mask = load_mask(mask_path)
    Image.fromarray(image).save(output_dir / "01_image.png")
    Image.fromarray(mask_to_rgb(mask)).save(output_dir / "02_mask.png")

    moge_depth = load_moge_depth(moge_npz) if moge_npz else None
    if moge_depth is not None:
        Image.fromarray(depth_to_rgb(moge_depth)).save(output_dir / "03_moge_depth_full.png")

    moge_points = load_moge_points(moge_npz)
    moge_colors = None
    if moge_points is not None and moge_points.size > 0:
        moge_colors = np.tile(
            np.array([[0.2, 0.6, 1.0]], dtype=np.float32),
            (moge_points.shape[0], 1),
        )
        write_ply_points(output_dir / "04_moge_points.ply", moge_points, moge_colors)
        render_pointcloud_png(
            output_dir / "04_moge_points.png",
            moge_points,
            moge_colors,
            args.max_points,
            "MoGe point cloud",
        )

    points, colors = load_ply_points(sam3d_ply)
    pointmap_points = None
    pointmap_colors = None
    if pointmap_ply.exists():
        try:
            pointmap_points, pointmap_colors = load_ply_points(pointmap_ply)
        except Exception:
            pointmap_points = None
            pointmap_colors = None

    quat, trans, pose_scale = parse_pose_txt(pose_txt)
    quat_wxyz = reorder_quat(quat, "wxyz")
    pose_rot = quaternion_to_matrix(quat_wxyz)
    sam_pose_points = apply_transform_matrix(points, pose_scale, pose_rot, trans)
    sam_pose_colors = colors
    write_ply_points(output_dir / "05_sam3d_pose.ply", sam_pose_points, sam_pose_colors)
    render_pointcloud_png(
        output_dir / "05_sam3d_pose.png",
        sam_pose_points,
        sam_pose_colors,
        args.max_points,
        "SAM3D pose (object -> camera)",
    )

    if pointmap_points is not None and pointmap_points.size > 0:
        pointmap_colors = np.tile(
            np.array([[0.2, 0.9, 0.5]], dtype=np.float32),
            (pointmap_points.shape[0], 1),
        )
        write_ply_points(
            output_dir / "05b_sam3d_pointmap.ply", pointmap_points, pointmap_colors
        )
        render_pointcloud_png(
            output_dir / "05b_sam3d_pointmap.png",
            pointmap_points,
            pointmap_colors,
            args.max_points,
            "SAM3D pointmap (camera)",
        )

    # Fixed axis-rotation hypotheses (mesh local -> pose local)
    rot_fix_a = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot_fix_b = rot_fix_a.T
    rot_fix_c = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rot_x90 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot_xm90 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot_y90 = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot_ym90 = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot_z90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rot_zm90 = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rot_fix_b_y90 = rot_fix_b @ rot_y90
    sam_pose_fix_b_y90 = apply_transform_matrix(
        points, pose_scale, pose_rot @ rot_fix_b_y90, trans
    )
    sam_pose_fix_b_y90_pre = apply_transform_matrix(
        points, pose_scale, rot_fix_b_y90 @ pose_rot, trans
    )
    sam_pose_fix_b_y90_post_t = apply_transform_matrix(
        points, pose_scale, pose_rot.T @ rot_fix_b_y90, trans
    )
    sam_pose_fix_b_y90_pre_t = apply_transform_matrix(
        points, pose_scale, rot_fix_b_y90 @ pose_rot.T, trans
    )
    sam_pose_halo = apply_transform_halo(
        points, pose_scale, pose_rot, trans, rot_fix_a, rot_fix_c
    )
    sam_pose_halo_prepm = apply_transform_matrix(
        apply_rotation(points, rot_fix_c),
        pose_scale,
        rot_fix_a @ pose_rot,
        rot_fix_a @ trans,
    )
    sam_pose_halo_inv = apply_transform_halo(
        points, pose_scale, pose_rot, trans, rot_fix_a.T, rot_fix_c.T
    )
    sam_pose_fix_b_y90_swap_yz = swap_axes(sam_pose_fix_b_y90, (1, 2))
    sam_pose_fix_b_y90_swap_yz_flip_y = flip_axis(sam_pose_fix_b_y90_swap_yz, "y")
    sam_pose_fix_b_y90_swap_yz_flip_z = flip_axis(sam_pose_fix_b_y90_swap_yz, "z")
    sam_pose_fix_b_y90_flip_y = flip_axis(sam_pose_fix_b_y90, "y")
    sam_pose_fix_b_y90_flip_z = flip_axis(sam_pose_fix_b_y90, "z")

    def combine_clouds(
        base_points: Optional[np.ndarray],
        base_colors: Optional[np.ndarray],
        overlay_points: Optional[np.ndarray],
        overlay_color: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if overlay_points is None or overlay_points.size == 0:
            if base_points is None or base_points.size == 0:
                return np.empty((0, 3), dtype=np.float32), None
            return base_points, base_colors
        if base_points is None or base_points.size == 0:
            overlay_colors = np.tile(overlay_color, (overlay_points.shape[0], 1))
            return overlay_points, overlay_colors
        overlay_colors = np.tile(overlay_color, (overlay_points.shape[0], 1))
        combined_pts = np.vstack([base_points, overlay_points])
        combined_cols = (
            np.vstack([base_colors, overlay_colors])
            if base_colors is not None
            else None
        )
        return combined_pts, combined_cols

    sam_overlay_color = np.array([1.0, 0.3, 0.2], dtype=np.float32)

    baseline_base_points = pointmap_points
    baseline_base_colors = (
        pointmap_colors
        if pointmap_colors is not None and pointmap_points is not None and pointmap_points.size > 0
        else moge_colors
    )
    baseline_pts, baseline_cols = combine_clouds(
        baseline_base_points, baseline_base_colors, sam_pose_points, sam_overlay_color
    )
    write_ply_points(
        output_dir / "06_sam3d_pose_plus_pointmap.ply", baseline_pts, baseline_cols
    )
    render_pointcloud_png(
        output_dir / "06_sam3d_pose_plus_pointmap.png",
        baseline_pts,
        baseline_cols,
        args.max_points,
        "06 SAM3D pose + pointmap",
    )

    variants = []
    variants.append(
        (
            "12",
            "pointmap_plus_pose_fix_b_y90",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_fix_b_y90,
                sam_overlay_color,
            ),
        )
    )
    variants.append(
        (
            "13",
            "pointmap_plus_pose_fix_b_y90_pre",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_fix_b_y90_pre,
                sam_overlay_color,
            ),
        )
    )
    variants.append(
        (
            "14",
            "pointmap_plus_pose_fix_b_y90_post_t",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_fix_b_y90_post_t,
                sam_overlay_color,
            ),
        )
    )
    variants.append(
        (
            "15",
            "pointmap_plus_pose_fix_b_y90_pre_t",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_fix_b_y90_pre_t,
                sam_overlay_color,
            ),
        )
    )
    variants.append(
        (
            "16",
            "pointmap_plus_pose_halo_apply",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_halo,
                sam_overlay_color,
            ),
        )
    )
    variants.append(
        (
            "17",
            "pointmap_plus_pose_halo_prepm",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_halo_prepm,
                sam_overlay_color,
            ),
        )
    )
    variants.append(
        (
            "18",
            "pointmap_plus_pose_halo_inverse",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_halo_inv,
                sam_overlay_color,
            ),
        )
    )
    if pointmap_points is not None and pointmap_points.size > 0:
        ssi_scale, ssi_shift = compute_ssi_scale_shift(pointmap_points)
        pointmap_norm = normalize_points(pointmap_points, ssi_scale, ssi_shift)
        sam_pose_norm = normalize_points(sam_pose_points, ssi_scale, ssi_shift)
        sam_pose_fix_b_y90_norm = normalize_points(
            sam_pose_fix_b_y90, ssi_scale, ssi_shift
        )
        variants.append(
            (
                "19",
                "pointmap_norm_plus_pose",
                combine_clouds(
                    pointmap_norm,
                    pointmap_colors,
                    sam_pose_norm,
                    sam_overlay_color,
                ),
            )
        )
        variants.append(
            (
                "20",
                "pointmap_norm_plus_pose_fix_b_y90",
                combine_clouds(
                    pointmap_norm,
                    pointmap_colors,
                    sam_pose_fix_b_y90_norm,
                    sam_overlay_color,
                ),
            )
        )
    variants.append(
        (
            "21",
            "pointmap_plus_pose_fix_b_y90_swap_yz",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_fix_b_y90_swap_yz,
                sam_overlay_color,
            ),
        )
    )
    variants.append(
        (
            "22",
            "pointmap_plus_pose_fix_b_y90_swap_yz_flip_y",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_fix_b_y90_swap_yz_flip_y,
                sam_overlay_color,
            ),
        )
    )
    variants.append(
        (
            "23",
            "pointmap_plus_pose_fix_b_y90_swap_yz_flip_z",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_fix_b_y90_swap_yz_flip_z,
                sam_overlay_color,
            ),
        )
    )
    variants.append(
        (
            "24",
            "pointmap_plus_pose_fix_b_y90_flip_y",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_fix_b_y90_flip_y,
                sam_overlay_color,
            ),
        )
    )
    variants.append(
        (
            "25",
            "pointmap_plus_pose_fix_b_y90_flip_z",
            combine_clouds(
                baseline_base_points,
                baseline_base_colors,
                sam_pose_fix_b_y90_flip_z,
                sam_overlay_color,
            ),
        )
    )

    variant_outputs = []
    for tag, name, (combined_pts, combined_cols) in variants:
        out_ply = output_dir / f"{tag}_{name}.ply"
        out_png = output_dir / f"{tag}_{name}.png"
        write_ply_points(out_ply, combined_pts, combined_cols)
        render_pointcloud_png(
            out_png,
            combined_pts,
            combined_cols,
            args.max_points,
            f"{tag} MoGe + SAM3D ({name})",
        )
        variant_outputs.append((tag, name, out_ply, out_png))

    summary = [
        f"mask_stem: {mask_stem}",
        f"pose_scale (sam3d): {pose_scale.tolist()}",
        f"pose_translation: {trans.tolist()}",
        f"pose_quaternion (wxyz): {quat.tolist()}",
        "rotation_fixes:",
        f"  fix_a: {rot_fix_a.tolist()}",
        f"  fix_b: {rot_fix_b.tolist()}",
        f"  fix_c: {rot_fix_c.tolist()}",
        f"  rot_x90: {rot_x90.tolist()}",
        f"  rot_xm90: {rot_xm90.tolist()}",
        f"  rot_y90: {rot_y90.tolist()}",
        f"  rot_ym90: {rot_ym90.tolist()}",
        f"  rot_z90: {rot_z90.tolist()}",
        f"  rot_zm90: {rot_zm90.tolist()}",
        f"  fix_b_y90: {rot_fix_b_y90.tolist()}",
        f"outputs: {output_dir}",
        "variants:",
        *[f"  {tag}: {name}" for tag, name, _, _ in variant_outputs],
    ]
    if pointmap_points is not None and pointmap_points.size > 0:
        summary.extend(
            [
                f"ssi_scale (pointmap): {ssi_scale.tolist()}",
                f"ssi_shift (pointmap): {ssi_shift.tolist()}",
            ]
        )
    (output_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved visualization outputs to: {output_dir}")

    if not args.no_viewer:
        try:
            import gradio as gr
        except ImportError:
            print("gradio is required for the interactive viewer.")
            return 0

        image_rgb = Image.open(output_dir / "01_image.png")
        mask_rgb = Image.open(output_dir / "02_mask.png")
        moge_depth_img = (
            Image.open(output_dir / "03_moge_depth_full.png")
            if (output_dir / "03_moge_depth_full.png").exists()
            else None
        )
        moge_points = np.empty((0, 3), dtype=np.float32)
        moge_colors = None
        moge_path = output_dir / "04_moge_points.ply"
        if moge_path.exists():
            moge_points, moge_colors = load_ply_points(moge_path)
        sam_pose_points, sam_pose_colors = load_ply_points(output_dir / "05_sam3d_pose.ply")
        combined_points, combined_colors = load_ply_points(
            output_dir / "06_sam3d_pose_plus_pointmap.ply"
        )

        moge_fig, moge_note = build_pointcloud_figure(
            moge_points, moge_colors, args.max_points, "MoGe point cloud"
        )
        sam_pose_fig, sam_pose_note = build_pointcloud_figure(
            sam_pose_points,
            sam_pose_colors,
            args.max_points,
            "SAM3D pose (object -> camera)",
        )
        combined_fig, combined_note = build_pointcloud_figure(
            combined_points,
            combined_colors,
            args.max_points,
            "06 SAM3D pose + pointmap",
        )

        variant_figs = []
        variant_notes = []
        for tag, name, out_ply, _ in variant_outputs:
            if not out_ply.exists():
                variant_figs.append(None)
                variant_notes.append(f"{tag}: missing")
                continue
            pts, cols = load_ply_points(out_ply)
            fig, note = build_pointcloud_figure(
                pts,
                cols,
                args.max_points,
                f"{tag} {name}",
            )
            variant_figs.append(fig)
            if note:
                variant_notes.append(f"{tag}: {note}")

        notes = []
        for note in (moge_note, sam_pose_note, combined_note):
            if note:
                notes.append(note)
        notes.extend(variant_notes)
        status_text = " / ".join(notes) if notes else "OK"
        summary_text = (output_dir / "summary.txt").read_text(encoding="utf-8")

        with gr.Blocks() as demo:
            gr.Markdown("# SAM3D Camera/Scale Viewer")
            status_view = gr.Markdown(status_text)
            summary_view = gr.Markdown(f"```\n{summary_text}\n```")
            with gr.Row():
                img_view = gr.Image(label="01 Image", value=image_rgb)
                mask_view = gr.Image(label="02 Mask", value=mask_rgb)
                moge_view = gr.Image(label="03 MoGe depth (full)", value=moge_depth_img)
            with gr.Row():
                moge_pc_view = gr.Plot(label="04 MoGe point cloud", value=moge_fig)
                sam_pose_view = gr.Plot(label="05 SAM3D pose", value=sam_pose_fig)
                combined_view = gr.Plot(label="06 Pose + pointmap", value=combined_fig)
            gr.Markdown("## Variants (12+)")
            for idx in range(0, len(variant_figs), 2):
                with gr.Row():
                    for offset in range(2):
                        vidx = idx + offset
                        if vidx >= len(variant_figs):
                            continue
                        label = f"{variant_outputs[vidx][0]} {variant_outputs[vidx][1]}"
                        gr.Plot(label=label, value=variant_figs[vidx])

        demo.launch(
            share=args.share,
            server_port=args.server_port,
            inbrowser=not args.no_browser,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
