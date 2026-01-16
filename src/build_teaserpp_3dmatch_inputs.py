#!/usr/bin/env python3
"""Build 3DMatch-style inputs from custom point clouds for TEASER++ test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FPFH_AUTO_RATIO = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 3DMatch-style inputs.")
    parser.add_argument("--sam3d-ply", type=Path, required=True)
    parser.add_argument("--moge-ply", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--frag1", type=int, default=2)
    parser.add_argument("--frag2", type=int, default=36)
    parser.add_argument("--keypoints", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--desc-tag",
        type=str,
        default="0.150000_16_1.750000_3DSmoothNet",
    )
    parser.add_argument("--fpfh-voxel", type=float, default=0.0)
    parser.add_argument("--fpfh-normal-radius", type=float, default=0.0)
    parser.add_argument("--fpfh-feature-radius", type=float, default=0.0)
    return parser.parse_args()


def auto_fpfh_params(points: np.ndarray) -> tuple[float, float, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    voxel = max(1e-6, diag * FPFH_AUTO_RATIO)
    normal_radius = voxel * 2.0
    feature_radius = voxel * 5.0
    return voxel, normal_radius, feature_radius


def compute_fpfh(
    points: np.ndarray,
    voxel_size: float,
    normal_radius: float,
    feature_radius: float,
):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required to compute FPFH.\n"
            "Install with: conda install -c conda-forge open3d"
        ) from exc

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )
    return np.asarray(fpfh.data, dtype=np.float32).T


def write_scene(
    points: np.ndarray,
    out_root: Path,
    frag_id: int,
    desc_tag: str,
    keypoints: int,
    seed: int,
    voxel_size: float,
    normal_radius: float,
    feature_radius: float,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required to write PLY files.\n"
            "Install with: conda install -c conda-forge open3d"
        ) from exc

    out_root.mkdir(parents=True, exist_ok=True)
    key_dir = out_root / "01_Keypoints"
    key_dir.mkdir(parents=True, exist_ok=True)

    ply_path = out_root / f"cloud_bin_{frag_id}.ply"
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(ply_path), cloud, write_ascii=True)

    desc = compute_fpfh(points, voxel_size, normal_radius, feature_radius)
    if desc.shape[0] != points.shape[0]:
        raise RuntimeError("FPFH descriptor count mismatch.")

    total = points.shape[0]
    if keypoints <= 0 or keypoints >= total:
        idx = np.arange(total)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=keypoints, replace=False)

    key_path = key_dir / f"cloud_bin_{frag_id}Keypoints.txt"
    np.savetxt(key_path, idx.astype(int), fmt="%d")

    desc_path = out_root / f"cloud_bin_{frag_id}.ply_{desc_tag}.npz"
    np.savez(desc_path, data=desc[idx].astype(np.float32))


def main() -> int:
    args = parse_args()
    sam3d_ply = args.sam3d_ply
    moge_ply = args.moge_ply
    scene_root = args.scene_root

    if not sam3d_ply.exists():
        raise FileNotFoundError(f"Missing SAM3D ply: {sam3d_ply}")
    if not moge_ply.exists():
        raise FileNotFoundError(f"Missing MoGe ply: {moge_ply}")

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required to read PLY files.\n"
            "Install with: conda install -c conda-forge open3d"
        ) from exc

    sam_cloud = o3d.io.read_point_cloud(str(sam3d_ply))
    moge_cloud = o3d.io.read_point_cloud(str(moge_ply))
    sam_points = np.asarray(sam_cloud.points)
    moge_points = np.asarray(moge_cloud.points)

    if sam_points.size == 0 or moge_points.size == 0:
        raise RuntimeError("Empty point cloud.")

    voxel = args.fpfh_voxel
    normal_radius = args.fpfh_normal_radius
    feature_radius = args.fpfh_feature_radius
    if voxel <= 0 or normal_radius <= 0 or feature_radius <= 0:
        auto_voxel, auto_normal, auto_feature = auto_fpfh_params(
            np.vstack([sam_points, moge_points])
        )
        if voxel <= 0:
            voxel = auto_voxel
        if normal_radius <= 0:
            normal_radius = auto_normal
        if feature_radius <= 0:
            feature_radius = auto_feature

    write_scene(
        sam_points,
        scene_root,
        args.frag1,
        args.desc_tag,
        args.keypoints,
        args.seed,
        voxel,
        normal_radius,
        feature_radius,
    )
    write_scene(
        moge_points,
        scene_root,
        args.frag2,
        args.desc_tag,
        args.keypoints,
        args.seed + 1,
        voxel,
        normal_radius,
        feature_radius,
    )

    print(f"Scene root: {scene_root}")
    print(f"Descriptor tag: {args.desc_tag}")
    print(f"Keypoints: {args.keypoints}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
