#!/usr/bin/env python3
"""TEASER++ 3DMatch example runner for descriptor-based correspondences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sam3d_scale_utils import nearest_neighbors


def load_keypoints(path: Path) -> np.ndarray:
    idx = np.genfromtxt(path)
    if np.isscalar(idx):
        idx = np.array([idx])
    return idx.astype(int)


def mutual_nn(src_desc: np.ndarray, dst_desc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_to_dst, _ = nearest_neighbors(src_desc, dst_desc, chunk=256)
    dst_to_src, _ = nearest_neighbors(dst_desc, src_desc, chunk=256)
    src_idx = np.arange(src_desc.shape[0])
    keep = src_idx == dst_to_src[src_to_dst]
    return src_idx[keep], src_to_dst[keep]


def auto_noise_bound(points: np.ndarray) -> float:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    return max(1e-6, 0.01 * diag)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TEASER++ on 3DMatch sample data.")
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path(
            "/home/vision/Sim2Real_Data_Augmentation_for_VLA/sam3d_metric_scale/"
            "TEASER-plusplus/examples/example_data/3dmatch_sample"
        ),
    )
    parser.add_argument("--frag1", type=int, default=2)
    parser.add_argument("--frag2", type=int, default=36)
    parser.add_argument("--desc1", type=Path, default=None)
    parser.add_argument("--desc2", type=Path, default=None)
    parser.add_argument("--keypoints-dir", type=Path, default=None)
    parser.add_argument("--noise-bound", type=float, default=0.05)
    parser.add_argument("--gnc-factor", type=float, default=1.4)
    parser.add_argument("--cbar2", type=float, default=1.0)
    parser.add_argument("--rot-max-iters", type=int, default=100)
    parser.add_argument("--estimate-scaling", action="store_true")
    parser.add_argument("--show-viz", action="store_true")
    parser.add_argument("--max-spheres", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_root = args.scene_root
    frag1 = args.frag1
    frag2 = args.frag2

    desc1 = args.desc1 or scene_root / (
        f"cloud_bin_{frag1}.ply_0.150000_16_1.750000_3DSmoothNet.npz"
    )
    desc2 = args.desc2 or scene_root / (
        f"cloud_bin_{frag2}.ply_0.150000_16_1.750000_3DSmoothNet.npz"
    )
    keypoints_dir = args.keypoints_dir or scene_root / "01_Keypoints"
    key1 = keypoints_dir / f"cloud_bin_{frag1}Keypoints.txt"
    key2 = keypoints_dir / f"cloud_bin_{frag2}Keypoints.txt"
    ply1 = scene_root / f"cloud_bin_{frag1}.ply"
    ply2 = scene_root / f"cloud_bin_{frag2}.ply"

    if not (desc1.exists() and desc2.exists()):
        raise FileNotFoundError("Missing descriptor npz files.")
    if not (key1.exists() and key2.exists()):
        raise FileNotFoundError("Missing keypoint index files.")
    if not (ply1.exists() and ply2.exists()):
        raise FileNotFoundError("Missing ply files.")

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required to load PLY files for this script."
        ) from exc

    try:
        import teaserpp_python
    except ImportError as exc:
        raise RuntimeError(
            "teaserpp_python is required. Install with:\n"
            "  pip install teaserpp-python"
        ) from exc

    desc1_data = np.load(desc1)["data"].astype(np.float32)
    desc2_data = np.load(desc2)["data"].astype(np.float32)
    key_idx1 = load_keypoints(key1)
    key_idx2 = load_keypoints(key2)

    frag1_pc = o3d.io.read_point_cloud(str(ply1))
    frag2_pc = o3d.io.read_point_cloud(str(ply2))
    frag1_points = np.asarray(frag1_pc.points)
    frag2_points = np.asarray(frag2_pc.points)

    frag1_key = frag1_points[key_idx1]
    frag2_key = frag2_points[key_idx2]

    src_idx, dst_idx = mutual_nn(desc2_data, desc1_data)
    src_corr = frag2_key[src_idx]
    dst_corr = frag1_key[dst_idx]

    if src_corr.size == 0:
        print("No mutual correspondences found.")
        return 1

    noise_bound = args.noise_bound
    if noise_bound <= 0.0:
        noise_bound = auto_noise_bound(dst_corr)

    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.cbar2 = float(args.cbar2)
    params.noise_bound = float(noise_bound)
    params.estimate_scaling = bool(args.estimate_scaling)
    params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    params.rotation_gnc_factor = float(args.gnc_factor)
    params.rotation_max_iterations = int(args.rot_max_iters)
    params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(src_corr.T.astype(np.float64), dst_corr.T.astype(np.float64))
    solution = solver.getSolution()
    r = np.asarray(solution.rotation, dtype=np.float64)
    t = np.asarray(solution.translation, dtype=np.float64).reshape(3)
    scale = float(solution.scale) if args.estimate_scaling else 1.0

    print(f"correspondences: {src_corr.shape[0]}")
    print(f"scale: {scale:.6f}")
    print("rotation:\n", r)
    print("translation:", t)

    if args.show_viz and o3d is not None:
        frag1_pc.paint_uniform_color([1.0, 0.3, 0.05])
        frag2_pc.paint_uniform_color([0.0, 0.629, 0.9])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = scale * r
        transform[:3, 3] = t
        frag2_pc.transform(transform)

        geoms = [frag1_pc, frag2_pc]
        try:
            max_clique = solver.getTranslationInliersMap()
        except Exception:
            max_clique = None
        if max_clique is not None:
            idx = np.asarray(list(max_clique), dtype=int)
            idx = idx[(idx >= 0) & (idx < dst_corr.shape[0])]
            if idx.size:
                rng = np.random.default_rng(0)
                if idx.size > args.max_spheres:
                    idx = rng.choice(idx, size=args.max_spheres, replace=False)
                mins = dst_corr.min(axis=0)
                maxs = dst_corr.max(axis=0)
                diag = float(np.linalg.norm(maxs - mins))
                radius = max(1e-6, diag * 0.01)
                for point in dst_corr[idx]:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                    sphere.compute_vertex_normals()
                    sphere.paint_uniform_color([0.0, 0.9, 0.1])
                    sphere.translate(point)
                    geoms.append(sphere)

        o3d.visualization.draw_geometries(geoms, window_name="TEASER++ 3DMatch")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
