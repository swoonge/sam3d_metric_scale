#!/usr/bin/env python3
"""Run BUFFER-X on a single pair of point clouds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from sam3d_scale_utils import visualize_alignment_open3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BUFFER-X pair inference.")
    parser.add_argument("--src-ply", type=Path, required=True)
    parser.add_argument("--tgt-ply", type=Path, required=True)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument(
        "--bufferx-root",
        type=Path,
        default=Path("/home/vision/Sim2Real_Data_Augmentation_for_VLA/sam3d_metric_scale/BUFFER-X"),
    )
    parser.add_argument("--dataset", type=str, default="3DMatch")
    parser.add_argument("--root-dir", type=Path, default=None)
    parser.add_argument("--downsample", type=float, default=0.0)
    parser.add_argument("--voxel-size-0", type=float, default=0.0)
    parser.add_argument("--max-num-points", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--src-scale", type=float, default=1.0)
    parser.add_argument("--show-viz", action="store_true")
    parser.add_argument("--viz-max-points", type=int, default=5000)
    parser.add_argument("--viz-max-pairs", type=int, default=200)
    return parser.parse_args()


def load_point_cloud(path: Path) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(cloud.points, dtype=np.float32)


def main() -> int:
    args = parse_args()
    if not args.src_ply.exists():
        raise FileNotFoundError(f"Missing src ply: {args.src_ply}")
    if not args.tgt_ply.exists():
        raise FileNotFoundError(f"Missing tgt ply: {args.tgt_ply}")

    bufferx_root = args.bufferx_root
    if not bufferx_root.exists():
        raise FileNotFoundError(f"Missing BUFFER-X root: {bufferx_root}")
    sys.path.insert(0, str(bufferx_root))

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required for BUFFER-X inference.\n"
            "Install with: conda install -c conda-forge open3d"
        ) from exc

    from config import make_cfg
    from utils.tools import sphericity_based_voxel_analysis
    from models.BUFFERX import BufferX

    if args.root_dir is None:
        args.root_dir = bufferx_root / "datasets"
    cfg = make_cfg(args.dataset, args.root_dir)
    cfg.stage = "test"

    if args.max_num_points > 0:
        cfg.data.max_numPts = int(args.max_num_points)

    src_cloud = o3d.io.read_point_cloud(str(args.src_ply))
    tgt_cloud = o3d.io.read_point_cloud(str(args.tgt_ply))

    if args.src_scale != 1.0:
        src_cloud.scale(float(args.src_scale), center=(0.0, 0.0, 0.0))

    if args.downsample > 0.0:
        cfg.data.downsample = float(args.downsample)
        sphericity, is_aligned_to_global_z = 0.0, False
    else:
        cfg.data.downsample, sphericity, is_aligned_to_global_z = sphericity_based_voxel_analysis(
            src_cloud, tgt_cloud
        )

    src_fds = src_cloud.voxel_down_sample(voxel_size=cfg.data.downsample)
    tgt_fds = tgt_cloud.voxel_down_sample(voxel_size=cfg.data.downsample)
    src_pts = np.asarray(src_fds.points, dtype=np.float32)
    tgt_pts = np.asarray(tgt_fds.points, dtype=np.float32)
    np.random.default_rng(args.seed).shuffle(src_pts)
    np.random.default_rng(args.seed + 1).shuffle(tgt_pts)

    ds_size = cfg.data.voxel_size_0
    if args.voxel_size_0 > 0.0:
        ds_size = float(args.voxel_size_0)
        cfg.data.voxel_size_0 = ds_size
        cfg.data.voxel_size_1 = ds_size

    src_sds = src_fds.voxel_down_sample(voxel_size=ds_size)
    tgt_sds = tgt_fds.voxel_down_sample(voxel_size=ds_size)
    src_kpt = np.asarray(src_sds.points, dtype=np.float32)
    tgt_kpt = np.asarray(tgt_sds.points, dtype=np.float32)
    np.random.default_rng(args.seed + 2).shuffle(src_kpt)
    np.random.default_rng(args.seed + 3).shuffle(tgt_kpt)

    if src_kpt.shape[0] > cfg.data.max_numPts:
        idx = np.random.default_rng(args.seed).choice(
            src_kpt.shape[0], cfg.data.max_numPts, replace=False
        )
        src_kpt = src_kpt[idx]
    if tgt_kpt.shape[0] > cfg.data.max_numPts:
        idx = np.random.default_rng(args.seed + 1).choice(
            tgt_kpt.shape[0], cfg.data.max_numPts, replace=False
        )
        tgt_kpt = tgt_kpt[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BufferX(cfg).to(device)
    experiment_id = args.experiment_id or cfg.test.experiment_id
    for stage in cfg.train.all_stage:
        ckpt = bufferx_root / "snapshot" / experiment_id / stage / "best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        state_dict = torch.load(ckpt, map_location=device)
        new_dict = {k: v for k, v in state_dict.items() if stage in k}
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

    model.eval()
    data_source = {
        "src_fds_pcd": torch.tensor(src_pts, dtype=torch.float32, device=device),
        "tgt_fds_pcd": torch.tensor(tgt_pts, dtype=torch.float32, device=device),
        "src_sds_pcd": torch.tensor(src_kpt, dtype=torch.float32, device=device),
        "tgt_sds_pcd": torch.tensor(tgt_kpt, dtype=torch.float32, device=device),
        "relt_pose": torch.eye(4, dtype=torch.float32, device=device),
        "src_id": str(args.src_ply),
        "tgt_id": str(args.tgt_ply),
        "voxel_sizes": torch.tensor([ds_size], dtype=torch.float32, device=device),
        "dataset_names": [cfg.data.dataset],
        "sphericity": torch.tensor([sphericity], dtype=torch.float32, device=device),
        "is_aligned_to_global_z": is_aligned_to_global_z,
    }

    with torch.no_grad():
        pose, _ = model(data_source)

    if pose is None:
        print("BUFFER-X returned None (check inputs).")
        return 1

    if isinstance(pose, torch.Tensor):
        pose = np.array(pose.detach().cpu().tolist(), dtype=np.float32)
    else:
        pose = np.asarray(pose, dtype=np.float32)
    print("BUFFER-X transform:")
    print(pose)

    if args.show_viz:
        visualize_alignment_open3d(
            src_pts,
            tgt_pts,
            scale=1.0,
            r=pose[:3, :3],
            t=pose[:3, 3],
            matched_dst=None,
            max_points=args.viz_max_points,
            max_spheres=args.viz_max_pairs,
            seed=args.seed,
            title="BUFFER-X alignment",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
