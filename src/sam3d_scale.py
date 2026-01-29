"""SAM3D 스케일 추정 알고리즘 테스트 러너.

알고리즘 구현은 별도 파일로 분리되어 있으며, 여기서는 공통 입출력/시각화를 담당한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sam3d_scale_icp import estimate_scale as estimate_scale_icp
from sam3d_scale_teaserpp import estimate_scale as estimate_scale_teaserpp
from sam3d_scale_utils import (
    load_moge_points,
    load_ply_points,
    resolve_path,
    sample_points,
    visualize_debug_views,
    visualize_alignment,
    visualize_alignment_open3d,
    write_scaled_ply,
)


def strip_pose_suffix(stem: str) -> str:
    if stem.endswith("_pose"):
        return stem[: -len("_pose")]
    return stem


def load_pose_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_pose_payload(payload: dict | None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not payload:
        return None, None, None
    rot = payload.get("rotation_matrix")
    if rot is None:
        rot = payload.get("rotation")
        if isinstance(rot, dict):
            rot = rot.get("value")
    trans = payload.get("translation")
    scale = payload.get("scale")
    rot_mat = np.asarray(rot, dtype=np.float32).reshape(3, 3) if rot is not None else None
    trans_vec = np.asarray(trans, dtype=np.float32).reshape(3) if trans is not None else None
    if scale is None:
        scale_vec = None
    else:
        scale_arr = np.asarray(scale, dtype=np.float32).reshape(-1)
        if scale_arr.size == 1:
            scale_vec = np.repeat(scale_arr[0], 3)
        else:
            scale_vec = scale_arr[:3]
    return rot_mat, trans_vec, scale_vec


def apply_pose(points: np.ndarray, scale: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    scaled = points * scale.reshape(1, 3)
    rotated = (rot @ scaled.T).T
    return rotated + trans.reshape(1, 3)


def write_pose_ply(ply, points: np.ndarray, out_path: Path, scale_vec: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    vertex = ply["vertex"].data.copy()
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    names = vertex.dtype.names or ()
    if all(name in names for name in ("scale_0", "scale_1", "scale_2")):
        safe_scale = np.maximum(np.abs(scale_vec), 1e-12).astype(np.float32)
        scale_vals = vertex["scale_0"].astype(np.float32)
        scale_median = float(np.median(scale_vals))
        if scale_median < 0:
            log_scale = np.log(safe_scale)
            vertex["scale_0"] += log_scale[0]
            vertex["scale_1"] += log_scale[1]
            vertex["scale_2"] += log_scale[2]
        else:
            vertex["scale_0"] *= safe_scale[0]
            vertex["scale_1"] *= safe_scale[1]
            vertex["scale_2"] *= safe_scale[2]

    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(vertex, "vertex"))
        else:
            elements.append(element)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        elements,
        text=ply.text,
        comments=ply.comments,
        obj_info=ply.obj_info,
    ).write(str(out_path))


def load_mesh(path: Path):
    try:
        import trimesh
    except Exception:
        return None
    if not path.exists():
        return None
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            return None
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    return mesh


def save_transformed_mesh(src_path: Path, out_path: Path, scale_vec: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> None:
    mesh = load_mesh(src_path)
    if mesh is None:
        return
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rot @ np.diag(scale_vec.astype(np.float32))
    transform[:3, 3] = trans.astype(np.float32)
    mesh_t = mesh.copy()
    mesh_t.apply_transform(transform)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_t.export(out_path)


def save_scaled_mesh(src_path: Path, out_path: Path, scale: float) -> None:
    mesh = load_mesh(src_path)
    if mesh is None:
        return
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.eye(3, dtype=np.float32) * float(scale)
    mesh_t = mesh.copy()
    mesh_t.apply_transform(transform)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_t.export(out_path)


def parse_args() -> argparse.Namespace:
    """CLI 인자를 정의한다."""
    parser = argparse.ArgumentParser(
        description="Test multiple SAM3D scale estimation algorithms."
    )
    parser.add_argument("--sam3d-ply", type=Path, required=True)
    parser.add_argument("--moge-npz", type=Path, required=True)
    parser.add_argument(
        "--algo",
        choices=[
            "icp",
            "teaserpp",
        ],
        default="teaserpp",
        help="Scale estimation algorithm.",
    )
    parser.add_argument("--output-scale", type=Path, default=None)
    parser.add_argument("--output-scaled-ply", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["default", "scale_only"],
        default="scale_only",
        help=(
            "default: use sam3d-ply + moge-npz; "
            "scale_only: assume pose is aligned and estimate scale only."
        ),
    )
    parser.add_argument(
        "--fine-registration",
        action="store_true",
        help="After scale-only, run TEASER++ once (without scale) for R/t refinement.",
    )
    parser.add_argument(
        "--refine-registration",
        action="store_true",
        help="Alias for --fine-registration.",
    )
    parser.add_argument(
        "--save-posed",
        action="store_true",
        help="Also save pose-only outputs (_posed.*) without global scale.",
    )

    # ICP 옵션
    parser.add_argument("--icp-max-iters", type=int, default=30)
    parser.add_argument("--icp-tolerance", type=float, default=1e-5)
    parser.add_argument("--icp-nn-max-points", type=int, default=8000)
    parser.add_argument(
        "--icp-trim-ratio",
        type=float,
        default=0.8,
        help="Use closest ratio of matches for ICP update (0<r<=1).",
    )

    # TEASER++ 옵션
    parser.add_argument(
        "--teaser-noise-bound",
        type=float,
        default=0.0,
        help="Noise bound (<=0 uses auto based on bbox diag).",
    )
    parser.add_argument("--teaser-nn-max-points", type=int, default=4000)
    parser.add_argument("--teaser-max-correspondences", type=int, default=3000)
    parser.add_argument("--teaser-gnc-factor", type=float, default=1.4)
    parser.add_argument("--teaser-rot-max-iters", type=int, default=100)
    parser.add_argument("--teaser-cbar2", type=float, default=1.0)
    parser.add_argument("--teaser-iterations", type=int, default=1)
    parser.add_argument(
        "--teaser-correspondence",
        choices=["nn", "fpfh"],
        default="nn",
        help="Correspondence mode for TEASER++ (nn: raw XYZ, fpfh: Open3D FPFH).",
    )
    parser.add_argument(
        "--teaser-fpfh-voxel",
        type=float,
        default=0.0,
        help="FPFH voxel size (<=0 uses auto).",
    )
    parser.add_argument(
        "--teaser-fpfh-normal-radius",
        type=float,
        default=0.0,
        help="FPFH normal radius (<=0 uses auto).",
    )
    parser.add_argument(
        "--teaser-fpfh-feature-radius",
        type=float,
        default=0.0,
        help="FPFH feature radius (<=0 uses auto).",
    )
    parser.add_argument(
        "--teaser-estimate-scaling",
        action="store_true",
        help="Enable scale estimation in TEASER++ (default: off).",
    )
    parser.add_argument(
        "--teaser-icp-refine",
        action="store_true",
        help="Enable ICP refinement after TEASER++ (default: off).",
    )
    parser.add_argument("--teaser-icp-max-iters", type=int, default=100)
    parser.add_argument(
        "--teaser-icp-distance",
        type=float,
        default=0.0,
        help="ICP max correspondence distance (<=0 uses noise bound).",
    )

    # 시각화 옵션
    parser.add_argument(
        "--show-viz",
        action="store_true",
        help="Show alignment/matching visualization.",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument(
        "--debug-viz",
        action="store_true",
        help="Show debug 5-panel view for transform direction check.",
    )
    parser.add_argument(
        "--viz-method",
        choices=["matplotlib", "open3d"],
        default="matplotlib",
        help="Visualization backend (default: matplotlib).",
    )
    parser.add_argument("--viz-path", type=Path, default=None)
    parser.add_argument("--viz-dpi", type=int, default=150)
    parser.add_argument("--viz-max-points", type=int, default=5000)
    parser.add_argument("--viz-max-pairs", type=int, default=200)
    return parser.parse_args()


def pick_algorithm(args: argparse.Namespace, src: np.ndarray, dst: np.ndarray) -> dict:
    """알고리즘 선택 및 실행."""
    if args.algo == "icp":
        return estimate_scale_icp(
            src,
            dst,
            max_iters=args.icp_max_iters,
            tolerance=args.icp_tolerance,
            nn_max_points=args.icp_nn_max_points,
            trim_ratio=args.icp_trim_ratio,
            seed=args.seed,
        )
    if args.algo == "teaserpp":
        return estimate_scale_teaserpp(
            src,
            dst,
            nn_max_points=args.teaser_nn_max_points,
            max_correspondences=args.teaser_max_correspondences,
            noise_bound=args.teaser_noise_bound,
            cbar2=args.teaser_cbar2,
            gnc_factor=args.teaser_gnc_factor,
            rot_max_iters=args.teaser_rot_max_iters,
            estimate_scaling=args.teaser_estimate_scaling,
            iterations=args.teaser_iterations,
            correspondence=args.teaser_correspondence,
            fpfh_voxel=args.teaser_fpfh_voxel,
            fpfh_normal_radius=args.teaser_fpfh_normal_radius,
            fpfh_feature_radius=args.teaser_fpfh_feature_radius,
            icp_refine=args.teaser_icp_refine,
            icp_max_iters=args.teaser_icp_max_iters,
            icp_distance=args.teaser_icp_distance,
            seed=args.seed,
        )
    return estimate_scale_icp(
        src,
        dst,
        max_iters=args.icp_max_iters,
        tolerance=args.icp_tolerance,
        nn_max_points=args.icp_nn_max_points,
        trim_ratio=args.icp_trim_ratio,
        seed=args.seed,
    )


def main() -> int:
    """스케일 추정 실행."""
    args = parse_args()
    sam3d_ply = resolve_path(args.sam3d_ply, Path.cwd())
    moge_npz = resolve_path(args.moge_npz, Path.cwd())

    if not sam3d_ply.exists():
        print(f"Missing SAM3D ply: {sam3d_ply}")
        return 1
    if not moge_npz.exists():
        print(f"Missing MoGe npz: {moge_npz}")
        return 1

    try:
        pose_ply, sam_points = load_ply_points(sam3d_ply)
    except RuntimeError as exc:
        print(exc)
        print("Install with: conda run -n sam3d-objects python -m pip install plyfile")
        return 1

    sam3d_dir = sam3d_ply.parent
    base_stem = strip_pose_suffix(sam3d_ply.stem)
    raw_ply_path = sam3d_dir / f"{base_stem}.ply"
    if raw_ply_path.exists():
        raw_ply, raw_points = load_ply_points(raw_ply_path)
    else:
        raw_ply, raw_points = pose_ply, sam_points

    moge_points = load_moge_points(moge_npz)
    if moge_points is None or moge_points.size == 0:
        print("MoGe points not found in npz.")
        return 1

    raw_points = raw_points[np.isfinite(raw_points).all(axis=1)]
    moge_points = moge_points[np.isfinite(moge_points).all(axis=1)]
    if raw_points.size == 0 or moge_points.size == 0:
        print("Empty point set.")
        return 1

    # pose metadata (from sam3d outputs)
    pose_payload = load_pose_payload(sam3d_dir / f"{base_stem}_pose.json")
    pose_r, pose_t, pose_s = parse_pose_payload(pose_payload)
    if pose_r is None or pose_t is None or pose_s is None:
        pose_r = None
        pose_t = None
        pose_s = None
        pose_points = raw_points
    else:
        pose_points = apply_pose(raw_points, pose_s, pose_r, pose_t)

    # 알고리즘용 샘플링
    sam_sample = sample_points(pose_points, args.max_points, args.seed)
    moge_sample = sample_points(moge_points, args.max_points, args.seed + 1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = sam3d_ply.parent.parent / "sam3d_scale"
    output_dir = resolve_path(output_dir, Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = base_stem
    viz_src = pose_points
    viz_scale = None
    refine_registration = args.fine_registration or args.refine_registration

    if args.mode == "scale_only":
        denom = float(np.sum(sam_sample * sam_sample))
        if denom <= 0:
            print("Invalid SAM3D points for scale-only estimation.")
            return 1
        scale_value = float(np.sum(sam_sample * moge_sample) / denom)
        r = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
        result = {
            "scale": scale_value,
            "r": r,
            "t": t,
            "matched_src": None,
            "matched_dst": None,
            "match_mask": None,
            "inlier_indices": None,
            "metrics": {
                "noise_bound": 0.0,
                "corr_count": 0,
                "corr_mode": "scale_only",
                "iterations": 1,
                "voxel_size": 0.0,
                "icp_refine": False,
            },
        }
        if refine_registration:
            try:
                import open3d as o3d
            except Exception:
                print("open3d is required for --refine-registration (ICP). Skipping refine.")
            else:
                sam_sample_scaled = sam_sample * scale_value
                src_pcd = o3d.geometry.PointCloud()
                src_pcd.points = o3d.utility.Vector3dVector(sam_sample_scaled)
                dst_pcd = o3d.geometry.PointCloud()
                dst_pcd.points = o3d.utility.Vector3dVector(moge_sample)

                all_points = np.vstack([sam_sample_scaled, moge_sample])
                diag = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
                thresh = max(1e-6, diag * 0.05)

                init = np.eye(4, dtype=np.float64)
                reg = o3d.pipelines.registration.registration_icp(
                    src_pcd,
                    dst_pcd,
                    thresh,
                    init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=args.icp_max_iters),
                )
                r = reg.transformation[:3, :3].astype(np.float32)
                t = reg.transformation[:3, 3].astype(np.float32)
                result.update(
                    {
                        "r": r,
                        "t": t,
                        "metrics": {
                            **result["metrics"],
                            "corr_mode": "scale_only+icp",
                        },
                    }
                )
                viz_src = pose_points * scale_value
                viz_scale = 1.0
        scale = scale_value
    else:
        result = pick_algorithm(args, sam_sample, moge_sample)
        scale = float(result["scale"])
        r = result.get("r")
        t = result.get("t")
    scale_path = (
        resolve_path(args.output_scale, Path.cwd())
        if args.output_scale
        else output_dir / f"{stem}_scale.txt"
    )
    scale_str = f"{scale:.6f}".rstrip("0").rstrip(".")
    with scale_path.open("w", encoding="utf-8") as f:
        f.write(scale_str + "\n")

    if args.output_scaled_ply is not None:
        scaled_path = resolve_path(args.output_scaled_ply, Path.cwd())
    else:
        scaled_path = output_dir / f"{stem}_scaled.ply"
    write_scaled_ply(raw_ply, raw_points * scale, scaled_path, scale)

    for ext in ("glb", "ply", "obj"):
        src_mesh = sam3d_dir / f"{stem}_mesh.{ext}"
        if src_mesh.exists():
            scaled_mesh = output_dir / f"{stem}_scaled_mesh.{ext}"
            save_scaled_mesh(src_mesh, scaled_mesh, scale)

    if pose_r is None or pose_t is None or pose_s is None:
        print("Pose metadata missing; skipping pose-applied outputs.")
    else:
        pose_only_scale_vec = pose_s
        scaled_pose_vec = pose_s * scale

        r_total = pose_r
        t_total = pose_t
        if refine_registration and r is not None and t is not None:
            r_total = r @ pose_r
            t_total = r @ pose_t + t

        pose_out = {
            "rotation_matrix": r_total.tolist(),
            "translation": t_total.tolist(),
            "scale": scaled_pose_vec.tolist(),
            "base_rotation_matrix": pose_r.tolist(),
            "base_translation": pose_t.tolist(),
            "base_scale": pose_s.tolist(),
            "refine_registration": bool(refine_registration),
        }
        pose_json = output_dir / f"{stem}_pose.json"
        with pose_json.open("w", encoding="utf-8") as f:
            json.dump(pose_out, f, indent=2)

        pose_points = apply_pose(raw_points, scaled_pose_vec, r_total, t_total)
        pose_ply = output_dir / f"{stem}_pose.ply"
        write_pose_ply(raw_ply, pose_points, pose_ply, scaled_pose_vec)

        for ext in ("glb", "ply", "obj"):
            src_mesh = sam3d_dir / f"{stem}_mesh.{ext}"
            if src_mesh.exists():
                pose_mesh = output_dir / f"{stem}_pose_mesh.{ext}"
                save_transformed_mesh(src_mesh, pose_mesh, scaled_pose_vec, r_total, t_total)

        if args.save_posed:
            posed_json = output_dir / f"{stem}_posed.json"
            posed_out = {
                "rotation_matrix": pose_r.tolist(),
                "translation": pose_t.tolist(),
                "scale": pose_only_scale_vec.tolist(),
            }
            with posed_json.open("w", encoding="utf-8") as f:
                json.dump(posed_out, f, indent=2)

            posed_points = apply_pose(raw_points, pose_only_scale_vec, pose_r, pose_t)
            posed_ply = output_dir / f"{stem}_posed.ply"
            write_pose_ply(raw_ply, posed_points, posed_ply, pose_only_scale_vec)
            for ext in ("glb", "ply", "obj"):
                src_mesh = sam3d_dir / f"{stem}_mesh.{ext}"
                if src_mesh.exists():
                    posed_mesh = output_dir / f"{stem}_posed_mesh.{ext}"
                    save_transformed_mesh(src_mesh, posed_mesh, pose_only_scale_vec, pose_r, pose_t)

    if viz_scale is None:
        viz_scale = scale
    if args.show_viz or args.save_viz:
        matched_src = result.get("matched_src")
        matched_dst = result.get("matched_dst")
        match_mask = result.get("match_mask")
        inlier_indices = result.get("inlier_indices")
        if match_mask is not None and matched_src is not None and matched_dst is not None:
            matched_src = matched_src[match_mask]
            matched_dst = matched_dst[match_mask]

        viz_path = None
        if args.save_viz:
            if args.viz_path is not None:
                viz_path = resolve_path(args.viz_path, Path.cwd())
            else:
                viz_path = output_dir / f"{stem}_{args.algo}_match.png"

        if args.show_viz and args.viz_method == "open3d":
            inlier_dst = None
            if matched_dst is not None:
                if inlier_indices is not None:
                    idx = np.asarray(list(inlier_indices), dtype=int)
                    idx = idx[(idx >= 0) & (idx < matched_dst.shape[0])]
                    if idx.size:
                        inlier_dst = matched_dst[idx]
                elif match_mask is not None:
                    inlier_dst = matched_dst
            visualize_alignment_open3d(
                viz_src,
                moge_points,
                viz_scale,
                r,
                t,
                matched_dst=inlier_dst,
                max_points=args.viz_max_points,
                max_spheres=args.viz_max_pairs,
                seed=args.seed,
                title=f"{args.algo} alignment (scale={scale_str})",
            )
        else:
            visualize_alignment(
                viz_src,
                moge_points,
                viz_scale,
                r,
                t,
                matched_src=matched_src,
                matched_dst=matched_dst,
                max_points=args.viz_max_points,
                max_pairs=args.viz_max_pairs,
                seed=args.seed,
                title=f"{args.algo} alignment (scale={scale_str})",
                save_path=viz_path,
                show=args.show_viz,
                dpi=args.viz_dpi,
            )

        if args.save_viz and args.viz_method == "open3d":
            visualize_alignment(
                viz_src,
                moge_points,
                viz_scale,
                r,
                t,
                matched_src=matched_src,
                matched_dst=matched_dst,
                max_points=args.viz_max_points,
                max_pairs=args.viz_max_pairs,
                seed=args.seed,
                title=f"{args.algo} alignment (scale={scale_str})",
                save_path=viz_path,
                show=False,
                dpi=args.viz_dpi,
            )

    if args.debug_viz:
        visualize_debug_views(
            moge_points,
            viz_src,
            scale,
            r,
            t,
            max_points=args.viz_max_points,
            seed=args.seed,
            title_prefix=f"{args.algo} ",
            show=True,
        )

    # 표준 출력은 스케일 값만 출력
    print(scale_str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
