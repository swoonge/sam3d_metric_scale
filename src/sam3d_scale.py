"""SAM3D 스케일 추정 알고리즘 테스트 러너.

알고리즘 구현은 별도 파일로 분리되어 있으며, 여기서는 공통 입출력/시각화를 담당한다.
"""

from __future__ import annotations

import argparse
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
        default="icp",
        help="Scale estimation algorithm.",
    )
    parser.add_argument("--output-scale", type=Path, default=None)
    parser.add_argument("--output-scaled-ply", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)

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
        choices=["fpfh"],
        default="fpfh",
        help="Correspondence mode for TEASER++.",
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
    parser.add_argument("--show-viz", action="store_true")
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
        ply, sam_points = load_ply_points(sam3d_ply)
    except RuntimeError as exc:
        print(exc)
        print("Install with: conda run -n sam3d-objects python -m pip install plyfile")
        return 1

    moge_points = load_moge_points(moge_npz)
    if moge_points is None or moge_points.size == 0:
        print("MoGe points not found in npz.")
        return 1

    sam_points = sam_points[np.isfinite(sam_points).all(axis=1)]
    moge_points = moge_points[np.isfinite(moge_points).all(axis=1)]
    if sam_points.size == 0 or moge_points.size == 0:
        print("Empty point set.")
        return 1

    # 알고리즘용 샘플링
    sam_sample = sample_points(sam_points, args.max_points, args.seed)
    moge_sample = sample_points(moge_points, args.max_points, args.seed + 1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = sam3d_ply.parent.parent / "sam3d_scale"
    output_dir = resolve_path(output_dir, Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = sam3d_ply.stem
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
    write_scaled_ply(ply, sam_points * scale, scaled_path, scale)

    viz_src = sam_points
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
                scale,
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
                scale,
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
                scale,
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
