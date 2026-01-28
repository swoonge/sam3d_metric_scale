"""MoGe 기반 metric depth/point 결과를 마스크 영역으로 제한하고 스케일을 추정.

- 입력: 원본 이미지 + SAM2 마스크
- 출력: 마스크 영역의 depth/points + bbox 기반 스케일 통계(JSON/NPZ)
- 경계/MAD 기반 전처리를 적용해 포인트를 저장
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def resolve_moge_root(repo_root: Path) -> Path:
    """MoGe 레포 위치를 환경변수/기본 후보에서 탐색."""
    env_root = os.environ.get("MOGE_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(repo_root / "MoGe")
    candidates.append(repo_root.parent / "MoGe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else repo_root / "MoGe"


def parse_args() -> argparse.Namespace:
    """CLI 인자 정의."""
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "outputs" / "moge_scale"

    parser = argparse.ArgumentParser(description="Masked MoGe depth/points + scale estimate")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="Ruicheng/moge-2-vitl-normal",
        help="HF model id or local path.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--no-save-npz", action="store_true")
    parser.add_argument("--no-save-ply", action="store_true")
    parser.add_argument("--scale-method", choices=["bbox_diag", "bbox_max"], default="bbox_diag")
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--cam-k", type=Path, default=None)
    parser.add_argument(
        "--border-margin",
        type=int,
        default=5,
        help="경계에서 제외할 픽셀 두께(0이면 비활성).",
    )
    parser.add_argument(
        "--depth-mad",
        type=float,
        default=3.5,
        help="깊이(z) MAD 아웃라이어 제거 임계값(<=0 비활성).",
    )
    parser.add_argument(
        "--radius-mad",
        type=float,
        default=3.5,
        help="중심 거리 MAD 아웃라이어 제거 임계값(<=0 비활성).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=500,
        help="필터 후 최소 포인트 수(미만이면 필터를 무시).",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--show-viz", action="store_true")
    parser.add_argument("--viz-path", type=Path, default=None)
    parser.add_argument("--viz-dpi", type=int, default=150)
    parser.add_argument("--viz-max-points", type=int, default=20000)
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    """상대 경로를 절대 경로로 변환."""
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_mask(path: Path) -> np.ndarray:
    """마스크 이미지를 로드하여 bool 배열로 반환."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[:, :, -1]
    return mask > 0


def parse_cam_k(cam_k_path: Path) -> tuple[float, float, float, float]:
    """cam_K.txt(3x3 or fx fy cx cy) 로드."""
    k_mat = np.loadtxt(str(cam_k_path)).astype(np.float32)
    if k_mat.size == 4:
        fx, fy, cx, cy = [float(v) for v in k_mat.ravel()]
        return fx, fy, cx, cy
    k_mat = k_mat.reshape(3, 3)
    return float(k_mat[0, 0]), float(k_mat[1, 1]), float(k_mat[0, 2]), float(k_mat[1, 2])


def backproject_depth(
    depth: np.ndarray, valid_mask: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Depth + intrinsics 로 카메라 좌표계 포인트 생성."""
    ys, xs = np.where(valid_mask)
    z = depth[valid_mask].astype(np.float32)
    x = (xs - cx) * z / fx
    y = -(ys - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def compute_scale(points: np.ndarray, method: str) -> dict:
    """포인트 bbox 기반 스케일 통계를 계산."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    size = maxs - mins
    diag = float(np.linalg.norm(size))
    max_dim = float(size.max())
    if method == "bbox_max":
        value = max_dim
    else:
        value = diag
    return {
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
        "bbox_size": size.tolist(),
        "bbox_diag": diag,
        "bbox_max_dim": max_dim,
        "scale_method": method,
        "scale_value": value,
    }


def mad_keep_mask(values: np.ndarray, thresh: float) -> np.ndarray:
    """MAD 기반 keep mask 생성."""
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
    """경계 제거 + MAD 기반 아웃라이어 제거 keep mask 생성."""
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


def save_points_ply(points: np.ndarray, path: Path) -> None:
    """(N, 3) 포인트를 ASCII PLY로 저장."""
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


def build_visualization(
    image_rgb: np.ndarray,
    user_mask: np.ndarray,
    valid_mask: np.ndarray,
    depth: np.ndarray,
    combined_mask: np.ndarray,
    points_masked: np.ndarray,
    stats: dict,
    max_points: int,
):
    """요약 시각화(이미지/마스크/깊이/포인트클라우드) 생성."""
    import matplotlib.pyplot as plt

    overlay = image_rgb.copy()
    if user_mask.any():
        color = np.array([0, 255, 0], dtype=np.uint8)
        overlay[user_mask] = (
            0.6 * overlay[user_mask] + 0.4 * color
        ).astype(np.uint8)

    depth_vis = depth.copy()
    depth_vis[~valid_mask] = np.nan
    masked_depth_vis = np.full_like(depth, np.nan)
    masked_depth_vis[combined_mask] = depth[combined_mask]

    valid_vals = depth[combined_mask]
    if valid_vals.size > 0:
        vmin, vmax = np.percentile(valid_vals, [2, 98])
    else:
        vmin, vmax = float(np.nanmin(depth_vis)), float(np.nanmax(depth_vis))

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_depth = fig.add_subplot(gs[1, 0])
    ax_depth_masked = fig.add_subplot(gs[1, 1])
    ax_pc = fig.add_subplot(gs[:, 2], projection="3d")

    ax_img.imshow(image_rgb)
    ax_img.set_title("image")
    ax_img.axis("off")

    ax_overlay.imshow(overlay)
    ax_overlay.set_title("mask overlay")
    ax_overlay.axis("off")

    ax_depth.imshow(depth_vis, cmap="magma", vmin=vmin, vmax=vmax)
    ax_depth.set_title("depth (valid)")
    ax_depth.axis("off")

    ax_depth_masked.imshow(masked_depth_vis, cmap="magma", vmin=vmin, vmax=vmax)
    ax_depth_masked.set_title("depth (masked)")
    ax_depth_masked.axis("off")

    if points_masked.size > 0:
        # 포인트 수가 많을 경우 샘플링
        points_vis = points_masked
        if max_points > 0 and points_masked.shape[0] > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(points_masked.shape[0], size=max_points, replace=False)
            points_vis = points_masked[idx]

        colors = points_vis[:, 2]
        ax_pc.scatter(
            points_vis[:, 0],
            points_vis[:, 1],
            points_vis[:, 2],
            c=colors,
            s=1,
            cmap="viridis",
        )
        mins = points_vis.min(axis=0)
        maxs = points_vis.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(1e-6, float((maxs - mins).max() / 2.0))
        ax_pc.set_xlim(center[0] - radius, center[0] + radius)
        ax_pc.set_ylim(center[1] - radius, center[1] + radius)
        ax_pc.set_zlim(center[2] - radius, center[2] + radius)
    ax_pc.set_title("point cloud (masked)")
    ax_pc.set_xlabel("x")
    ax_pc.set_ylabel("y")
    ax_pc.set_zlabel("z")

    fig.suptitle(
        f"scale={stats['scale_value']:.4f} ({stats['scale_method']})  "
        f"points={stats['points_count']}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> int:
    """MoGe 추론 및 마스크 기반 스케일 계산 파이프라인."""
    args = parse_args()
    image_path = resolve_path(args.image, Path.cwd())
    mask_path = resolve_path(args.mask, Path.cwd())
    output_dir = resolve_path(args.output_dir, Path.cwd())

    repo_root = Path(__file__).resolve().parents[1]
    moge_root = resolve_moge_root(repo_root)
    # MoGe 패키지 import를 위해 경로 등록
    if str(moge_root) not in sys.path:
        sys.path.insert(0, str(moge_root))
    try:
        from moge.model.v2 import MoGeModel
    except ModuleNotFoundError as exc:
        print(
            "MoGe module not found. Install it with `pip install -e ./MoGe` "
            "or set MOGE_ROOT to the MoGe repository path."
        )
        raise exc

    if not image_path.exists():
        print(f"Missing image: {image_path}")
        return 1
    if not mask_path.exists():
        print(f"Missing mask: {mask_path}")
        return 1
    if args.cam_k is not None:
        cam_k_path = resolve_path(args.cam_k, Path.cwd())
        if not cam_k_path.exists():
            print(f"Missing cam_K: {cam_k_path}")
            return 1
    else:
        cam_k_path = None

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Failed to read image: {image_path}")
        return 1
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    model = MoGeModel.from_pretrained(args.model).to(device)
    model.eval()

    input_tensor = torch.tensor(
        img_rgb / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1)

    with torch.inference_mode():
        output = model.infer(input_tensor)

    points = output["points"].detach().cpu().numpy()
    depth = output["depth"].detach().cpu().numpy()
    valid = output["mask"].detach().cpu().numpy() > 0
    intrinsics = output.get("intrinsics")
    if intrinsics is not None:
        intrinsics = intrinsics.detach().cpu().numpy()

    user_mask = load_mask(mask_path)
    if user_mask.shape != depth.shape:
        # depth 해상도와 mask 해상도가 다르면 최근접 리사이즈
        user_mask = cv2.resize(
            user_mask.astype(np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    finite = np.isfinite(points).all(axis=2) & np.isfinite(depth) & (depth > 0)
    valid_full = finite & valid
    points_source = "moge_points"
    fx = fy = cx = cy = None
    if cam_k_path is not None:
        fx, fy, cx, cy = parse_cam_k(cam_k_path)
        points_full = backproject_depth(depth, valid_full, fx, fy, cx, cy)
        points_cam = np.full(points.shape, np.nan, dtype=np.float32)
        ys_all, xs_all = np.where(finite)
        pts_all = backproject_depth(depth, finite, fx, fy, cx, cy)
        points_cam[ys_all, xs_all] = pts_all
        points = points_cam
        points_source = "cam_k_provided"
    elif intrinsics is not None:
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics.ndim == 3:
            intrinsics = intrinsics[0]
        if intrinsics.shape == (3, 3):
            height, width = depth.shape
            fx = float(intrinsics[0, 0] * width)
            fy = float(intrinsics[1, 1] * height)
            cx = float(intrinsics[0, 2] * width)
            cy = float(intrinsics[1, 2] * height)
            points_full = backproject_depth(depth, valid_full, fx, fy, cx, cy)
            points_cam = np.full(points.shape, np.nan, dtype=np.float32)
            ys_all, xs_all = np.where(finite)
            pts_all = backproject_depth(depth, finite, fx, fy, cx, cy)
            points_cam[ys_all, xs_all] = pts_all
            points = points_cam
            points_source = "intrinsics_estimated"
        else:
            points_full = points[valid_full]
    else:
        points_full = points[valid_full]

    # user_mask + MoGe valid 마스크 + finite 조건을 모두 만족하는 영역만 사용
    combined = user_mask & valid_full

    if combined.sum() < args.min_pixels:
        print(f"Not enough valid pixels: {combined.sum()}")
        return 1

    ys, xs = np.where(combined)
    masked_points = points[combined]
    masked_depth = depth[combined]

    keep = build_filter_keep_mask(
        masked_points,
        ys,
        xs,
        combined.shape,
        args.border_margin,
        args.depth_mad,
        args.radius_mad,
    )
    filter_applied = int(keep.sum()) >= args.min_points
    if not filter_applied:
        keep = np.ones(masked_points.shape[0], dtype=bool)

    filtered_points = masked_points[keep]
    filtered_depth = masked_depth[keep]
    filtered_valid = combined.copy()
    if keep.shape[0] == ys.shape[0]:
        filtered_valid[ys[~keep], xs[~keep]] = False
    if filtered_points.size == 0:
        filtered_points = masked_points
        filtered_depth = masked_depth
        filtered_valid = combined
        filter_applied = False

    # 통계값 수집
    stats = {
        "image": str(image_path),
        "mask": str(mask_path),
        "model": args.model,
        "points_source": points_source,
        "points_count_raw": int(masked_points.shape[0]),
        "points_count": int(filtered_points.shape[0]),
        "filter_applied": bool(filter_applied),
        "filter_border_margin": int(args.border_margin),
        "filter_depth_mad": float(args.depth_mad),
        "filter_radius_mad": float(args.radius_mad),
        "filter_min_points": int(args.min_points),
        "depth_mean": float(filtered_depth.mean()),
        "depth_median": float(np.median(filtered_depth)),
        "depth_min": float(filtered_depth.min()),
        "depth_max": float(filtered_depth.max()),
    }
    if fx is not None and fy is not None and cx is not None and cy is not None:
        stats["camera_fx"] = float(fx)
        stats["camera_fy"] = float(fy)
        stats["camera_cx"] = float(cx)
        stats["camera_cy"] = float(cy)
    stats.update(compute_scale(filtered_points, args.scale_method))

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    stem = f"{image_path.stem}_{mask_path.stem}{suffix}"
    if args.output_json is not None:
        json_path = resolve_path(args.output_json, Path.cwd())
    else:
        json_path = output_dir / f"{stem}.json"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if not args.no_save_npz:
        # 후속 단계(스케일 추정/시각화)를 위해 NPZ 저장
        npz_path = output_dir / f"{stem}.npz"
        np.savez_compressed(
            npz_path,
            mask=user_mask.astype(np.uint8),
            valid_mask=filtered_valid.astype(np.uint8),
            points_masked=filtered_points.astype(np.float32),
            depth_masked=filtered_depth.astype(np.float32),
            valid_full=valid_full.astype(np.uint8),
            points_full=points_full.astype(np.float32),
            depth_full=depth.astype(np.float32),
        )
    else:
        npz_path = None

    ply_path = None
    full_ply_path = None
    if not args.no_save_ply:
        # 외부 툴에서 바로 확인할 수 있도록 PLY도 저장
        ply_path = output_dir / f"{stem}.ply"
        save_points_ply(filtered_points, ply_path)
        full_ply_path = output_dir / f"{stem}_full.ply"
        save_points_ply(points_full, full_ply_path)

    print(f"Saved scale stats: {json_path}")
    if not args.no_save_npz:
        print(f"Saved masked outputs: {npz_path}")
    if ply_path is not None:
        print(f"Saved MoGe point cloud: {ply_path}")
    if full_ply_path is not None:
        print(f"Saved MoGe full point cloud: {full_ply_path}")

    if args.save_viz or args.show_viz:
        # 옵션일 때만 matplotlib 시각화 생성
        import matplotlib.pyplot as plt

        fig = build_visualization(
            img_rgb,
            user_mask,
            valid,
            depth,
            filtered_valid,
            filtered_points,
            stats,
            args.viz_max_points,
        )
        if args.save_viz:
            if args.viz_path is not None:
                viz_path = resolve_path(args.viz_path, Path.cwd())
            else:
                viz_path = output_dir / f"{stem}_viz.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(viz_path, dpi=args.viz_dpi, bbox_inches="tight")
            print(f"Saved visualization: {viz_path}")
        if args.show_viz:
            plt.show()
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
