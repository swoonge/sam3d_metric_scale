"""SAM3D PLY에 MoGe 포인트클라우드 스케일을 맞추기 위한 스케일 추정 스크립트.

핵심 아이디어:
- SAM3D PLY는 스케일이 없는 반면, MoGe는 metric depth 기반의 포인트클라우드를 제공한다.
- 두 포인트셋 사이의 스케일을 Kabsch-Umeyama(유사변환) 기반 ICP로 추정한다.
- 추정된 스케일만 텍스트로 저장하고, 원본 SAM3D PLY에 스케일만 적용해 저장한다.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    """CLI 인자를 정의한다."""
    parser = argparse.ArgumentParser(
        description="Estimate SAM3D scale using MoGe point cloud (Umeyama ICP)."
    )
    parser.add_argument("--sam3d-ply", type=Path, required=True)
    parser.add_argument("--moge-npz", type=Path, required=True)
    parser.add_argument("--output-scale", type=Path, default=None)
    parser.add_argument("--output-scaled-ply", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--nn-max-points", type=int, default=4000)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--use-ransac",
        action="store_true",
        help="Umeyama 기반 RANSAC을 사용(기본: 비활성).",
    )
    parser.add_argument(
        "--ransac-iters",
        type=int,
        default=200,
        help="RANSAC 반복 횟수.",
    )
    parser.add_argument(
        "--ransac-sample",
        type=int,
        default=64,
        help="RANSAC 1회에서 사용할 샘플 수.",
    )
    parser.add_argument(
        "--ransac-inlier-thresh",
        type=float,
        default=0.02,
        help="RANSAC inlier 거리 임계값(미터 단위 추정).",
    )
    parser.add_argument(
        "--border-margin",
        type=int,
        default=5,
        help="이미지 경계에서 제외할 픽셀 두께(0=비활성).",
    )
    parser.add_argument(
        "--depth-mad",
        type=float,
        default=3.5,
        help="깊이(z) 아웃라이어 제거용 MAD 임계값(<=0 비활성).",
    )
    parser.add_argument(
        "--radius-mad",
        type=float,
        default=3.5,
        help="중심 거리 아웃라이어 제거용 MAD 임계값(<=0 비활성).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=500,
        help="필터링 후 최소 포인트 수(미만이면 필터를 완화).",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    """상대 경로를 실행 디렉터리 기준 절대 경로로 변환."""
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_ply_points(path: Path) -> Tuple[object, np.ndarray]:
    """PLY 파일에서 vertex 좌표만 로드."""
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("plyfile is required to read/write SAM3D ply files.") from exc

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise RuntimeError("PLY file missing vertex data.")
    vertex = ply["vertex"].data
    points = np.stack(
        (vertex["x"], vertex["y"], vertex["z"]),
        axis=1,
    ).astype(np.float32)
    return ply, points


def write_scaled_ply(ply, points: np.ndarray, out_path: Path) -> None:
    """기존 PLY 구조를 유지한 채 vertex 좌표만 교체해 저장."""
    from plyfile import PlyData, PlyElement

    vertex = ply["vertex"].data.copy()
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]

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


def load_moge_points(
    npz_path: Path,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]]:
    """MoGe NPZ에서 포인트를 로드하고, 픽셀 좌표 매핑을 함께 반환."""
    if npz_path is None or not npz_path.exists():
        return None, None
    data = np.load(npz_path)
    points = data.get("points_masked")
    valid_mask = data.get("valid_mask")
    pixel_coords = None
    if points is not None:
        points = points.astype(np.float32)
        points = points[np.isfinite(points).all(axis=1)]
        if valid_mask is not None:
            valid_mask = valid_mask.astype(bool)
            ys, xs = np.where(valid_mask)
            if points.shape[0] == ys.shape[0]:
                pixel_coords = (ys, xs, valid_mask.shape)
        return points, pixel_coords

    depth_masked = data.get("depth_masked")
    if valid_mask is None or depth_masked is None:
        return None, None
    valid_mask = valid_mask.astype(bool)
    depth_masked = depth_masked.astype(np.float32)
    if depth_masked.shape[0] != int(valid_mask.sum()):
        return None, None
    # 유효 픽셀 위치를 이용해 간단한 pinhole 기반 3D 좌표 복원(근사)
    ys, xs = np.where(valid_mask)
    height, width = valid_mask.shape
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    focal = max(height, width)
    z = depth_masked
    x = (xs - cx) * z / focal
    y = -(ys - cy) * z / focal
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    pixel_coords = (ys, xs, valid_mask.shape)
    return points, pixel_coords


def filter_border_points(
    points: np.ndarray,
    pixel_coords: Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]],
    margin: int,
) -> np.ndarray:
    """이미지 경계 근처 포인트를 제거."""
    if margin <= 0 or pixel_coords is None:
        return points
    ys, xs, shape = pixel_coords
    height, width = shape
    keep = (
        (ys >= margin)
        & (ys < height - margin)
        & (xs >= margin)
        & (xs < width - margin)
    )
    if keep.shape[0] != points.shape[0]:
        return points
    return points[keep]


def mad_keep_mask(values: np.ndarray, thresh: float) -> np.ndarray:
    """MAD 기반으로 아웃라이어를 제거하기 위한 keep mask 생성."""
    if thresh <= 0:
        return np.ones(values.shape[0], dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-8:
        return np.ones(values.shape[0], dtype=bool)
    z_score = 0.6745 * (values - median) / mad
    return np.abs(z_score) <= thresh


def filter_outliers(points: np.ndarray, depth_mad: float, radius_mad: float) -> np.ndarray:
    """깊이(z) + 중심 거리 기반 아웃라이어 제거."""
    if points.size == 0:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    if depth_mad > 0:
        keep &= mad_keep_mask(points[:, 2], depth_mad)
    if radius_mad > 0:
        center = np.median(points, axis=0)
        radius = np.linalg.norm(points - center, axis=1)
        keep &= mad_keep_mask(radius, radius_mad)
    return points[keep]


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """포인트 수가 많을 경우 샘플링으로 계산 비용 제한."""
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(0)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Kabsch-Umeyama 기반 유사변환(스케일/회전/이동) 추정."""
    if src.shape[0] == 0 or dst.shape[0] == 0:
        raise ValueError("Empty point set.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = (dst_centered.T @ src_centered) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3, dtype=np.float32)
    # 반사(reflection) 제거를 위해 determinant 부호 보정
    if np.linalg.det(u @ vt) < 0:
        s[2, 2] = -1.0
    r = (u @ s @ vt).astype(np.float32)

    var_src = np.mean(np.sum(src_centered**2, axis=1))
    # 스케일은 분산과 특이값의 비율로 추정
    scale = float(np.sum(d * np.diag(s)) / max(var_src, 1e-8))
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t.astype(np.float32)


def nearest_neighbors(src: np.ndarray, dst: np.ndarray, chunk: int = 512):
    """브루트포스 최근접 탐색(메모리 폭주 방지용 chunk 처리)."""
    indices = np.empty(src.shape[0], dtype=np.int64)
    distances = np.empty(src.shape[0], dtype=np.float32)
    for start in range(0, src.shape[0], chunk):
        end = min(start + chunk, src.shape[0])
        diff = src[start:end, None, :] - dst[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        idx = np.argmin(dist, axis=1)
        indices[start:end] = idx
        distances[start:end] = dist[np.arange(end - start), idx]
    return indices, distances


def estimate_scale_icp(
    src: np.ndarray,
    dst: np.ndarray,
    max_iters: int,
    tol: float,
    nn_max_points: int,
) -> Tuple[float, np.ndarray, np.ndarray, float, int]:
    """ICP 방식으로 Umeyama 정합을 반복하여 스케일을 추정."""
    src_sample = sample_points(src, nn_max_points)
    dst_sample = sample_points(dst, nn_max_points)

    # 초기값: 두 포인트셋의 RMS 크기 비율
    mu_src = src_sample.mean(axis=0)
    mu_dst = dst_sample.mean(axis=0)
    rms_src = float(np.sqrt(np.mean(np.sum((src_sample - mu_src) ** 2, axis=1))))
    rms_dst = float(np.sqrt(np.mean(np.sum((dst_sample - mu_dst) ** 2, axis=1))))
    scale = rms_dst / max(rms_src, 1e-8)
    r = np.eye(3, dtype=np.float32)
    t = mu_dst - scale * (r @ mu_src)

    prev_error = None
    for it in range(max_iters):
        # 현재 추정값으로 변환 후 최근접 매칭 구성
        transformed = scale * (src_sample @ r.T) + t
        nn_idx, nn_dist = nearest_neighbors(transformed, dst_sample)
        matched = dst_sample[nn_idx]
        # 매칭 쌍을 대상으로 Umeyama 업데이트
        scale, r, t = umeyama_alignment(src_sample, matched)
        rmse = float(np.sqrt(np.mean(nn_dist**2)))
        if prev_error is not None and abs(prev_error - rmse) < tol:
            return scale, r, t, rmse, it + 1
        prev_error = rmse
    return scale, r, t, float(prev_error or 0.0), max_iters


def estimate_scale_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    iters: int,
    sample_size: int,
    inlier_thresh: float,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """Umeyama를 RANSAC으로 감싸 아웃라이어에 강인한 스케일 추정."""
    if src.shape[0] < 3 or dst.shape[0] < 3:
        raise ValueError("Not enough points for RANSAC.")

    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    rng = np.random.default_rng(0)

    best_inliers = 0
    best_scale = 1.0
    best_r = np.eye(3, dtype=np.float32)
    best_t = np.zeros(3, dtype=np.float32)

    sample_size = max(3, min(sample_size, src.shape[0]))
    for _ in range(iters):
        idx = rng.choice(src.shape[0], size=sample_size, replace=False)
        try:
            scale, r, t = umeyama_alignment(src[idx], dst[idx])
        except ValueError:
            continue
        transformed = scale * (src @ r.T) + t
        dist = np.linalg.norm(transformed - dst, axis=1)
        inliers = int(np.sum(dist <= inlier_thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_scale, best_r, best_t = scale, r, t

    if best_inliers == 0:
        # fallback: 전체 포인트로 한번 추정
        best_scale, best_r, best_t = umeyama_alignment(src, dst)

    return best_scale, best_r, best_t, best_inliers


def main() -> int:
    """전체 파이프라인 실행."""
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
    moge_points, pixel_coords = load_moge_points(moge_npz)
    if moge_points is None or moge_points.size == 0:
        print("MoGe points not found in npz.")
        return 1

    sam_points = sam_points[np.isfinite(sam_points).all(axis=1)]
    moge_points = moge_points[np.isfinite(moge_points).all(axis=1)]
    if sam_points.size == 0 or moge_points.size == 0:
        print("Empty point set after filtering.")
        return 1

    # 경계/아웃라이어 제거(선택)
    filtered_moge = filter_border_points(moge_points, pixel_coords, args.border_margin)
    filtered_moge = filter_outliers(filtered_moge, args.depth_mad, args.radius_mad)
    if filtered_moge.shape[0] < args.min_points:
        # 너무 적으면 원본 포인트를 사용(스케일 계산 실패 방지)
        filtered_moge = moge_points

    # 계산량을 줄이기 위해 샘플링
    sam_sample = sample_points(sam_points, args.max_points)
    moge_sample = sample_points(filtered_moge, args.max_points)

    # Umeyama ICP 또는 RANSAC으로 스케일 추정
    if args.use_ransac:
        scale, _, _, _ = estimate_scale_ransac(
            sam_sample,
            moge_sample,
            args.ransac_iters,
            args.ransac_sample,
            args.ransac_inlier_thresh,
        )
    else:
        scale, _, _, _, _ = estimate_scale_icp(
            sam_sample,
            moge_sample,
            args.max_iters,
            args.tolerance,
            args.nn_max_points,
        )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = sam3d_ply.parent.parent / "sam3d_scale"
    output_dir = resolve_path(output_dir, Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = sam3d_ply.stem
    scale_path = (
        resolve_path(args.output_scale, Path.cwd())
        if args.output_scale
        else output_dir / f"{stem}_scale.txt"
    )
    scale_str = f"{scale:.6f}".rstrip("0").rstrip(".")
    with scale_path.open("w", encoding="utf-8") as f:
        f.write(scale_str + "\n")

    # 스케일만 적용한 PLY 저장(회전/이동 미적용)
    if args.output_scaled_ply is not None:
        scaled_path = resolve_path(args.output_scaled_ply, Path.cwd())
    else:
        scaled_path = output_dir / f"{stem}_scaled.ply"
    scaled_points = sam_points * scale
    write_scaled_ply(ply, scaled_points, scaled_path)
    # 표준 출력은 스케일 값만 출력(요청 사항)
    print(scale_str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
