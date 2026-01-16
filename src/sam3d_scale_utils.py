"""SAM3D 스케일 추정 공통 유틸리티."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


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


def load_moge_points(npz_path: Path) -> Optional[np.ndarray]:
    """MoGe NPZ에서 포인트를 로드(없으면 depth로 복원)."""
    if npz_path is None or not npz_path.exists():
        return None
    data = np.load(npz_path)
    points = data.get("points_masked")
    if points is not None:
        points = points.astype(np.float32)
        return points[np.isfinite(points).all(axis=1)]

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
    focal = max(height, width)
    z = depth_masked
    x = (xs - cx) * z / focal
    y = -(ys - cy) * z / focal
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    return points[np.isfinite(points).all(axis=1)]


def sample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """포인트 수가 많을 경우 샘플링으로 계산 비용 제한."""
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def estimate_rms_scale(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """두 포인트셋의 RMS 크기 비율로 스케일 초기값을 계산."""
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    rms_src = float(np.sqrt(np.mean(np.sum((src - mu_src) ** 2, axis=1))))
    rms_dst = float(np.sqrt(np.mean(np.sum((dst - mu_dst) ** 2, axis=1))))
    scale = rms_dst / max(rms_src, 1e-8)
    r = np.eye(3, dtype=np.float32)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t.astype(np.float32)


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
    if np.linalg.det(u @ vt) < 0:
        s[2, 2] = -1.0
    r = (u @ s @ vt).astype(np.float32)

    var_src = np.mean(np.sum(src_centered**2, axis=1))
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


def apply_similarity(
    points: np.ndarray, scale: float, r: Optional[np.ndarray], t: Optional[np.ndarray]
) -> np.ndarray:
    """스케일/회전/이동을 적용."""
    if r is None:
        r = np.eye(3, dtype=np.float32)
    if t is None:
        t = np.zeros(3, dtype=np.float32)
    return scale * (points @ r.T) + t


def visualize_alignment(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    scale: float,
    r: Optional[np.ndarray],
    t: Optional[np.ndarray],
    matched_src: Optional[np.ndarray] = None,
    matched_dst: Optional[np.ndarray] = None,
    max_points: int = 5000,
    max_pairs: int = 200,
    seed: int = 0,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    dpi: int = 150,
) -> bool:
    """정합 결과를 3D 스캐터로 시각화."""
    if save_path is None and not show:
        return False

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required for visualization.")
        return False

    src_vis = sample_points(src_points, max_points, seed)
    dst_vis = sample_points(dst_points, max_points, seed + 1)
    src_vis_t = apply_similarity(src_vis, scale, r, t)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        dst_vis[:, 0],
        dst_vis[:, 1],
        dst_vis[:, 2],
        s=1,
        c="#2E7D32",
        alpha=0.6,
        label="MoGe",
    )
    ax.scatter(
        src_vis_t[:, 0],
        src_vis_t[:, 1],
        src_vis_t[:, 2],
        s=1,
        c="#EF6C00",
        alpha=0.6,
        label="SAM3D (aligned)",
    )

    if matched_src is not None and matched_dst is not None:
        match_src = matched_src
        match_dst = matched_dst
        if match_src.shape[0] > max_pairs:
            rng = np.random.default_rng(seed)
            idx = rng.choice(match_src.shape[0], size=max_pairs, replace=False)
            match_src = match_src[idx]
            match_dst = match_dst[idx]
        match_src_t = apply_similarity(match_src, scale, r, t)
        for i in range(match_src_t.shape[0]):
            ax.plot(
                [match_src_t[i, 0], match_dst[i, 0]],
                [match_src_t[i, 1], match_dst[i, 1]],
                [match_src_t[i, 2], match_dst[i, 2]],
                color="#9E9E9E",
                linewidth=0.5,
                alpha=0.6,
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title or "Alignment view")
    ax.legend(loc="upper right")
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return True


def visualize_alignment_open3d(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    scale: float,
    r: Optional[np.ndarray],
    t: Optional[np.ndarray],
    matched_dst: Optional[np.ndarray] = None,
    max_points: int = 5000,
    max_spheres: int = 200,
    seed: int = 0,
    title: str = "TEASER++ alignment",
) -> bool:
    """Open3D 기반 정합 시각화."""
    try:
        import open3d as o3d
    except Exception:
        print("open3d is required for Open3D visualization.")
        return False

    src_vis = sample_points(src_points, max_points, seed)
    dst_vis = sample_points(dst_points, max_points, seed + 1)

    src_cloud = o3d.geometry.PointCloud()
    src_cloud.points = o3d.utility.Vector3dVector(src_vis)
    dst_cloud = o3d.geometry.PointCloud()
    dst_cloud.points = o3d.utility.Vector3dVector(dst_vis)

    src_cloud.paint_uniform_color([0.0, 0.629, 0.9])
    dst_cloud.paint_uniform_color([1.0, 0.3, 0.05])

    r_eff = np.eye(3, dtype=np.float32) if r is None else r
    t_eff = np.zeros(3, dtype=np.float32) if t is None else t
    scale_eff = float(scale) if abs(scale) > 1e-8 else 1e-8
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = scale_eff * r_eff
    transform[:3, 3] = t_eff
    src_cloud.transform(transform)

    geoms = [dst_cloud, src_cloud]

    if matched_dst is not None and matched_dst.size > 0:
        rng = np.random.default_rng(seed)
        dst_spheres = matched_dst
        if dst_spheres.shape[0] > max_spheres:
            idx = rng.choice(dst_spheres.shape[0], size=max_spheres, replace=False)
            dst_spheres = dst_spheres[idx]
        mins = dst_vis.min(axis=0)
        maxs = dst_vis.max(axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        radius = max(1e-6, diag * 0.01)
        for point in dst_spheres:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([0.0, 0.9, 0.1])
            sphere.translate(point.astype(np.float64))
            geoms.append(sphere)

    o3d.visualization.draw_geometries(geoms, window_name=title)
    return True


def visualize_debug_views(
    moge_points: np.ndarray,
    sam_points: np.ndarray,
    scale: float,
    r: Optional[np.ndarray],
    t: Optional[np.ndarray],
    max_points: int = 5000,
    seed: int = 0,
    title_prefix: str = "",
    show: bool = True,
) -> bool:
    """Debug view: include inverse transform panels for direction check."""
    if not show:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required for visualization.")
        return False

    moge_vis = sample_points(moge_points, max_points, seed)
    sam_vis = sample_points(sam_points, max_points, seed + 1)

    r_eff = np.eye(3, dtype=np.float32) if r is None else r
    t_eff = np.zeros(3, dtype=np.float32) if t is None else t
    scale_eff = float(scale) if abs(scale) > 1e-8 else 1e-8

    sam_to_moge = apply_similarity(sam_vis, scale_eff, r_eff, t_eff)
    moge_to_sam = apply_similarity(moge_vis, scale_eff, r_eff, t_eff)
    inv_scale = 1.0 / scale_eff
    sam_inv = (sam_vis - t_eff) @ r_eff * inv_scale
    moge_inv = (moge_vis - t_eff) @ r_eff * inv_scale

    fig = plt.figure(figsize=(18, 9))
    axes = [
        fig.add_subplot(2, 4, 1, projection="3d"),
        fig.add_subplot(2, 4, 2, projection="3d"),
        fig.add_subplot(2, 4, 3, projection="3d"),
        fig.add_subplot(2, 4, 4, projection="3d"),
        fig.add_subplot(2, 4, 5, projection="3d"),
        fig.add_subplot(2, 4, 6, projection="3d"),
        fig.add_subplot(2, 4, 7, projection="3d"),
    ]
    ax_blank = fig.add_subplot(2, 4, 8)
    ax_blank.axis("off")

    def _scatter(ax, pts, color, label):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=color, alpha=0.6, label=label)

    _scatter(axes[0], moge_vis, "#2E7D32", "MoGe")
    axes[0].set_title(f"{title_prefix}1) MoGe")

    _scatter(axes[1], sam_vis, "#EF6C00", "SAM3D cropped")
    axes[1].set_title(f"{title_prefix}2) SAM3D cropped")

    _scatter(axes[2], moge_vis, "#2E7D32", "MoGe")
    _scatter(axes[2], sam_vis, "#EF6C00", "SAM3D cropped")
    axes[2].set_title(f"{title_prefix}3) Combined")

    _scatter(axes[3], moge_to_sam, "#1B5E20", "MoGe transformed")
    _scatter(axes[3], sam_vis, "#EF6C00", "SAM3D cropped")
    axes[3].set_title(f"{title_prefix}4) (T·MoGe) + SAM3D")

    _scatter(axes[4], sam_to_moge, "#FB8C00", "SAM3D transformed")
    _scatter(axes[4], moge_vis, "#2E7D32", "MoGe")
    axes[4].set_title(f"{title_prefix}5) (T·SAM3D) + MoGe")

    _scatter(axes[5], moge_inv, "#1B5E20", "MoGe inv")
    _scatter(axes[5], sam_vis, "#EF6C00", "SAM3D cropped")
    axes[5].set_title(f"{title_prefix}6) (T^-1·MoGe) + SAM3D")

    _scatter(axes[6], sam_inv, "#FB8C00", "SAM3D inv")
    _scatter(axes[6], moge_vis, "#2E7D32", "MoGe")
    axes[6].set_title(f"{title_prefix}7) (T^-1·SAM3D) + MoGe")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return True
