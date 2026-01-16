"""Umeyama + RANSAC 기반 스케일 추정."""

from __future__ import annotations

import numpy as np

from sam3d_scale_utils import (
    apply_similarity,
    estimate_rms_scale,
    nearest_neighbors,
    sample_points,
    umeyama_alignment,
)


def estimate_scale(
    src: np.ndarray,
    dst: np.ndarray,
    iters: int = 800,
    sample_size: int = 128,
    inlier_thresh: float = 0.0,
    nn_max_points: int = 8000,
    seed: int = 0,
) -> dict:
    """최근접 매칭을 만든 뒤 RANSAC으로 Umeyama 스케일을 강인하게 추정."""
    src_sample = sample_points(src, nn_max_points, seed)
    dst_sample = sample_points(dst, nn_max_points, seed + 1)

    # 초기 최근접 매칭 구성
    scale0, r0, t0 = estimate_rms_scale(src_sample, dst_sample)
    transformed0 = apply_similarity(src_sample, scale0, r0, t0)
    nn_idx, _ = nearest_neighbors(transformed0, dst_sample)
    matched_dst = dst_sample[nn_idx]

    if inlier_thresh <= 0.0:
        mins = matched_dst.min(axis=0)
        maxs = matched_dst.max(axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        inlier_thresh = max(0.005, 0.03 * diag)

    rng = np.random.default_rng(seed)
    best_inliers = -1
    best_mask = None
    best_scale = 1.0
    best_r = np.eye(3, dtype=np.float32)
    best_t = np.zeros(3, dtype=np.float32)

    sample_size = max(3, min(sample_size, src_sample.shape[0]))
    for _ in range(iters):
        idx = rng.choice(src_sample.shape[0], size=sample_size, replace=False)
        try:
            scale, r, t = umeyama_alignment(src_sample[idx], matched_dst[idx])
        except ValueError:
            continue
        transformed = apply_similarity(src_sample, scale, r, t)
        dist = np.linalg.norm(transformed - matched_dst, axis=1)
        inliers = dist <= inlier_thresh
        inlier_count = int(np.sum(inliers))
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_scale, best_r, best_t = scale, r, t
            best_mask = inliers

    if best_mask is None or best_inliers <= 0:
        best_scale, best_r, best_t = umeyama_alignment(src_sample, matched_dst)
        best_mask = np.ones(src_sample.shape[0], dtype=bool)
        best_inliers = int(best_mask.sum())
    else:
        best_scale, best_r, best_t = umeyama_alignment(
            src_sample[best_mask], matched_dst[best_mask]
        )

    inlier_ratio = float(best_inliers / max(1, src_sample.shape[0]))
    return {
        "scale": float(best_scale),
        "r": best_r,
        "t": best_t,
        "matched_src": src_sample,
        "matched_dst": matched_dst,
        "match_mask": best_mask,
        "metrics": {"inliers": best_inliers, "inlier_ratio": inlier_ratio},
    }
