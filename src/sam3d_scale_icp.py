"""Umeyama + ICP 기반 스케일 추정."""

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
    max_iters: int = 30,
    tolerance: float = 1e-5,
    nn_max_points: int = 8000,
    trim_ratio: float = 0.8,
    seed: int = 0,
) -> dict:
    """ICP로 최근접 매칭을 반복하면서 Umeyama 스케일을 추정."""
    src_sample = sample_points(src, nn_max_points, seed)
    dst_sample = sample_points(dst, nn_max_points, seed + 1)

    scale, r, t = estimate_rms_scale(src_sample, dst_sample)
    prev_error = None
    iter_count = 0
    matched = None

    for it in range(max_iters):
        iter_count = it + 1
        transformed = apply_similarity(src_sample, scale, r, t)
        nn_idx, nn_dist = nearest_neighbors(transformed, dst_sample)
        matched = dst_sample[nn_idx]

        if 0.0 < trim_ratio < 1.0 and nn_dist.size > 0:
            keep_n = max(3, int(nn_dist.shape[0] * trim_ratio))
            keep_idx = np.argpartition(nn_dist, keep_n - 1)[:keep_n]
            src_used = src_sample[keep_idx]
            matched_used = matched[keep_idx]
            dist_used = nn_dist[keep_idx]
        else:
            src_used = src_sample
            matched_used = matched
            dist_used = nn_dist

        scale, r, t = umeyama_alignment(src_used, matched_used)
        rmse = float(np.sqrt(np.mean(dist_used**2))) if dist_used.size else 0.0
        if prev_error is not None and abs(prev_error - rmse) < tolerance:
            break
        prev_error = rmse

    # 최종 매칭 갱신(시각화용)
    transformed = apply_similarity(src_sample, scale, r, t)
    nn_idx, nn_dist = nearest_neighbors(transformed, dst_sample)
    matched = dst_sample[nn_idx]
    if 0.0 < trim_ratio < 1.0 and nn_dist.size > 0:
        keep_n = max(3, int(nn_dist.shape[0] * trim_ratio))
        keep_idx = np.argpartition(nn_dist, keep_n - 1)[:keep_n]
        matched_src = src_sample[keep_idx]
        matched_dst = matched[keep_idx]
        dist_used = nn_dist[keep_idx]
    else:
        matched_src = src_sample
        matched_dst = matched
        dist_used = nn_dist
    rmse = float(np.sqrt(np.mean(dist_used**2))) if dist_used.size else 0.0

    return {
        "scale": float(scale),
        "r": r,
        "t": t,
        "matched_src": matched_src,
        "matched_dst": matched_dst,
        "metrics": {"rmse": rmse, "iters": iter_count},
    }
