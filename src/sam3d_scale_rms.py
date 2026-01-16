"""RMS 비율 기반 스케일 추정(간단 기준선)."""

from __future__ import annotations

import numpy as np

from sam3d_scale_utils import (
    apply_similarity,
    estimate_rms_scale,
    nearest_neighbors,
    sample_points,
)


def estimate_scale(
    src: np.ndarray,
    dst: np.ndarray,
    nn_max_points: int = 8000,
    seed: int = 0,
) -> dict:
    """RMS 크기 비율로 스케일을 계산."""
    src_sample = sample_points(src, nn_max_points, seed)
    dst_sample = sample_points(dst, nn_max_points, seed + 1)

    scale, r, t = estimate_rms_scale(src_sample, dst_sample)
    transformed = apply_similarity(src_sample, scale, r, t)
    nn_idx, nn_dist = nearest_neighbors(transformed, dst_sample)
    matched_dst = dst_sample[nn_idx]
    rmse = float(np.sqrt(np.mean(nn_dist**2))) if nn_dist.size else 0.0

    return {
        "scale": float(scale),
        "r": r,
        "t": t,
        "matched_src": src_sample,
        "matched_dst": matched_dst,
        "metrics": {"rmse": rmse},
    }
