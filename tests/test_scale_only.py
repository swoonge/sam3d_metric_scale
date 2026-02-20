from __future__ import annotations

import numpy as np

from sam3d_scale_utils import clamp_scale_value, estimate_rms_scale


def test_scale_only_rms_accepts_unequal_point_counts():
    rng = np.random.default_rng(123)
    src = rng.normal(size=(137, 3)).astype(np.float32)
    dst = (rng.normal(size=(89, 3)) * 2.0).astype(np.float32)

    scale, _, _ = estimate_rms_scale(src, dst)
    clamped = clamp_scale_value(scale)

    assert np.isfinite(scale)
    assert scale > 0.0
    assert np.isfinite(clamped)
    assert clamped > 0.0


def test_scale_clamp_bounds():
    assert clamp_scale_value(0.0, min_value=1e-3, max_value=10.0) == 1e-3
    assert clamp_scale_value(1234.0, min_value=1e-3, max_value=10.0) == 10.0
