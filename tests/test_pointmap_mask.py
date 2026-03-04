from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from sam3d_export import apply_pointmap_mask, sanitize_depth_for_pointmap


def test_pointmap_mask_zeros_background():
    pointmap = np.ones((6, 6, 3), dtype=np.float32)
    mask = np.zeros((6, 6), dtype=bool)
    mask[:3, :3] = True

    masked = apply_pointmap_mask(pointmap, mask)
    assert np.count_nonzero(masked) < np.count_nonzero(pointmap)
    assert np.all(masked[~mask] == 0.0)


def test_pointmap_mask_can_fill_background_with_nan():
    pointmap = np.ones((4, 4, 3), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True

    masked = apply_pointmap_mask(pointmap, mask, fill_value=np.nan)
    assert np.isnan(masked[~mask]).all()
    assert np.isfinite(masked[mask]).all()


def test_sanitize_depth_for_pointmap_marks_non_positive_as_nan():
    depth = np.array(
        [
            [0.1, 0.0, -1.0],
            [np.inf, np.nan, 2.0],
        ],
        dtype=np.float32,
    )
    cleaned = sanitize_depth_for_pointmap(depth)

    assert np.isfinite(cleaned[0, 0])
    assert np.isfinite(cleaned[1, 2])
    assert np.isnan(cleaned[0, 1])
    assert np.isnan(cleaned[0, 2])
    assert np.isnan(cleaned[1, 0])
    assert np.isnan(cleaned[1, 1])
