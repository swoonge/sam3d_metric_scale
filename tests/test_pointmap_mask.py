from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from sam3d_export import apply_pointmap_mask


def test_pointmap_mask_zeros_background():
    pointmap = np.ones((6, 6, 3), dtype=np.float32)
    mask = np.zeros((6, 6), dtype=bool)
    mask[:3, :3] = True

    masked = apply_pointmap_mask(pointmap, mask)
    assert np.count_nonzero(masked) < np.count_nonzero(pointmap)
    assert np.all(masked[~mask] == 0.0)
