from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sam3d_export import PointmapInputCompat


def test_pointmap_input_compat_supports_dict_and_legacy_to():
    points = torch.randn(4, 5, 3)
    intrinsics = np.eye(3, dtype=np.float32)
    compat = PointmapInputCompat(pointmap=points, intrinsics=intrinsics)

    assert isinstance(compat, dict)
    assert compat["pointmap"].shape == (4, 5, 3)
    assert np.allclose(compat["intrinsics"], intrinsics)

    moved = compat.to("cpu")
    assert torch.allclose(moved, points)
