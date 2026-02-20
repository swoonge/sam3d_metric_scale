from __future__ import annotations

import numpy as np
import pytest

from camera_intrinsics import load_intrinsics_matrix, load_intrinsics_tuple


def test_intrinsics_parser_supports_matrix_and_flat(tmp_path):
    matrix_path = tmp_path / "cam_matrix.txt"
    flat_path = tmp_path / "cam_flat.txt"

    expected = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 510.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.savetxt(matrix_path, expected, fmt="%.6f")
    flat_path.write_text("500 510 320 240\n", encoding="utf-8")

    k_mat = load_intrinsics_matrix(matrix_path)
    k_flat = load_intrinsics_matrix(flat_path)
    fx, fy, cx, cy = load_intrinsics_tuple(flat_path)

    assert np.allclose(k_mat, expected)
    assert np.allclose(k_flat, expected)
    assert (fx, fy, cx, cy) == (500.0, 510.0, 320.0, 240.0)


def test_intrinsics_parser_rejects_invalid_format(tmp_path):
    invalid = tmp_path / "bad_cam.txt"
    invalid.write_text("1 2 3\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_intrinsics_matrix(invalid)
