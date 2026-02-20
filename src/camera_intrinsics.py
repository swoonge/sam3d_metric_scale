"""Camera intrinsics parsing helpers.

Supported formats:
- 3x3 matrix (9 values)
- flat tuple/list: fx fy cx cy (4 values)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _parse_intrinsics_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 9:
        k = values.reshape(3, 3).astype(np.float32)
    elif values.size == 4:
        fx, fy, cx, cy = [float(v) for v in values.tolist()]
        k = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            "Invalid camera intrinsics: expected 9 values (3x3) "
            "or 4 values (fx fy cx cy)."
        )

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    if not np.isfinite(k).all():
        raise ValueError("Camera intrinsics contain non-finite values.")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid focal lengths in camera intrinsics: fx={fx}, fy={fy}")
    return k


def load_intrinsics_matrix(path: Path) -> np.ndarray:
    raw = np.loadtxt(str(path), dtype=np.float32)
    return _parse_intrinsics_values(raw)


def load_intrinsics_tuple(path: Path) -> tuple[float, float, float, float]:
    k = load_intrinsics_matrix(path)
    return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
