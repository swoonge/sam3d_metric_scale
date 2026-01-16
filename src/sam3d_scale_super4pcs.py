"""Super4PCS 기반 스케일/정합 추정."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from sam3d_scale_utils import apply_similarity, nearest_neighbors

_FLOAT_RE = re.compile(
    r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)",
    re.IGNORECASE,
)


def _save_points_ply(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = points.astype(np.float32)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        if points.shape[0] > 0:
            np.savetxt(f, points, fmt="%.6f %.6f %.6f")


def _resolve_binary(bin_path: str | None) -> str:
    if bin_path:
        candidate = Path(bin_path)
        if candidate.is_dir():
            candidate = candidate / "Super4PCS"
        if candidate.exists():
            return str(candidate)
    env_bin = os.environ.get("SUPER4PCS_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    env_root = os.environ.get("SUPER4PCS_ROOT")
    if env_root:
        for rel in (
            "Super4PCS",
            "build/Super4PCS",
            "build/demos/Super4PCS/Super4PCS",
            "bin/Super4PCS",
        ):
            candidate = Path(env_root) / rel
            if candidate.exists():
                return str(candidate)
    which = shutil.which("Super4PCS")
    if which:
        return which
    raise RuntimeError(
        "Super4PCS binary not found. Set SUPER4PCS_BIN or SUPER4PCS_ROOT, "
        "or add Super4PCS to PATH."
    )


def _parse_transform(text: str) -> np.ndarray:
    def _parse_tokens(tokens: list[str]) -> np.ndarray:
        if len(tokens) >= 16:
            values = [float(x) for x in tokens[:16]]
            mat = np.array(values, dtype=np.float32).reshape(4, 4)
        elif len(tokens) >= 12:
            values = [float(x) for x in tokens[:12]]
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :] = np.array(values, dtype=np.float32).reshape(3, 4)
        else:
            raise RuntimeError("Failed to parse Super4PCS transform output.")
        if not np.isfinite(mat).all():
            raise RuntimeError("Super4PCS produced non-finite transform values.")
        return mat

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "MATRIX" in line.upper():
            matrix_lines = lines[idx + 1 : idx + 5]
            if len(matrix_lines) >= 4:
                tokens: list[str] = []
                for row in matrix_lines[:4]:
                    tokens.extend(_FLOAT_RE.findall(row))
                try:
                    return _parse_tokens(tokens)
                except RuntimeError:
                    break

    tokens = _FLOAT_RE.findall(text)
    if "VERSION" in text.upper() and len(tokens) == 17:
        tokens = tokens[1:]
    return _parse_tokens(tokens)


def _estimate_scale_from_rt(
    src: np.ndarray, dst: np.ndarray, r: np.ndarray, t: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, float]:
    src_rt = src @ r.T
    transformed = apply_similarity(src, 1.0, r, t)
    nn_idx, nn_dist = nearest_neighbors(transformed, dst)
    dst_match = dst[nn_idx]
    numer = float(np.sum((dst_match - t) * src_rt))
    denom = float(np.sum(src_rt**2))
    scale = numer / max(denom, 1e-8)

    transformed_scaled = apply_similarity(src, scale, r, t)
    nn_idx, nn_dist = nearest_neighbors(transformed_scaled, dst)
    dst_match = dst[nn_idx]
    rmse = float(np.sqrt(np.mean(nn_dist**2))) if nn_dist.size else 0.0
    return scale, dst_match, nn_dist, rmse


def estimate_scale(
    src: np.ndarray,
    dst: np.ndarray,
    bin_path: str | None = None,
    overlap: float = 0.7,
    delta: float = 0.0,
    timeout: int = 1000,
    seed: int = 0,
) -> dict:
    """Super4PCS로 정합 후 스케일을 추정."""
    _ = seed  # reserved for future deterministic sampling
    binary = _resolve_binary(bin_path)

    if delta <= 0.0:
        mins = dst.min(axis=0)
        maxs = dst.max(axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        delta = max(1e-6, 0.01 * diag)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src_ply = tmpdir / "src.ply"
        dst_ply = tmpdir / "dst.ply"
        out_txt = tmpdir / "transform.txt"

        _save_points_ply(src, src_ply)
        _save_points_ply(dst, dst_ply)

        cmd = [
            binary,
            "-i",
            str(src_ply),
            str(dst_ply),
            "-o",
            str(overlap),
            "-d",
            str(delta),
            "-t",
            str(timeout),
            "-m",
            str(out_txt),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Super4PCS failed:\n"
                f"command: {' '.join(cmd)}\n"
                f"output:\n{result.stdout}"
            )

        file_text = out_txt.read_text(encoding="utf-8") if out_txt.exists() else ""
        stdout_text = result.stdout or ""

    errors = []
    for label, text in (("matrix_file", file_text), ("stdout", stdout_text)):
        if not text.strip():
            continue
        try:
            mat = _parse_transform(text)
            break
        except RuntimeError as exc:
            errors.append(f"{label}: {exc}")
            mat = None
    if mat is None:
        snippet = file_text.strip() or stdout_text.strip()
        snippet = snippet[:400] + ("..." if len(snippet) > 400 else "")
        raise RuntimeError(
            "Failed to parse Super4PCS transform output.\n"
            + "\n".join(errors)
            + ("\nOutput snippet:\n" + snippet if snippet else "")
        )
    r = mat[:3, :3].astype(np.float32)
    t = mat[:3, 3].astype(np.float32)

    scale, matched_dst, _, rmse = _estimate_scale_from_rt(src, dst, r, t)

    return {
        "scale": float(scale),
        "r": r,
        "t": t,
        "matched_src": src,
        "matched_dst": matched_dst,
        "metrics": {
            "rmse": rmse,
            "delta": float(delta),
            "overlap": float(overlap),
            "timeout": int(timeout),
        },
    }
