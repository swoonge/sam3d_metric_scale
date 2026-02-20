"""Preflight checks for run_full_pipeline.sh."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for SAM3D metric scale pipeline.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--depth-image", type=Path, default=None)
    parser.add_argument("--cam-k", type=Path, default=None)
    parser.add_argument("--sam2-env", type=str, required=True)
    parser.add_argument("--sam3d-env", type=str, required=True)
    parser.add_argument("--scale-env", type=str, required=True)
    parser.add_argument("--run-moge", action="store_true")
    parser.add_argument("--moge-env", type=str, default="moge")
    return parser.parse_args()


def check_import(env_name: str, module_name: str) -> tuple[bool, str]:
    cmd = ["conda", "run", "-n", env_name, "python", "-c", f"import {module_name}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, ""

    stderr = (proc.stderr or "").strip()
    if len(stderr) > 400:
        stderr = stderr[-400:]
    message = (
        f"Conda env '{env_name}' missing import '{module_name}'.\n"
        f"Command: {' '.join(cmd)}\n"
        f"Error: {stderr}"
    )
    return False, message


def main() -> int:
    args = parse_args()
    errors: list[str] = []

    if shutil.which("conda") is None:
        errors.append("conda not found in PATH.")

    image_path = args.image.resolve()
    if not image_path.exists():
        errors.append(f"Missing image: {image_path}")

    depth_path = args.depth_image.resolve() if args.depth_image is not None else None
    cam_k_path = args.cam_k.resolve() if args.cam_k is not None else None

    if depth_path is not None:
        if not depth_path.exists():
            errors.append(f"Missing depth image: {depth_path}")
        if cam_k_path is None:
            errors.append("--depth-image requires --cam-k (3x3 or fx fy cx cy).")
    if cam_k_path is not None and not cam_k_path.exists():
        errors.append(f"Missing camera intrinsics: {cam_k_path}")

    checks: list[tuple[str, str]] = [
        (args.sam2_env, "sam2"),
        (args.sam3d_env, "sam3d_objects"),
        (args.scale_env, "trimesh"),
        (args.scale_env, "plyfile"),
    ]
    if args.run_moge:
        checks.append((args.moge_env, "moge"))

    for env_name, module_name in checks:
        ok, msg = check_import(env_name, module_name)
        if not ok:
            errors.append(msg)

    if errors:
        print("Preflight failed with the following issues:")
        for err in errors:
            print(f"- {err}")
        print(
            "\nInstall missing modules in the reported conda env(s) and retry."
        )
        return 1

    print("Preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
