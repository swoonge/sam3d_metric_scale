"""Preflight checks for run_full_pipeline.sh."""

from __future__ import annotations

import argparse
import os
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


def _run_python(env_name: str, code: str) -> tuple[bool, str, str]:
    cmd = ["conda", "run", "-n", env_name, "python", "-c", code]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    cmd_text = " ".join(cmd)
    if proc.returncode == 0:
        return True, "", cmd_text
    stderr = (proc.stderr or "").strip()
    if len(stderr) > 400:
        stderr = stderr[-400:]
    return False, stderr, cmd_text


def _import_code(module_name: str) -> str:
    if module_name == "sam3d_objects":
        return "import os; os.environ['LIDRA_SKIP_INIT']='true'; import sam3d_objects"
    return f"import {module_name}"


def check_import(env_name: str, module_name: str) -> tuple[bool, str]:
    ok, stderr, cmd = _run_python(env_name, _import_code(module_name))
    if ok:
        return True, ""
    return (
        False,
        f"Conda env '{env_name}' missing import '{module_name}'.\n"
        f"Command: {cmd}\n"
        f"Error: {stderr}",
    )


def _module_candidates(repo_root: Path, module_name: str) -> list[Path]:
    if module_name == "sam2":
        env_root = os.environ.get("SAM2_ROOT")
        cands = [Path(env_root)] if env_root else []
        cands += [repo_root / "sam2", repo_root.parent / "sam2"]
        return cands
    if module_name == "sam3d_objects":
        env_root = os.environ.get("SAM3D_ROOT")
        cands = [Path(env_root)] if env_root else []
        cands += [repo_root / "sam-3d-objects", repo_root.parent / "sam-3d-objects"]
        return cands
    if module_name == "moge":
        env_root = os.environ.get("MOGE_ROOT")
        cands = [Path(env_root)] if env_root else []
        cands += [repo_root / "MoGe", repo_root.parent / "MoGe"]
        return cands
    return []


def _module_marker(module_name: str, root: Path) -> Path:
    if module_name == "sam2":
        return root / "sam2"
    if module_name == "sam3d_objects":
        return root / "sam3d_objects"
    if module_name == "moge":
        return root / "moge"
    return root


def _resolve_local_repo(repo_root: Path, module_name: str) -> Path | None:
    for candidate in _module_candidates(repo_root, module_name):
        marker = _module_marker(module_name, candidate)
        if marker.exists():
            return candidate
    return None


def check_local_repo_import(env_name: str, module_name: str, repo_root: Path) -> tuple[bool, str]:
    local_root = _resolve_local_repo(repo_root, module_name)
    if local_root is None:
        candidates = ", ".join(str(p) for p in _module_candidates(repo_root, module_name))
        return (
            False,
            f"Local repo for '{module_name}' is required but not found.\n"
            f"Searched: {candidates}",
        )

    code = (
        "import sys; "
        f"sys.path.insert(0, {repr(str(local_root))}); "
        f"{_import_code(module_name)}"
    )
    ok, stderr, cmd = _run_python(env_name, code)
    if ok:
        return True, ""

    return (
        False,
        f"Conda env '{env_name}' cannot import '{module_name}' from required local repo.\n"
        f"Local repo: {local_root}\n"
        f"Command: {cmd}\n"
        f"Error: {stderr}"
    )


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    repo_root = Path(__file__).resolve().parents[1]

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

    local_dep_checks: list[tuple[str, str]] = [
        (args.sam2_env, "sam2"),
        (args.sam3d_env, "sam3d_objects"),
    ]
    if args.run_moge:
        local_dep_checks.append((args.moge_env, "moge"))

    for env_name, module_name in local_dep_checks:
        ok, msg = check_local_repo_import(env_name, module_name, repo_root)
        if not ok:
            errors.append(msg)

    scale_checks: list[tuple[str, str]] = [
        (args.scale_env, "trimesh"),
        (args.scale_env, "plyfile"),
    ]
    for env_name, module_name in scale_checks:
        ok, msg = check_import(env_name, module_name)
        if not ok:
            errors.append(msg)

    if errors:
        print("Preflight failed with the following issues:")
        for err in errors:
            print(f"- {err}")
        print(
            "\nFix the issues above and retry. "
            "Core deps (sam2/sam3d-objects/moge) must be available from local repos."
        )
        return 1

    print("Preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
