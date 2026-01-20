#!/usr/bin/env python3
"""SAM3D inference 분석 및 시각화용 산출물 생성.

- 입력: outputs/<sample>/ 이미지 + 마스크
- 출력: 분석 JSON/MD, pointmap/coords/mesh/gs 시각화 보조 파일
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


def resolve_sam3d_root(repo_root: Path) -> Path:
    """SAM3D Objects 레포 경로를 환경변수/기본 후보에서 탐색."""
    env_root = os.environ.get("SAM3D_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(repo_root / "sam-3d-objects")
    candidates.append(repo_root.parent / "sam-3d-objects")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else repo_root / "sam-3d-objects"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    sam3d_root = resolve_sam3d_root(repo_root)
    default_config = sam3d_root / "checkpoints" / "hf" / "pipeline.yaml"

    parser = argparse.ArgumentParser(description="Analyze SAM3D inference outputs")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--sam3d-config", type=Path, default=default_config)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--pointmap-max-points", type=int, default=200000)
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


def find_first_image(output_root: Path) -> Path | None:
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        candidates = sorted(output_root.glob(ext))
        if candidates:
            return candidates[0]
    return None


def find_first_mask(output_root: Path) -> Path | None:
    mask_dir = output_root / "sam2_masks"
    if not mask_dir.exists():
        return None
    candidates = sorted(mask_dir.glob("*.png"))
    return candidates[0] if candidates else None


def summarize_tensor(tensor) -> Dict[str, Any]:
    import torch

    t = tensor.detach().cpu()
    summary = {
        "type": "torch.Tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(tensor.device),
    }
    if t.numel() > 0 and t.dtype.is_floating_point:
        summary["min"] = float(t.min())
        summary["max"] = float(t.max())
        summary["mean"] = float(t.mean())
    return summary


def summarize_array(array: np.ndarray) -> Dict[str, Any]:
    summary = {
        "type": "np.ndarray",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size > 0 and np.issubdtype(array.dtype, np.floating):
        summary["min"] = float(np.nanmin(array))
        summary["max"] = float(np.nanmax(array))
        summary["mean"] = float(np.nanmean(array))
    return summary


def summarize_trimesh(mesh) -> Dict[str, Any]:
    summary = {
        "type": type(mesh).__name__,
    }
    try:
        summary["vertices"] = int(mesh.vertices.shape[0])
        summary["faces"] = int(mesh.faces.shape[0])
    except Exception:
        pass
    return summary


def summarize_gaussian(gs) -> Dict[str, Any]:
    summary = {"type": type(gs).__name__}
    for attr in (
        "xyz",
        "positions",
        "means",
        "scales",
        "scale",
        "opacities",
        "opacity",
        "rotations",
        "rotation",
        "features_dc",
        "features_rest",
        "features",
    ):
        if hasattr(gs, attr):
            value = getattr(gs, attr)
            try:
                summary[attr] = summarize_tensor(value)
            except Exception:
                summary[attr] = str(type(value))
    return summary


def summarize_value(value) -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        torch = None

    if isinstance(value, (int, float, bool, str)):
        return {"type": type(value).__name__, "value": value}
    if torch is not None and isinstance(value, torch.Tensor):
        return summarize_tensor(value)
    if isinstance(value, np.ndarray):
        return summarize_array(value)
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys()),
        }
    if isinstance(value, (list, tuple)):
        summary = {
            "type": type(value).__name__,
            "length": len(value),
        }
        if len(value) > 0:
            summary["element_type"] = type(value[0]).__name__
        return summary
    if value is None:
        return {"type": "None"}
    if type(value).__name__ == "Trimesh":
        return summarize_trimesh(value)
    return {"type": type(value).__name__}


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    values = depth[valid]
    vmin, vmax = np.percentile(values, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (depth - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap("magma")
        rgba = cmap(norm)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        rgb[~valid] = 0
        return rgb
    except Exception:
        gray = (norm * 255).astype(np.uint8)
        gray[~valid] = 0
        return np.stack([gray] * 3, axis=2)


def write_ply_points(path: Path, points: np.ndarray, colors: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for idx, pt in enumerate(points):
            line = f"{pt[0]} {pt[1]} {pt[2]}"
            if colors is not None:
                c = colors[idx]
                line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
            f.write(line + "\n")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = resolve_path(args.output_root, Path.cwd())
    sam3d_root = resolve_sam3d_root(repo_root)

    image_path = resolve_path(args.image, Path.cwd()) if args.image else find_first_image(output_root)
    mask_path = resolve_path(args.mask, Path.cwd()) if args.mask else find_first_mask(output_root)

    if image_path is None or not image_path.exists():
        print("Missing image for analysis.")
        return 1
    if mask_path is None or not mask_path.exists():
        print("Missing mask for analysis.")
        return 1

    analysis_dir = args.analysis_dir
    if analysis_dir is None:
        analysis_dir = output_root / "sam3d_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_path(args.sam3d_config, sam3d_root)
    if str(sam3d_root) not in sys.path:
        sys.path.insert(0, str(sam3d_root))
    os.chdir(sam3d_root)
    sys.path.append(str(sam3d_root / "notebook"))
    from inference import Inference, load_image, load_mask

    inference = Inference(str(config_path), compile=args.compile)
    image = load_image(str(image_path))
    mask = load_mask(str(mask_path))

    output = inference(image, mask, seed=args.seed)

    stem = mask_path.stem
    summary = {}
    for key, value in output.items():
        if key in ("glb", "gs"):
            continue
        summary[key] = summarize_value(value)
    if "glb" in output:
        summary["glb"] = summarize_trimesh(output["glb"])
    if "gs" in output:
        summary["gs"] = summarize_gaussian(output["gs"])

    summary_path = analysis_dir / f"{stem}_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Pose info
    pose_path = analysis_dir / f"{stem}_pose.txt"
    pose_lines = []
    for key in ("rotation", "translation", "scale"):
        if key in output:
            try:
                val = output[key].detach().cpu().numpy()
            except Exception:
                val = output[key]
            pose_lines.append(f"{key}: {val}")
    pose_path.write_text("\n".join(pose_lines), encoding="utf-8")

    # Pointmap depth visualization + point cloud
    pointmap = output.get("pointmap")
    pointmap_colors = output.get("pointmap_colors")
    if pointmap is not None:
        try:
            import torch

            if isinstance(pointmap, torch.Tensor):
                pointmap_np = pointmap.detach().cpu().numpy()
            else:
                pointmap_np = np.asarray(pointmap)
        except Exception:
            pointmap_np = np.asarray(pointmap)

        depth = pointmap_np[..., 2]
        depth_rgb = depth_to_rgb(depth)
        Image.fromarray(depth_rgb).save(analysis_dir / f"{stem}_pointmap_depth.png")

        points = pointmap_np.reshape(-1, 3)
        mask_valid = np.isfinite(points).all(axis=1)
        points = points[mask_valid]

        colors = None
        if pointmap_colors is not None:
            try:
                import torch

                if isinstance(pointmap_colors, torch.Tensor):
                    colors_np = pointmap_colors.detach().cpu().numpy()
                else:
                    colors_np = np.asarray(pointmap_colors)
            except Exception:
                colors_np = np.asarray(pointmap_colors)
            colors = colors_np.reshape(-1, 3)[mask_valid]
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = np.clip(colors, 0, 255).astype(np.uint8)

        max_points = max(1, int(args.pointmap_max_points))
        if points.shape[0] > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(points.shape[0], size=max_points, replace=False)
            points = points[idx]
            if colors is not None:
                colors = colors[idx]

        write_ply_points(analysis_dir / f"{stem}_pointmap_cloud.ply", points, colors)

    coords = output.get("coords")
    if coords is not None:
        try:
            import torch

            if isinstance(coords, torch.Tensor):
                coords_np = coords.detach().cpu().numpy()
            else:
                coords_np = np.asarray(coords)
        except Exception:
            coords_np = np.asarray(coords)
        # coords: [batch, x, y, z] -> normalize to [-0.5, 0.5]
        coords_xyz = coords_np[:, 1:4].astype(np.float32)
        coords_xyz = coords_xyz / 64.0 - 0.5
        write_ply_points(analysis_dir / f"{stem}_coords_cloud.ply", coords_xyz, None)

    # Mesh export (if needed)
    glb = output.get("glb")
    if glb is not None:
        glb.export(analysis_dir / f"{stem}_analysis_mesh.glb")
        glb.export(analysis_dir / f"{stem}_analysis_mesh.ply")
        glb.export(analysis_dir / f"{stem}_analysis_mesh.obj")

    # Gaussian splat export for analysis
    gs = output.get("gs")
    if gs is not None:
        gs.save_ply(str(analysis_dir / f"{stem}_analysis_gs.ply"))

    # Analysis markdown summary
    md_lines = [
        f"# SAM3D 분석 요약 ({stem})",
        "",
        f"- image: {image_path}",
        f"- mask: {mask_path}",
        f"- analysis_dir: {analysis_dir}",
        "",
        "## 출력 키 요약",
    ]
    for key, value in summary.items():
        md_lines.append(f"- {key}: {value}")
    (analysis_dir / f"{stem}_analysis.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Analysis saved to: {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
