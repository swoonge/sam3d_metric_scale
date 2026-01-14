import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def resolve_moge_root(repo_root: Path) -> Path:
    env_root = os.environ.get("MOGE_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(repo_root / "MoGe")
    candidates.append(repo_root.parent / "MoGe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else repo_root / "MoGe"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "outputs" / "moge_scale"

    parser = argparse.ArgumentParser(description="Masked MoGe depth/points + scale estimate")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="Ruicheng/moge-2-vitl-normal",
        help="HF model id or local path.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--no-save-npz", action="store_true")
    parser.add_argument("--scale-method", choices=["bbox_diag", "bbox_max"], default="bbox_diag")
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--show-viz", action="store_true")
    parser.add_argument("--viz-path", type=Path, default=None)
    parser.add_argument("--viz-dpi", type=int, default=150)
    parser.add_argument("--viz-max-points", type=int, default=20000)
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[:, :, -1]
    return mask > 0


def compute_scale(points: np.ndarray, method: str) -> dict:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    size = maxs - mins
    diag = float(np.linalg.norm(size))
    max_dim = float(size.max())
    if method == "bbox_max":
        value = max_dim
    else:
        value = diag
    return {
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
        "bbox_size": size.tolist(),
        "bbox_diag": diag,
        "bbox_max_dim": max_dim,
        "scale_method": method,
        "scale_value": value,
    }


def build_visualization(
    image_rgb: np.ndarray,
    user_mask: np.ndarray,
    valid_mask: np.ndarray,
    depth: np.ndarray,
    combined_mask: np.ndarray,
    points_masked: np.ndarray,
    stats: dict,
    max_points: int,
):
    import matplotlib.pyplot as plt

    overlay = image_rgb.copy()
    if user_mask.any():
        color = np.array([0, 255, 0], dtype=np.uint8)
        overlay[user_mask] = (
            0.6 * overlay[user_mask] + 0.4 * color
        ).astype(np.uint8)

    depth_vis = depth.copy()
    depth_vis[~valid_mask] = np.nan
    masked_depth_vis = np.full_like(depth, np.nan)
    masked_depth_vis[combined_mask] = depth[combined_mask]

    valid_vals = depth[combined_mask]
    if valid_vals.size > 0:
        vmin, vmax = np.percentile(valid_vals, [2, 98])
    else:
        vmin, vmax = float(np.nanmin(depth_vis)), float(np.nanmax(depth_vis))

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_depth = fig.add_subplot(gs[1, 0])
    ax_depth_masked = fig.add_subplot(gs[1, 1])
    ax_pc = fig.add_subplot(gs[:, 2], projection="3d")

    ax_img.imshow(image_rgb)
    ax_img.set_title("image")
    ax_img.axis("off")

    ax_overlay.imshow(overlay)
    ax_overlay.set_title("mask overlay")
    ax_overlay.axis("off")

    ax_depth.imshow(depth_vis, cmap="magma", vmin=vmin, vmax=vmax)
    ax_depth.set_title("depth (valid)")
    ax_depth.axis("off")

    ax_depth_masked.imshow(masked_depth_vis, cmap="magma", vmin=vmin, vmax=vmax)
    ax_depth_masked.set_title("depth (masked)")
    ax_depth_masked.axis("off")

    if points_masked.size > 0:
        points_vis = points_masked
        if max_points > 0 and points_masked.shape[0] > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(points_masked.shape[0], size=max_points, replace=False)
            points_vis = points_masked[idx]

        colors = points_vis[:, 2]
        ax_pc.scatter(
            points_vis[:, 0],
            points_vis[:, 1],
            points_vis[:, 2],
            c=colors,
            s=1,
            cmap="viridis",
        )
        mins = points_vis.min(axis=0)
        maxs = points_vis.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(1e-6, float((maxs - mins).max() / 2.0))
        ax_pc.set_xlim(center[0] - radius, center[0] + radius)
        ax_pc.set_ylim(center[1] - radius, center[1] + radius)
        ax_pc.set_zlim(center[2] - radius, center[2] + radius)
    ax_pc.set_title("point cloud (masked)")
    ax_pc.set_xlabel("x")
    ax_pc.set_ylabel("y")
    ax_pc.set_zlabel("z")

    fig.suptitle(
        f"scale={stats['scale_value']:.4f} ({stats['scale_method']})  "
        f"points={stats['points_count']}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> int:
    args = parse_args()
    image_path = resolve_path(args.image, Path.cwd())
    mask_path = resolve_path(args.mask, Path.cwd())
    output_dir = resolve_path(args.output_dir, Path.cwd())

    repo_root = Path(__file__).resolve().parents[1]
    moge_root = resolve_moge_root(repo_root)
    if str(moge_root) not in sys.path:
        sys.path.insert(0, str(moge_root))
    try:
        from moge.model.v2 import MoGeModel
    except ModuleNotFoundError as exc:
        print(
            "MoGe module not found. Install it with `pip install -e ./MoGe` "
            "or set MOGE_ROOT to the MoGe repository path."
        )
        raise exc

    if not image_path.exists():
        print(f"Missing image: {image_path}")
        return 1
    if not mask_path.exists():
        print(f"Missing mask: {mask_path}")
        return 1

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Failed to read image: {image_path}")
        return 1
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    model = MoGeModel.from_pretrained(args.model).to(device)
    model.eval()

    input_tensor = torch.tensor(
        img_rgb / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1)

    with torch.inference_mode():
        output = model.infer(input_tensor)

    points = output["points"].detach().cpu().numpy()
    depth = output["depth"].detach().cpu().numpy()
    valid = output["mask"].detach().cpu().numpy() > 0

    user_mask = load_mask(mask_path)
    if user_mask.shape != depth.shape:
        user_mask = cv2.resize(
            user_mask.astype(np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    combined = user_mask & valid
    finite = np.isfinite(points).all(axis=2) & np.isfinite(depth)
    combined &= finite

    if combined.sum() < args.min_pixels:
        print(f"Not enough valid pixels: {combined.sum()}")
        return 1

    masked_points = points[combined]
    masked_depth = depth[combined]

    stats = {
        "image": str(image_path),
        "mask": str(mask_path),
        "model": args.model,
        "points_count": int(masked_points.shape[0]),
        "depth_mean": float(masked_depth.mean()),
        "depth_median": float(np.median(masked_depth)),
        "depth_min": float(masked_depth.min()),
        "depth_max": float(masked_depth.max()),
    }
    stats.update(compute_scale(masked_points, args.scale_method))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_path.stem}_{mask_path.stem}"
    if args.output_json is not None:
        json_path = resolve_path(args.output_json, Path.cwd())
    else:
        json_path = output_dir / f"{stem}.json"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if not args.no_save_npz:
        npz_path = output_dir / f"{stem}.npz"
        np.savez_compressed(
            npz_path,
            mask=user_mask.astype(np.uint8),
            valid_mask=combined.astype(np.uint8),
            points_masked=masked_points.astype(np.float32),
            depth_masked=masked_depth.astype(np.float32),
        )

    print(f"Saved scale stats: {json_path}")
    if not args.no_save_npz:
        print(f"Saved masked outputs: {npz_path}")

    if args.save_viz or args.show_viz:
        import matplotlib.pyplot as plt

        fig = build_visualization(
            img_rgb,
            user_mask,
            valid,
            depth,
            combined,
            masked_points,
            stats,
            args.viz_max_points,
        )
        if args.save_viz:
            if args.viz_path is not None:
                viz_path = resolve_path(args.viz_path, Path.cwd())
            else:
                viz_path = output_dir / f"{stem}_viz.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(viz_path, dpi=args.viz_dpi, bbox_inches="tight")
            print(f"Saved visualization: {viz_path}")
        if args.show_viz:
            plt.show()
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
