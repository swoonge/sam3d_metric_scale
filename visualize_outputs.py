import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAM2/MoGe/SAM3D outputs")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser window.",
    )
    parser.add_argument(
        "--moge-max-points",
        type=int,
        default=120000,
        help="Maximum points for MoGe point cloud (<=0 for no limit).",
    )
    parser.add_argument(
        "--moge-axis-fraction",
        type=float,
        default=0.2,
        help="Axis length as fraction of bbox max dimension.",
    )
    parser.add_argument(
        "--moge-axis-steps",
        type=int,
        default=80,
        help="Number of points per axis for visualization.",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


def find_image_path(output_root: Path, override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        return override

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        candidates = sorted(output_root.glob(ext))
        if candidates:
            return candidates[0]

    moge_dir = output_root / "moge_scale"
    for json_path in sorted(moge_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            continue
        image_path = payload.get("image")
        if image_path:
            candidate = Path(image_path)
            if candidate.exists():
                return candidate

    return None


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def load_mask(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    return np.array(mask) > 0


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask * 255] * 3, axis=2).astype(np.uint8)


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


def load_moge_depth(npz_path: Path) -> Optional[np.ndarray]:
    if npz_path is None or not npz_path.exists():
        return None
    data = np.load(npz_path)
    valid_mask = data.get("valid_mask")
    depth_masked = data.get("depth_masked")
    if valid_mask is None or depth_masked is None:
        return None
    valid_mask = valid_mask.astype(bool)
    depth_masked = depth_masked.astype(np.float32)
    depth_full = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    if depth_masked.shape[0] == int(valid_mask.sum()):
        depth_full[valid_mask] = depth_masked
    else:
        return None
    return depth_full


def points_from_masked_depth(
    valid_mask: np.ndarray, depth_masked: np.ndarray
) -> Optional[np.ndarray]:
    if depth_masked.shape[0] != int(valid_mask.sum()):
        return None
    ys, xs = np.where(valid_mask)
    height, width = valid_mask.shape
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    focal = max(height, width)
    z = depth_masked.astype(np.float32)
    x = (xs - cx) * z / focal
    y = -(ys - cy) * z / focal
    return np.stack([x, y, z], axis=1)


def load_moge_points(npz_path: Path) -> Optional[np.ndarray]:
    if npz_path is None or not npz_path.exists():
        return None
    data = np.load(npz_path)
    points = data.get("points_masked")
    if points is not None:
        points = points.astype(np.float32)
        points = points[np.isfinite(points).all(axis=1)]
        return points
    valid_mask = data.get("valid_mask")
    depth_masked = data.get("depth_masked")
    if valid_mask is None or depth_masked is None:
        return None
    valid_mask = valid_mask.astype(bool)
    depth_masked = depth_masked.astype(np.float32)
    points = points_from_masked_depth(valid_mask, depth_masked)
    if points is None:
        return None
    points = points[np.isfinite(points).all(axis=1)]
    return points


def colorize_values(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    vmin, vmax = np.percentile(values, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (values - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap("viridis")
        rgba = cmap(norm)
        return (rgba[:, :3] * 255).astype(np.uint8)
    except Exception:
        gray = (norm * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=1)


def build_pointcloud_figure(
    points: np.ndarray,
    max_points: int,
    axis_fraction: float,
    axis_steps: int,
) -> tuple[object, Optional[str]]:
    if points is None or points.size == 0:
        return None, "MoGe PC: empty"

    points = points.astype(np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        return None, "MoGe PC: invalid points"

    points_vis = points
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points_vis = points[idx]

    mins = points_vis.min(axis=0)
    maxs = points_vis.max(axis=0)
    max_dim = float((maxs - mins).max())
    axis_length = max(1e-6, max_dim * float(axis_fraction))
    center = (mins + maxs) / 2.0
    steps = max(2, int(axis_steps))

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=points_vis[:, 0],
                y=points_vis[:, 1],
                z=points_vis[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=points_vis[:, 2],
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=True,
                ),
                name="points",
            )
        )

        t = np.linspace(-axis_length, axis_length, steps, dtype=np.float32)
        fig.add_trace(
            go.Scatter3d(
                x=center[0] + t,
                y=np.full_like(t, center[1]),
                z=np.full_like(t, center[2]),
                mode="lines",
                line=dict(color="red", width=4),
                name="x",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=np.full_like(t, center[0]),
                y=center[1] + t,
                z=np.full_like(t, center[2]),
                mode="lines",
                line=dict(color="green", width=4),
                name="y",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=np.full_like(t, center[0]),
                y=np.full_like(t, center[1]),
                z=center[2] + t,
                mode="lines",
                line=dict(color="blue", width=4),
                name="z",
                showlegend=False,
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )
        return fig, None
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        colors = colorize_values(points_vis[:, 2]) / 255.0
        ax.scatter(
            points_vis[:, 0],
            points_vis[:, 1],
            points_vis[:, 2],
            c=colors,
            s=1,
        )

        t = np.linspace(-axis_length, axis_length, steps, dtype=np.float32)
        ax.plot(center[0] + t, center[1] + 0 * t, center[2] + 0 * t, color="r")
        ax.plot(center[0] + 0 * t, center[1] + t, center[2] + 0 * t, color="g")
        ax.plot(center[0] + 0 * t, center[1] + 0 * t, center[2] + t, color="b")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect([1, 1, 1])
        fig.tight_layout()
        return fig, None
    except Exception as exc:
        return None, f"MoGe PC: {exc}"


def find_moge_npz(moge_dir: Path, mask_stem: str) -> Optional[Path]:
    candidates = sorted(moge_dir.glob(f"*_{mask_stem}.npz"))
    if not candidates:
        candidates = sorted(moge_dir.glob(f"{mask_stem}.npz"))
    return candidates[0] if candidates else None


def collect_items(output_root: Path):
    mask_dir = output_root / "sam2_masks"
    moge_dir = output_root / "moge_scale"
    sam3d_dir = output_root / "sam3d"

    masks = sorted(mask_dir.glob("*.png"))
    items = []
    for mask_path in masks:
        stem = mask_path.stem
        moge_npz = find_moge_npz(moge_dir, stem)
        sam3d_ply = sam3d_dir / f"{stem}.ply"
        items.append(
            {
                "stem": stem,
                "mask": mask_path,
                "moge_npz": moge_npz,
                "sam3d_ply": sam3d_ply if sam3d_ply.exists() else None,
            }
        )
    return items


def print_summary(image_path: Path, items) -> None:
    print("Visualization summary:")
    print(f"Image: {image_path}")
    for item in items:
        stem = item["stem"]
        mask_path = item["mask"]
        moge_npz = item["moge_npz"]
        sam3d_ply = item["sam3d_ply"]
        moge_status = str(moge_npz) if moge_npz else "missing"
        sam3d_status = str(sam3d_ply) if sam3d_ply else "missing"
        print(f"- {stem}")
        print(f"  mask: {mask_path}")
        print(f"  moge: {moge_status}")
        print(f"  sam3d: {sam3d_status}")


def main() -> int:
    args = parse_args()
    output_root = resolve_path(args.output_root, Path.cwd())
    if not output_root.exists():
        print(f"Missing output root: {output_root}")
        return 1

    image_path = find_image_path(output_root, args.image)
    if image_path is None or not image_path.exists():
        print("Could not find source image. Provide --image or copy image into output root.")
        return 1

    items = collect_items(output_root)
    if not items:
        print(f"No masks found under: {output_root / 'sam2_masks'}")
        return 1

    try:
        import gradio as gr
    except ImportError:
        print("gradio is required for visualization. Install it in sam3d-objects env.")
        print_summary(image_path, items)
        return 1

    image_rgb = load_image(image_path)
    item_map = {item["stem"]: item for item in items}

    def render(stem: str):
        item = item_map[stem]
        mask = load_mask(item["mask"])
        mask_rgb = mask_to_rgb(mask)
        depth_full = load_moge_depth(item["moge_npz"]) if item["moge_npz"] else None
        depth_rgb = depth_to_rgb(depth_full) if depth_full is not None else None

        moge_fig = None
        moge_note = None
        if item["moge_npz"] is not None:
            points = load_moge_points(item["moge_npz"])
            moge_fig, moge_note = build_pointcloud_figure(
                points,
                args.moge_max_points,
                args.moge_axis_fraction,
                args.moge_axis_steps,
            )

        sam3d_path = str(item["sam3d_ply"]) if item["sam3d_ply"] else None

        notes = []
        if item["moge_npz"] is None:
            notes.append("MoGe: missing NPZ")
        elif moge_note:
            notes.append(moge_note)
        if item["sam3d_ply"] is None:
            notes.append("SAM3D: missing PLY")
        status = "OK" if not notes else " / ".join(notes)
        return image_rgb, mask_rgb, depth_rgb, moge_fig, sam3d_path, status

    with gr.Blocks() as demo:
        gr.Markdown("# SAM3D Output Viewer")
        dropdown = gr.Dropdown(
            choices=[item["stem"] for item in items],
            value=items[0]["stem"],
            label="Mask",
        )
        status = gr.Markdown()
        with gr.Row():
            img_view = gr.Image(label="Image")
            mask_view = gr.Image(label="Mask")
            depth_view = gr.Image(label="MoGe depth")
        with gr.Row():
            moge_pc_view = gr.Plot(label="MoGe point cloud (axes)")
        with gr.Row():
            model_view = gr.Model3D(label="SAM3D PLY")

        dropdown.change(
            render,
            inputs=dropdown,
            outputs=[
                img_view,
                mask_view,
                depth_view,
                moge_pc_view,
                model_view,
                status,
            ],
        )
        demo.load(
            render,
            inputs=dropdown,
            outputs=[
                img_view,
                mask_view,
                depth_view,
                moge_pc_view,
                model_view,
                status,
            ],
        )

    if args.server_port:
        print(f"Launching Gradio on port {args.server_port}...")
    else:
        print("Launching Gradio on the default port (usually 7860)...")
    demo.launch(
        share=args.share,
        server_port=args.server_port,
        inbrowser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
