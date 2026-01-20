"""결과 폴더 기반 통합 시각화(Gradio UI).

- 입력: outputs/<image_stem>/ 하위의 mask/moge/sam3d 결과
- 출력: 이미지/마스크/깊이/포인트클라우드/PLY/mesh 뷰어를 한 화면에 제공
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image



def parse_args() -> argparse.Namespace:
    """CLI 인자 정의."""
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
    """상대 경로를 절대 경로로 변환."""
    if path.is_absolute():
        return path
    return (base / path).resolve()


def find_image_path(output_root: Path, override: Optional[Path]) -> Optional[Path]:
    """출력 루트/메타정보에서 원본 이미지 경로를 유추."""
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
    """이미지를 RGB numpy 배열로 로드."""
    image = Image.open(path).convert("RGB")
    return np.array(image)


def load_mask(path: Path) -> np.ndarray:
    """마스크를 bool 배열로 로드."""
    mask = Image.open(path).convert("L")
    return np.array(mask) > 0


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """bool 마스크를 0/255 RGB 이미지로 변환."""
    return np.stack([mask * 255] * 3, axis=2).astype(np.uint8)


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """깊이를 컬러맵으로 시각화할 수 있는 RGB 이미지로 변환."""
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
    """MoGe NPZ에서 valid_mask + depth_masked를 복원하여 full depth 생성."""
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
    """valid_mask + depth_masked로 간단한 3D 포인트 복원."""
    if depth_masked.shape[0] != int(valid_mask.sum()):
        return None
    ys, xs = np.where(valid_mask)
    height, width = valid_mask.shape
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    focal = max(height, width)
    # 간단한 pinhole 모델 근사(정확한 intrinsics 없을 때 fallback)
    z = depth_masked.astype(np.float32)
    x = (xs - cx) * z / focal
    y = -(ys - cy) * z / focal
    return np.stack([x, y, z], axis=1)


def load_moge_points(
    npz_path: Path,
) -> tuple[Optional[np.ndarray], Optional[tuple[np.ndarray, np.ndarray, tuple[int, int]]]]:
    """MoGe NPZ에서 points_masked 우선 사용, 없으면 depth로 복원.

    반환:
    - points: (N,3) 포인트
    - pixel_coords: (ys, xs, (H, W)) 형태의 픽셀 좌표 매핑(경계 필터용)
    """
    if npz_path is None or not npz_path.exists():
        return None, None
    data = np.load(npz_path)
    points = data.get("points_masked")
    valid_mask = data.get("valid_mask")
    pixel_coords = None
    if points is not None:
        points = points.astype(np.float32)
        points = points[np.isfinite(points).all(axis=1)]
        if valid_mask is not None:
            valid_mask = valid_mask.astype(bool)
            ys, xs = np.where(valid_mask)
            if points.shape[0] == ys.shape[0]:
                pixel_coords = (ys, xs, valid_mask.shape)
        return points, pixel_coords
    depth_masked = data.get("depth_masked")
    if valid_mask is None or depth_masked is None:
        return None, None
    valid_mask = valid_mask.astype(bool)
    depth_masked = depth_masked.astype(np.float32)
    points = points_from_masked_depth(valid_mask, depth_masked)
    if points is None:
        return None, None
    points = points[np.isfinite(points).all(axis=1)]
    ys, xs = np.where(valid_mask)
    pixel_coords = (ys, xs, valid_mask.shape)
    return points, pixel_coords


def colorize_values(values: np.ndarray) -> np.ndarray:
    """스칼라 값 배열을 RGB 컬러로 매핑."""
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
    """MoGe 포인트클라우드를 3D 플롯으로 렌더링."""
    if points is None or points.size == 0:
        return None, "MoGe PC: empty"

    points = points.astype(np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        return None, "MoGe PC: invalid points"

    points_vis = points
    if max_points > 0 and points.shape[0] > max_points:
        # 시각화용 포인트 수 제한
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

        # plotly가 있으면 인터랙티브 3D로 렌더링
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
        # 축 시각화를 위해 x/y/z 축 라인 추가
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

        # plotly가 없을 때 matplotlib 3D로 fallback
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


def load_ply_points(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """PLY에서 포인트/색상 로드."""
    try:
        from plyfile import PlyData
    except ImportError:
        raise RuntimeError("plyfile is required to parse PLY.")

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise RuntimeError("PLY file missing vertex data.")
    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()
    points = np.stack(
        (vertex["x"], vertex["y"], vertex["z"]),
        axis=1,
    ).astype(np.float32)

    colors = None
    if all(name in names for name in ("red", "green", "blue")):
        colors = np.stack(
            (vertex["red"], vertex["green"], vertex["blue"]),
            axis=1,
        ).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
    elif all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        colors = np.stack(
            (vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]),
            axis=1,
        ).astype(np.float32)
        colors = np.clip(colors + 0.5, 0.0, 1.0)

    return points, colors


def build_pointcloud_plot_from_ply(
    path: Optional[Path],
    max_points: int,
    axis_fraction: float,
    axis_steps: int,
) -> tuple[object, Optional[str]]:
    if path is None or not path.exists():
        return None, "PLY missing"
    try:
        points, _ = load_ply_points(path)
    except Exception as exc:
        return None, f"PLY load failed: {exc}"
    return build_pointcloud_figure(points, max_points, axis_fraction, axis_steps)


def find_moge_npz(moge_dir: Path, mask_stem: str) -> Optional[Path]:
    """마스크 stem에 대응되는 MoGe NPZ 파일 탐색."""
    candidates = sorted(moge_dir.glob(f"*_{mask_stem}.npz"))
    if not candidates:
        candidates = sorted(moge_dir.glob(f"{mask_stem}.npz"))
    return candidates[0] if candidates else None


def find_sam3d_mesh(sam3d_dir: Path, mask_stem: str) -> Optional[Path]:
    """SAM3D 메쉬 출력(glb) 탐색."""
    candidates = [
        sam3d_dir / f"{mask_stem}_mesh.glb",
        sam3d_dir / f"{mask_stem}.glb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_items(output_root: Path):
    """출력 루트 기준으로 mask/moge/sam3d/scale 파일 묶음을 구성."""
    mask_dir = output_root / "sam2_masks"
    moge_dir = output_root / "moge_scale"
    sam3d_dir = output_root / "sam3d"
    scale_dir = output_root / "sam3d_scale"
    analysis_dir = output_root / "sam3d_analysis"
    camera_scale_dir = output_root / "sam3d_camera_scale_viz"

    masks = sorted(mask_dir.glob("*.png"))
    items = []
    for mask_path in masks:
        stem = mask_path.stem
        moge_npz = find_moge_npz(moge_dir, stem)
        sam3d_ply = sam3d_dir / f"{stem}.ply"
        sam3d_mesh = find_sam3d_mesh(sam3d_dir, stem)
        scale_txt = scale_dir / f"{stem}_scale.txt"
        scaled_ply = scale_dir / f"{stem}_scaled.ply"
        analysis_depth = analysis_dir / f"{stem}_pointmap_depth.png"
        analysis_pointmap = analysis_dir / f"{stem}_pointmap_cloud.ply"
        analysis_coords = analysis_dir / f"{stem}_coords_cloud.ply"
        analysis_md = analysis_dir / f"{stem}_analysis.md"
        camera_dir = camera_scale_dir / stem
        camera_moge_depth = camera_dir / "03_moge_depth_full.png"
        camera_pointmap_depth = camera_dir / "04_pointmap_depth.png"
        camera_sam3d_ply = camera_dir / "05_sam3d_gaussian.ply"
        camera_scaled_ply = camera_dir / "06_sam3d_scaled.ply"
        camera_camera_ply = camera_dir / "07_sam3d_camera.ply"
        camera_depth = camera_dir / "08_camera_depth.png"
        camera_summary = camera_dir / "summary.txt"
        items.append(
            {
                "stem": stem,
                "mask": mask_path,
                "moge_npz": moge_npz,
                "sam3d_ply": sam3d_ply if sam3d_ply.exists() else None,
                "sam3d_mesh": sam3d_mesh,
                "scale_txt": scale_txt if scale_txt.exists() else None,
                "scaled_ply": scaled_ply if scaled_ply.exists() else None,
                "analysis_depth": analysis_depth if analysis_depth.exists() else None,
                "analysis_pointmap": analysis_pointmap if analysis_pointmap.exists() else None,
                "analysis_coords": analysis_coords if analysis_coords.exists() else None,
                "analysis_md": analysis_md if analysis_md.exists() else None,
                "camera_dir": camera_dir if camera_dir.exists() else None,
                "camera_moge_depth": camera_moge_depth if camera_moge_depth.exists() else None,
                "camera_pointmap_depth": camera_pointmap_depth
                if camera_pointmap_depth.exists()
                else None,
                "camera_sam3d_ply": camera_sam3d_ply if camera_sam3d_ply.exists() else None,
                "camera_scaled_ply": camera_scaled_ply if camera_scaled_ply.exists() else None,
                "camera_camera_ply": camera_camera_ply if camera_camera_ply.exists() else None,
                "camera_depth": camera_depth if camera_depth.exists() else None,
                "camera_summary": camera_summary if camera_summary.exists() else None,
            }
        )
    return items


def print_summary(image_path: Path, items) -> None:
    """gradio 미설치 시 콘솔 요약 출력."""
    print("Visualization summary:")
    print(f"Image: {image_path}")
    for item in items:
        stem = item["stem"]
        mask_path = item["mask"]
        moge_npz = item["moge_npz"]
        sam3d_ply = item["sam3d_ply"]
        sam3d_mesh = item["sam3d_mesh"]
        analysis_depth = item["analysis_depth"]
        analysis_pointmap = item["analysis_pointmap"]
        analysis_coords = item["analysis_coords"]
        analysis_md = item["analysis_md"]
        camera_moge_depth = item["camera_moge_depth"]
        camera_pointmap_depth = item["camera_pointmap_depth"]
        camera_sam3d_ply = item["camera_sam3d_ply"]
        camera_scaled_ply = item["camera_scaled_ply"]
        camera_camera_ply = item["camera_camera_ply"]
        camera_depth = item["camera_depth"]
        camera_summary = item["camera_summary"]
        moge_status = str(moge_npz) if moge_npz else "missing"
        sam3d_status = str(sam3d_ply) if sam3d_ply else "missing"
        print(f"- {stem}")
        print(f"  mask: {mask_path}")
        print(f"  moge: {moge_status}")
        print(f"  sam3d: {sam3d_status}")
        if sam3d_mesh:
            print(f"  sam3d mesh: {sam3d_mesh}")


def main() -> int:
    """Gradio UI 진입점."""
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
        """드롭다운 선택에 따라 각 결과를 로드/갱신."""
        item = item_map[stem]
        mask = load_mask(item["mask"])
        mask_rgb = mask_to_rgb(mask)
        depth_full = load_moge_depth(item["moge_npz"]) if item["moge_npz"] else None
        depth_rgb = depth_to_rgb(depth_full) if depth_full is not None else None
        analysis_depth = item["analysis_depth"]
        analysis_pointmap = item["analysis_pointmap"]
        analysis_coords = item["analysis_coords"]
        analysis_md = item["analysis_md"]

        moge_fig = None
        moge_note = None
        if item["moge_npz"] is not None:
            # MoGe 포인트클라우드 플롯 생성
            points, _ = load_moge_points(item["moge_npz"])
            moge_fig, moge_note = build_pointcloud_figure(
                points,
                args.moge_max_points,
                args.moge_axis_fraction,
                args.moge_axis_steps,
            )

        sam3d_fig, sam3d_note = build_pointcloud_plot_from_ply(
            item["sam3d_ply"],
            args.moge_max_points,
            args.moge_axis_fraction,
            args.moge_axis_steps,
        )
        sam3d_mesh_path = str(item["sam3d_mesh"]) if item["sam3d_mesh"] else None
        scaled_fig, scaled_note = build_pointcloud_plot_from_ply(
            item["scaled_ply"],
            args.moge_max_points,
            args.moge_axis_fraction,
            args.moge_axis_steps,
        )
        scale_info = "Scale: N/A"
        if item["scale_txt"] is not None:
            try:
                scale_text = item["scale_txt"].read_text(encoding="utf-8").strip()
                scale_info = f"Scale: {scale_text}"
            except Exception:
                scale_info = "Scale: parse error"

        notes = []
        if item["moge_npz"] is None:
            notes.append("MoGe: missing NPZ")
        elif moge_note:
            notes.append(moge_note)
        if item["sam3d_ply"] is None:
            notes.append("SAM3D: missing PLY")
        if item["sam3d_mesh"] is None:
            notes.append("SAM3D: missing mesh")
        if item["scale_txt"] is None:
            notes.append("Scale: missing TXT")
        if item["analysis_depth"] is None:
            notes.append("Analysis: missing pointmap depth")
        if item["analysis_pointmap"] is None:
            notes.append("Analysis: missing pointmap cloud")
        if sam3d_note and sam3d_note != "PLY missing":
            notes.append(f"SAM3D PLY: {sam3d_note}")
        if scaled_note and scaled_note != "PLY missing":
            notes.append(f"Scaled PLY: {scaled_note}")
        status = "OK" if not notes else " / ".join(notes)
        analysis_text = None
        if analysis_md is not None:
            try:
                analysis_text = analysis_md.read_text(encoding="utf-8")
            except Exception:
                analysis_text = None
        analysis_pointmap_fig, analysis_pointmap_note = build_pointcloud_plot_from_ply(
            analysis_pointmap,
            args.moge_max_points,
            args.moge_axis_fraction,
            args.moge_axis_steps,
        )
        analysis_coords_fig, analysis_coords_note = build_pointcloud_plot_from_ply(
            analysis_coords,
            args.moge_max_points,
            args.moge_axis_fraction,
            args.moge_axis_steps,
        )
        if analysis_pointmap_note and analysis_pointmap_note != "PLY missing":
            notes.append(f"Analysis pointmap: {analysis_pointmap_note}")
        if analysis_coords_note and analysis_coords_note != "PLY missing":
            notes.append(f"Analysis coords: {analysis_coords_note}")
        camera_sam3d_fig, camera_sam3d_note = build_pointcloud_plot_from_ply(
            camera_sam3d_ply,
            args.moge_max_points,
            args.moge_axis_fraction,
            args.moge_axis_steps,
        )
        camera_scaled_fig, camera_scaled_note = build_pointcloud_plot_from_ply(
            camera_scaled_ply,
            args.moge_max_points,
            args.moge_axis_fraction,
            args.moge_axis_steps,
        )
        camera_camera_fig, camera_camera_note = build_pointcloud_plot_from_ply(
            camera_camera_ply,
            args.moge_max_points,
            args.moge_axis_fraction,
            args.moge_axis_steps,
        )
        if camera_sam3d_note and camera_sam3d_note != "PLY missing":
            notes.append(f"Camera viz SAM3D: {camera_sam3d_note}")
        if camera_scaled_note and camera_scaled_note != "PLY missing":
            notes.append(f"Camera viz scaled: {camera_scaled_note}")
        if camera_camera_note and camera_camera_note != "PLY missing":
            notes.append(f"Camera viz camera: {camera_camera_note}")
        camera_summary_text = None
        if camera_summary is not None:
            try:
                camera_summary_text = camera_summary.read_text(encoding="utf-8")
            except Exception:
                camera_summary_text = None
        return (
            image_rgb,
            mask_rgb,
            depth_rgb,
            moge_fig,
            sam3d_mesh_path,
            sam3d_fig,
            scaled_fig,
            status,
            scale_info,
            str(analysis_depth) if analysis_depth else None,
            analysis_pointmap_fig,
            analysis_coords_fig,
            analysis_text,
            str(camera_moge_depth) if camera_moge_depth else None,
            str(camera_pointmap_depth) if camera_pointmap_depth else None,
            camera_sam3d_fig,
            camera_scaled_fig,
            camera_camera_fig,
            str(camera_depth) if camera_depth else None,
            camera_summary_text,
        )

    with gr.Blocks() as demo:
        gr.Markdown("# SAM3D Output Viewer")
        # 마스크별 결과 선택
        dropdown = gr.Dropdown(
            choices=[item["stem"] for item in items],
            value=items[0]["stem"],
            label="Mask",
        )
        status = gr.Markdown()
        scale_info = gr.Markdown()
        with gr.Row():
            img_view = gr.Image(label="Image")
            mask_view = gr.Image(label="Mask")
            depth_view = gr.Image(label="MoGe depth")
        with gr.Row():
            moge_pc_view = gr.Plot(label="MoGe point cloud (axes)")
        with gr.Row():
            mesh_view = gr.Model3D(label="SAM3D Mesh (GLB)")
        with gr.Row():
            sam3d_ply_view = gr.Plot(label="SAM3D Gaussian PLY (point cloud)")
            scaled_ply_view = gr.Plot(label="SAM3D scaled Gaussian (point cloud)")
        with gr.Row():
            analysis_depth_view = gr.Image(label="Analysis: pointmap depth")
        with gr.Row():
            analysis_pointmap_view = gr.Plot(label="Analysis: pointmap cloud")
            analysis_coords_view = gr.Plot(label="Analysis: sparse coords")
        analysis_md_view = gr.Markdown()
        gr.Markdown("## SAM3D Camera/Scale Viz")
        with gr.Row():
            camera_moge_depth_view = gr.Image(label="CameraViz: MoGe depth (full)")
            camera_pointmap_depth_view = gr.Image(label="CameraViz: pointmap depth")
            camera_depth_view = gr.Image(label="CameraViz: camera depth")
        with gr.Row():
            camera_sam3d_view = gr.Plot(label="CameraViz: SAM3D Gaussian")
            camera_scaled_view = gr.Plot(label="CameraViz: scaled SAM3D")
            camera_camera_view = gr.Plot(label="CameraViz: camera frame")
        camera_summary_view = gr.Markdown()

        # 드롭다운 변경 시 결과 갱신
        dropdown.change(
            render,
            inputs=dropdown,
            outputs=[
                img_view,
                mask_view,
                depth_view,
                moge_pc_view,
                mesh_view,
                sam3d_ply_view,
                scaled_ply_view,
                status,
                scale_info,
                analysis_depth_view,
                analysis_pointmap_view,
                analysis_coords_view,
                analysis_md_view,
                camera_moge_depth_view,
                camera_pointmap_depth_view,
                camera_sam3d_view,
                camera_scaled_view,
                camera_camera_view,
                camera_depth_view,
                camera_summary_view,
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
                mesh_view,
                sam3d_ply_view,
                scaled_ply_view,
                status,
                scale_info,
                analysis_depth_view,
                analysis_pointmap_view,
                analysis_coords_view,
                analysis_md_view,
                camera_moge_depth_view,
                camera_pointmap_depth_view,
                camera_sam3d_view,
                camera_scaled_view,
                camera_camera_view,
                camera_depth_view,
                camera_summary_view,
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
