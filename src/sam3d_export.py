"""SAM3D inference 실행 및 PLY 저장/시각화를 담당하는 유틸리티.

- 입력: 원본 이미지 + SAM2 마스크
- 출력: SAM3D 결과 PLY
- 옵션: 생성된 PLY를 다양한 백엔드로 시각화
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


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
    """CLI 인자 정의."""
    repo_root = Path(__file__).resolve().parents[1]
    sam3d_root = resolve_sam3d_root(repo_root)
    default_config = sam3d_root / "checkpoints" / "hf" / "pipeline.yaml"
    default_output_dir = repo_root / "outputs" / "sam3d"

    parser = argparse.ArgumentParser(description="SAM3D export from image + mask")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--sam3d-config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--ply", type=Path, default=None, help="PLY path for viz-only mode.")
    parser.add_argument("--show-viz", action="store_true")
    parser.add_argument(
        "--viz-method",
        choices=["auto", "gradio", "open3d", "trimesh", "matplotlib"],
        default="auto",
    )
    parser.add_argument("--viz-max-points", type=int, default=20000)
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    """상대 경로를 실행 디렉터리 기준 절대 경로로 변환."""
    if path.is_absolute():
        return path
    return (base / path).resolve()


def sample_points(points: np.ndarray, colors: np.ndarray | None, max_points: int):
    """시각화 부담을 줄이기 위해 포인트 수를 제한."""
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(0)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    sampled_colors = colors[idx] if colors is not None else None
    return points[idx], sampled_colors


def load_ply_points(path: Path):
    """PLY 파일에서 포인트/색상을 로드."""
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("plyfile is required to parse SAM3D ply outputs.") from exc

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
        # 일반 RGB 컬러
        colors = np.stack(
            (vertex["red"], vertex["green"], vertex["blue"]),
            axis=1,
        ).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
    elif all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        # Gaussian Splatting 계열의 색상 채널
        colors = np.stack(
            (vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]),
            axis=1,
        ).astype(np.float32)
        colors = np.clip(colors + 0.5, 0.0, 1.0)

    return points, colors


def visualize_with_open3d(points: np.ndarray, colors: np.ndarray | None) -> bool:
    """Open3D로 로컬 뷰어를 띄운다."""
    try:
        import open3d as o3d
    except ImportError:
        return False

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([cloud], window_name="sam3d output")
    return True


def visualize_with_trimesh(points: np.ndarray, colors: np.ndarray | None) -> bool:
    """trimesh 뷰어로 포인트클라우드 시각화."""
    try:
        import trimesh
    except ImportError:
        return False

    trimesh_colors = None
    if colors is not None:
        if colors.max() <= 1.0:
            trimesh_colors = (colors * 255).astype(np.uint8)
        else:
            trimesh_colors = colors.astype(np.uint8)
    cloud = trimesh.points.PointCloud(points, colors=trimesh_colors)
    cloud.show()
    return True


def visualize_with_matplotlib(points: np.ndarray, colors: np.ndarray | None) -> bool:
    """matplotlib 3D 산점도로 간단 시각화."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    else:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],
            s=1,
            cmap="viridis",
        )
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(1e-6, float((maxs - mins).max() / 2.0))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title("sam3d output")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    plt.show()
    return True


def visualize_with_gradio(ply_path: Path) -> bool:
    """gradio Model3D로 PLY를 웹 UI에서 확인."""
    try:
        import gradio as gr
    except ImportError:
        return False

    with gr.Blocks() as demo:
        gr.Markdown("# SAM3D Gaussian Splat Viewer")
        gr.Model3D(value=str(ply_path), label="3D Scene")
    demo.launch(share=False)
    return True


def visualize_ply(path: Path, method: str, max_points: int) -> bool:
    """지정된 백엔드로 PLY 시각화. auto는 순차 시도."""
    if method == "auto":
        for candidate in ("gradio", "open3d", "trimesh", "matplotlib"):
            if visualize_ply(path, candidate, max_points):
                return True
        return False

    if method == "gradio":
        return visualize_with_gradio(path)

    try:
        points, colors = load_ply_points(path)
    except RuntimeError as exc:
        print(f"Failed to parse ply: {exc}")
        return False

    if points.size == 0:
        print("No points found in ply output.")
        return False

    points, colors = sample_points(points, colors, max_points)

    if method == "open3d":
        return visualize_with_open3d(points, colors)
    if method == "trimesh":
        return visualize_with_trimesh(points, colors)
    if method == "matplotlib":
        return visualize_with_matplotlib(points, colors)

    return False


def main() -> int:
    """main 진입점."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sam3d_root = resolve_sam3d_root(repo_root)

    if args.ply is not None:
        # PLY만 전달된 경우 시각화 전용 모드로 동작
        ply_path = resolve_path(args.ply, Path.cwd())
        if not ply_path.exists():
            print(f"Missing ply: {ply_path}")
            return 1
        if not args.show_viz:
            print("PLY provided; launching visualization.")
        ok = visualize_ply(ply_path, args.viz_method, args.viz_max_points)
        if not ok:
            print(
                "Failed to visualize SAM3D output. Install one of: gradio, open3d, trimesh, matplotlib."
            )
            return 1
        return 0

    if args.image is None or args.mask is None:
        print("Missing --image/--mask. Provide both or use --ply for visualization only.")
        return 1

    image_path = resolve_path(args.image, Path.cwd())
    mask_path = resolve_path(args.mask, Path.cwd())
    config_path = resolve_path(args.sam3d_config, sam3d_root)

    if not sam3d_root.exists():
        print(f"SAM3D root not found: {sam3d_root}")
        return 1
    if not image_path.exists():
        print(f"Missing image: {image_path}")
        return 1
    if not mask_path.exists():
        print(f"Missing mask: {mask_path}")
        return 1
    if not config_path.exists():
        print(f"Missing SAM3D config: {config_path}")
        return 1

    if args.output is not None:
        output_path = resolve_path(args.output, Path.cwd())
    else:
        # 출력 경로 미지정 시 기본 출력 디렉터리 사용
        output_dir = resolve_path(args.output_dir, Path.cwd())
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{image_path.stem}_{mask_path.stem}.ply"
        output_path = output_dir / output_name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # sam-3d-objects 패키지를 import 가능하도록 경로 등록
    if str(sam3d_root) not in sys.path:
        sys.path.insert(0, str(sam3d_root))
    os.chdir(sam3d_root)
    sys.path.append(str(sam3d_root / "notebook"))
    from inference import Inference, load_image, load_mask

    inference = Inference(str(config_path), compile=args.compile)
    image = load_image(str(image_path))
    mask = load_mask(str(mask_path))

    # SAM3D 추론 실행
    output = inference(image, mask, seed=args.seed)
    if "gs" not in output:
        print("SAM3D output missing 'gs' key")
        return 1

    # Gaussian Splat 결과를 PLY로 저장
    output["gs"].save_ply(str(output_path))
    print(f"SAM3D saved: {output_path}")

    if args.show_viz:
        # 필요할 때만 PLY 시각화 수행
        ok = visualize_ply(output_path, args.viz_method, args.viz_max_points)
        if not ok:
            print(
                "Failed to visualize SAM3D output. Install one of: gradio, open3d, trimesh, matplotlib."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
