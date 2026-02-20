"""SAM3D inference 실행 및 PLY/mesh 저장/시각화를 담당하는 유틸리티.

- 입력: 원본 이미지 + SAM2 마스크
- 출력: SAM3D 결과 PLY + mesh(glb/ply)
- 옵션: 생성된 PLY를 다양한 백엔드로 시각화
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from camera_intrinsics import load_intrinsics_matrix
import sam3d_scale_utils as scale_utils


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
    parser.add_argument(
        "--pose-output-dir",
        type=Path,
        default=None,
        help="Directory to write pose-applied outputs (default: same as --output).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--depth-image", type=Path, default=None)
    parser.add_argument(
        "--cam-k",
        type=Path,
        default=None,
        help="Camera intrinsics (3x3) when using depth as pointmap input.",
    )
    parser.add_argument(
        "--depth-scale",
        type=str,
        default="0.001",
        help="Depth scale to convert depth image to meters (auto | numeric, default: 0.001).",
    )
    parser.add_argument(
        "--pointmap-from-depth",
        action="store_true",
        help="Use depth image to build pointmap input for SAM3D.",
    )
    parser.add_argument(
        "--pointmap-mask",
        dest="pointmap_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask out background when building pointmap from depth (default: enabled).",
    )
    parser.add_argument("--ply", type=Path, default=None, help="PLY path for viz-only mode.")
    parser.add_argument(
        "--save-pointmap",
        dest="save_pointmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full pointmap (H x W x 3) + colors from SAM3D depth model (default: enabled).",
    )
    parser.add_argument(
        "--save-mesh",
        dest="save_mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save SAM3D mesh output (default: enabled).",
    )
    parser.add_argument(
        "--mesh-format",
        choices=["glb", "ply", "obj", "both", "all"],
        default="all",
        help="Mesh output format when saving (default: all).",
    )
    parser.add_argument("--show-viz", action="store_true")
    parser.add_argument(
        "--viz-method",
        choices=["auto", "gradio", "open3d", "trimesh", "matplotlib"],
        default="auto",
    )
    parser.add_argument("--viz-max-points", type=int, default=20000)
    parser.add_argument(
        "--pose-rot-transpose",
        action="store_true",
        help="Use R.T instead of R when applying pose (if rotation convention is transposed).",
    )
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


def read_cam_k(path: Path) -> np.ndarray:
    """3x3 intrinsics 텍스트 파일 로드."""
    return load_intrinsics_matrix(path)


def _resolve_depth_scale(depth_raw: np.ndarray, scale: str | float) -> float:
    """Resolve depth scale from user input (auto | numeric)."""
    if isinstance(scale, str):
        scale_raw = scale.strip().lower()
        if scale_raw == "auto":
            # heuristic: uint16 or large values are usually millimeters
            if depth_raw.dtype.kind in ("u", "i") and float(np.max(depth_raw)) > 50:
                return 0.001
            if 20.0 < float(np.max(depth_raw)) < 10000.0:
                return 0.001
            return 1.0
        try:
            scale_val = float(scale_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --depth-scale '{scale}'. Use auto or numeric value."
            ) from exc
    else:
        scale_val = float(scale)

    if not np.isfinite(scale_val) or scale_val <= 0.0:
        raise ValueError(f"Invalid depth scale: {scale_val}")
    return float(scale_val)


def load_depth_image(path: Path, scale: str | float) -> tuple[np.ndarray, float]:
    """Load depth image and apply resolved depth scale."""
    import cv2

    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise ValueError(f"Failed to read depth image: {path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    scale_value = _resolve_depth_scale(depth_raw, scale)
    depth = depth_raw.astype(np.float32)
    if scale_value != 1.0:
        depth *= float(scale_value)
    return depth, float(scale_value)


def depth_to_pointmap(depth: np.ndarray, cam_k: np.ndarray) -> np.ndarray:
    """Depth(H,W) -> pointmap(H,W,3) in camera frame."""
    h, w = depth.shape
    fx, fy = float(cam_k[0, 0]), float(cam_k[1, 1])
    cx, cy = float(cam_k[0, 2]), float(cam_k[1, 2])
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    z = depth
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def apply_pointmap_mask(pointmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply foreground mask on pointmap by zeroing background points."""
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.shape != pointmap.shape[:2]:
        import cv2

        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (pointmap.shape[1], pointmap.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    masked = pointmap.copy()
    masked[~mask_bool] = 0.0
    return masked


def flatten_pointmap(pointmap_hwc: np.ndarray, colors_hwc: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    """HWC pointmap/색상을 Nx3으로 변환하고 NaN/Inf 포인트 제거."""
    points = pointmap_hwc.reshape(-1, 3)
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    colors = None
    if colors_hwc is not None:
        colors = colors_hwc.reshape(-1, 3)[valid_mask]
        colors = np.clip(colors, 0.0, 1.0)
    return points, colors


class PointmapInputCompat(dict):
    """Compatibility wrapper for pointmap inputs.

    - Newer SAM3D pipelines read dict keys: {"pointmap", "intrinsics"}
    - Legacy pipelines may call .to(device) directly on pointmap input
    """

    def __init__(self, pointmap: torch.Tensor, intrinsics: np.ndarray | None):
        super().__init__()
        self["pointmap"] = pointmap
        self["intrinsics"] = intrinsics

    def to(self, *args, **kwargs):
        return self["pointmap"].to(*args, **kwargs)


def write_pointmap_ply(path: Path, points: np.ndarray, colors: np.ndarray | None) -> None:
    """포인트클라우드를 ASCII PLY로 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for px, py, pz in points:
                f.write(f"{px} {py} {pz}\n")
        else:
            colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            for (px, py, pz), (r, g, b) in zip(points, colors_u8):
                f.write(f"{px} {py} {pz} {int(r)} {int(g)} {int(b)}\n")


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """quat: (4,) in (w, x, y, z)."""
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def apply_similarity(points: np.ndarray, scale: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """scale -> rotate -> translate."""
    scaled = points * scale.reshape(1, 3)
    rotated = (rotation @ scaled.T).T
    return rotated + translation.reshape(1, 3)


def rotation_6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """6D rotation -> 3x3 matrix (Zhou et al.)."""
    r6d = r6d.astype(np.float32).reshape(-1)
    a1 = r6d[:3]
    a2 = r6d[3:6]
    x = a1 / max(1e-12, np.linalg.norm(a1))
    y = a2 - np.dot(x, a2) * x
    y = y / max(1e-12, np.linalg.norm(y))
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def parse_pose(output: dict) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    """SAM3D 출력에서 rotation/translation/scale을 정규화해 반환."""
    pose_quat = to_numpy(output.get("rotation"))
    pose_trans = to_numpy(output.get("translation"))
    pose_scale = to_numpy(output.get("scale"))

    rot_mat = None
    rot_meta = {"type": None, "value": None}
    if pose_quat is not None:
        pose_quat = np.asarray(pose_quat, dtype=np.float32).reshape(-1)
        if pose_quat.size >= 4 and pose_quat.size < 6:
            rot_mat = quaternion_to_matrix(pose_quat[:4])
            rot_meta = {"type": "quaternion_wxyz", "value": pose_quat[:4].tolist()}
        elif pose_quat.size == 6:
            rot_mat = rotation_6d_to_matrix(pose_quat[:6])
            rot_meta = {"type": "rotation_6d", "value": pose_quat[:6].tolist()}
        elif pose_quat.size == 9:
            rot_mat = pose_quat.reshape(3, 3)
            rot_meta = {"type": "rotation_matrix", "value": rot_mat.tolist()}

    trans_vec = None
    if pose_trans is not None:
        pose_trans = np.asarray(pose_trans, dtype=np.float32).reshape(-1)
        if pose_trans.size >= 3:
            trans_vec = pose_trans[:3]

    scale_vec = None
    if pose_scale is not None:
        pose_scale = np.asarray(pose_scale, dtype=np.float32).reshape(-1)
        if pose_scale.size == 1:
            scale_vec = np.repeat(pose_scale[0], 3)
        elif pose_scale.size >= 3:
            scale_vec = pose_scale[:3]

    return rot_mat, trans_vec, scale_vec, rot_meta


def save_pose_transformed_gaussian(
    source_ply: Path,
    output_ply: Path,
    scale: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    """Gaussian PLY를 pose 변환하여 저장."""
    ply, points = scale_utils.load_ply_points(source_ply)
    points_t = apply_similarity(points, scale, rotation, translation)
    scale_utils.write_scaled_ply(ply, points_t, output_ply, scale=float(np.mean(scale)))


def save_pose_transformed_mesh(
    mesh,
    output_path: Path,
    scale: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    """Mesh를 pose 변환하여 저장."""
    try:
        import trimesh
    except ImportError:
        return
    ply_axis_fix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    if isinstance(mesh, trimesh.Trimesh):
        mesh_t = mesh.copy()
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation @ np.diag(scale.astype(np.float32))
        transform[:3, 3] = translation.astype(np.float32)
        mesh_t.apply_transform(transform)
        if output_path.suffix.lower() == ".ply":
            ply_transform = np.eye(4, dtype=np.float32)
            ply_transform[:3, :3] = ply_axis_fix
            mesh_t.apply_transform(ply_transform)
        mesh_t.export(output_path)


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

    pointmap_input = None
    if args.pointmap_from_depth:
        if args.depth_image is None or args.cam_k is None:
            raise ValueError("--pointmap-from-depth requires --depth-image and --cam-k.")
        depth_path = resolve_path(args.depth_image, Path.cwd())
        cam_k_path = resolve_path(args.cam_k, Path.cwd())
        if not depth_path.exists():
            raise ValueError(f"Missing depth image: {depth_path}")
        if not cam_k_path.exists():
            raise ValueError(f"Missing camera intrinsics: {cam_k_path}")

        depth, depth_scale_value = load_depth_image(depth_path, args.depth_scale)
        print(
            f"Using depth scale for SAM3D pointmap: {depth_scale_value:.6g} "
            f"(input={args.depth_scale})"
        )
        cam_k = read_cam_k(cam_k_path)
        if depth.shape[:2] != image.shape[:2]:
            import cv2

            depth = cv2.resize(
                depth,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        pointmap = depth_to_pointmap(depth, cam_k)
        if args.pointmap_mask:
            pointmap = apply_pointmap_mask(pointmap, mask)

        try:
            from sam3d_objects.pipeline.inference_pipeline_pointmap import (
                camera_to_pytorch3d_camera,
            )
            from pytorch3d.transforms import Transform3d

            pointmap_tensor = torch.from_numpy(pointmap).float()
            points_flat = pointmap_tensor.reshape(-1, 3)
            cam_to_p3d = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=points_flat.device).rotation)
                .to(points_flat.device)
            )
            points_p3d = cam_to_p3d.transform_points(points_flat)
            pointmap_input = PointmapInputCompat(
                pointmap=points_p3d.reshape(pointmap_tensor.shape),
                intrinsics=cam_k,
            )
        except Exception as exc:
            raise RuntimeError("Failed to convert pointmap to Pytorch3D frame.") from exc

    # SAM3D 추론 실행
    try:
        output = inference(image, mask, seed=args.seed, pointmap=pointmap_input)
    except AttributeError as exc:
        # Some older pipelines expect a tensor-like pointmap and call .to(device).
        # Retry once with tensor-only input to maximize cross-version compatibility.
        if pointmap_input is not None and "has no attribute 'to'" in str(exc):
            print(
                "Pointmap input compatibility fallback: retrying with tensor-only pointmap."
            )
            retry_pointmap = pointmap_input
            if isinstance(pointmap_input, dict):
                retry_pointmap = pointmap_input.get("pointmap", pointmap_input)
            output = inference(image, mask, seed=args.seed, pointmap=retry_pointmap)
        else:
            raise
    if "gs" not in output:
        print("SAM3D output missing 'gs' key")
        return 1

    # NOTE: pointmap은 SAM3D 내부에서 Pytorch3D camera frame으로 변환되어 사용된다.
    #       비교 기준이 되는 real/moge pointcloud는 일반적인 camera frame(R3)로 생성되므로,
    #       여기서는 pointmap을 P3D -> camera frame으로 역변환하여 저장한다.
    cam_from_p3d_rot = None
    if args.save_pointmap:
        try:
            rgba = inference.merge_mask_to_rgba(image, mask)
            pointmap_dict = inference._pipeline.compute_pointmap(rgba)
            pointmap_p3d = pointmap_dict["pointmap"].detach()
            colors = pointmap_dict["pts_color"].detach().cpu().permute(1, 2, 0).numpy()
            intrinsics = pointmap_dict.get("intrinsics", None)
            if intrinsics is not None:
                intrinsics = intrinsics.detach().cpu().numpy()

            try:
                from sam3d_objects.pipeline.inference_pipeline_pointmap import (
                    camera_to_pytorch3d_camera,
                )
                from pytorch3d.transforms import Transform3d

                h, w = pointmap_p3d.shape[1], pointmap_p3d.shape[2]
                points_flat = pointmap_p3d.permute(1, 2, 0).reshape(-1, 3)
                cam_to_p3d = (
                    Transform3d()
                    .rotate(camera_to_pytorch3d_camera(device=points_flat.device).rotation)
                    .to(points_flat.device)
                )
                # P3D -> camera 변환
                points_cam = cam_to_p3d.inverse().transform_points(points_flat)
                pointmap = points_cam.reshape(h, w, 3).cpu().numpy()
                cam_from_p3d_rot = cam_to_p3d.inverse().get_matrix()[0, :3, :3].cpu().numpy()
            except Exception:
                pointmap = pointmap_p3d.cpu().permute(1, 2, 0).numpy()

            pointmap_npz = output_path.with_name(f"{output_path.stem}_pointmap_full.npz")
            np.savez_compressed(
                pointmap_npz,
                pointmap=pointmap,
                pointmap_p3d=pointmap_p3d.cpu().permute(1, 2, 0).numpy(),
                colors=colors,
                intrinsics=intrinsics,
                pointmap_frame="camera",
            )
            flat_points, flat_colors = flatten_pointmap(pointmap, colors)
            pointmap_ply = output_path.with_name(f"{output_path.stem}_pointmap_full.ply")
            write_pointmap_ply(pointmap_ply, flat_points, flat_colors)
            print(f"SAM3D pointmap saved: {pointmap_npz}")
            print(f"SAM3D pointmap ply saved: {pointmap_ply}")
        except Exception as exc:
            print(f"Failed to save pointmap output: {exc}")

    pose_rot_mat, pose_trans_vec, pose_scale_vec, pose_rot_meta = parse_pose(output)
    if pose_rot_mat is not None and args.pose_rot_transpose:
        pose_rot_mat = pose_rot_mat.T.copy()

    # Gaussian Splat 결과를 PLY로 저장
    output["gs"].save_ply(str(output_path))
    print(f"SAM3D saved: {output_path}")

    pose_output_dir = output_path.parent
    if args.pose_output_dir is not None:
        pose_output_dir = resolve_path(args.pose_output_dir, Path.cwd())
    pose_output_dir.mkdir(parents=True, exist_ok=True)

    if pose_rot_mat is not None and pose_scale_vec is not None and pose_trans_vec is not None:
        # pose가 P3D frame 기준으로 예측되었다면 camera frame으로 역변환한다.
        # (pointmap 저장 시 적용한 P3D->camera 변환과 동일한 회전)
        if cam_from_p3d_rot is not None:
            pose_rot_mat = cam_from_p3d_rot @ pose_rot_mat
            pose_trans_vec = (cam_from_p3d_rot @ pose_trans_vec.reshape(3, 1)).reshape(3)
        pose_json = pose_output_dir / f"{output_path.stem}_pose.json"
        pose_payload = {
            "rotation": pose_rot_meta,
            "translation": pose_trans_vec.tolist(),
            "scale": pose_scale_vec.tolist(),
            "rotation_matrix": pose_rot_mat.tolist(),
        }
        with pose_json.open("w", encoding="utf-8") as f:
            import json

            json.dump(pose_payload, f, indent=2)
        print(f"SAM3D pose saved: {pose_json}")

        pose_ply = pose_output_dir / f"{output_path.stem}_pose.ply"
        save_pose_transformed_gaussian(
            output_path, pose_ply, pose_scale_vec, pose_rot_mat, pose_trans_vec
        )
        print(f"SAM3D pose PLY saved: {pose_ply}")

    if args.save_mesh:
        mesh_raw = output.get("mesh")
        if isinstance(mesh_raw, (list, tuple)) and mesh_raw:
            mesh_raw = mesh_raw[0]
        mesh_glb = output.get("glb")
        mesh = mesh_glb or mesh_raw
        if mesh is None:
            print("SAM3D output missing mesh; skip mesh export.")
        else:
            try:
                if args.mesh_format in ("glb", "both", "all") and mesh_glb is not None:
                    mesh_path = output_path.with_name(f"{output_path.stem}_mesh.glb")
                    mesh_glb.export(mesh_path)
                    print(f"SAM3D mesh saved: {mesh_path}")
                if args.mesh_format in ("ply", "both", "all"):
                    mesh_path = output_path.with_name(f"{output_path.stem}_mesh.ply")
                    mesh_to_export = mesh.copy()
                    ply_transform = np.eye(4, dtype=np.float32)
                    ply_transform[:3, :3] = np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, -1.0, 0.0],
                        ],
                        dtype=np.float32,
                    )
                    mesh_to_export.apply_transform(ply_transform)
                    mesh_to_export.export(mesh_path)
                    print(f"SAM3D mesh saved: {mesh_path}")
                if args.mesh_format in ("obj", "all"):
                    mesh_path = output_path.with_name(f"{output_path.stem}_mesh.obj")
                    mesh.export(mesh_path)
                    print(f"SAM3D mesh saved: {mesh_path}")
                if pose_rot_mat is not None and pose_scale_vec is not None and pose_trans_vec is not None:
                    pose_mesh_source = mesh_raw if mesh_raw is not None else mesh
                    if args.mesh_format in ("glb", "both", "all"):
                        mesh_path = pose_output_dir / f"{output_path.stem}_pose_mesh.glb"
                        save_pose_transformed_mesh(pose_mesh_source, mesh_path, pose_scale_vec, pose_rot_mat, pose_trans_vec)
                        print(f"SAM3D pose mesh saved: {mesh_path}")
                    if args.mesh_format in ("ply", "both", "all"):
                        mesh_path = pose_output_dir / f"{output_path.stem}_pose_mesh.ply"
                        save_pose_transformed_mesh(pose_mesh_source, mesh_path, pose_scale_vec, pose_rot_mat, pose_trans_vec)
                        print(f"SAM3D pose mesh saved: {mesh_path}")
                    if args.mesh_format in ("obj", "all"):
                        mesh_path = pose_output_dir / f"{output_path.stem}_pose_mesh.obj"
                        save_pose_transformed_mesh(pose_mesh_source, mesh_path, pose_scale_vec, pose_rot_mat, pose_trans_vec)
                        print(f"SAM3D pose mesh saved: {mesh_path}")
            except Exception as exc:
                print(f"Failed to save mesh: {exc}")

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
