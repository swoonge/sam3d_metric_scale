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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
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
    if isinstance(mesh, trimesh.Trimesh):
        mesh_t = mesh.copy()
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation @ np.diag(scale.astype(np.float32))
        transform[:3, 3] = translation.astype(np.float32)
        mesh_t.apply_transform(transform)
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

    # SAM3D 추론 실행
    output = inference(image, mask, seed=args.seed)
    if "gs" not in output:
        print("SAM3D output missing 'gs' key")
        return 1

    if args.save_pointmap:
        try:
            rgba = inference.merge_mask_to_rgba(image, mask)
            pointmap_dict = inference._pipeline.compute_pointmap(rgba)
            pointmap = pointmap_dict["pointmap"].detach().cpu().permute(1, 2, 0).numpy()
            colors = pointmap_dict["pts_color"].detach().cpu().permute(1, 2, 0).numpy()
            intrinsics = pointmap_dict.get("intrinsics", None)
            if intrinsics is not None:
                intrinsics = intrinsics.detach().cpu().numpy()

            pointmap_npz = output_path.with_name(f"{output_path.stem}_pointmap_full.npz")
            np.savez_compressed(
                pointmap_npz,
                pointmap=pointmap,
                colors=colors,
                intrinsics=intrinsics,
            )
            flat_points, flat_colors = flatten_pointmap(pointmap, colors)
            pointmap_ply = output_path.with_name(f"{output_path.stem}_pointmap_full.ply")
            write_pointmap_ply(pointmap_ply, flat_points, flat_colors)
            print(f"SAM3D pointmap saved: {pointmap_npz}")
            print(f"SAM3D pointmap ply saved: {pointmap_ply}")
        except Exception as exc:
            print(f"Failed to save pointmap output: {exc}")

    pose_quat = to_numpy(output.get("rotation"))
    pose_trans = to_numpy(output.get("translation"))
    pose_scale = to_numpy(output.get("scale"))
    pose_rot_mat = None
    pose_scale_vec = None
    pose_trans_vec = None

    if pose_quat is not None:
        pose_quat = np.asarray(pose_quat, dtype=np.float32).reshape(-1)
        if pose_quat.size >= 4:
            pose_rot_mat = quaternion_to_matrix(pose_quat[:4])
        elif pose_quat.size == 6:
            pose_rot_mat = rotation_6d_to_matrix(pose_quat[:6])
        elif pose_quat.size == 9:
            pose_rot_mat = pose_quat.reshape(3, 3)

    if pose_scale is not None:
        pose_scale = np.asarray(pose_scale, dtype=np.float32).reshape(-1)
        if pose_scale.size == 1:
            pose_scale_vec = np.repeat(pose_scale[0], 3)
        elif pose_scale.size >= 3:
            pose_scale_vec = pose_scale[:3]

    if pose_trans is not None:
        pose_trans = np.asarray(pose_trans, dtype=np.float32).reshape(-1)
        if pose_trans.size >= 3:
            pose_trans_vec = pose_trans[:3]

    # Gaussian Splat 결과를 PLY로 저장
    output["gs"].save_ply(str(output_path))
    print(f"SAM3D saved: {output_path}")

    if pose_rot_mat is not None and pose_scale_vec is not None and pose_trans_vec is not None:
        pose_json = output_path.with_name(f"{output_path.stem}_pose.json")
        rotation_payload = {"type": None, "value": None}
        if pose_quat is not None:
            if pose_quat.size >= 4 and pose_quat.size < 6:
                rotation_payload = {
                    "type": "quaternion_wxyz",
                    "value": pose_quat[:4].tolist(),
                }
            elif pose_quat.size == 6:
                rotation_payload = {
                    "type": "rotation_6d",
                    "value": pose_quat[:6].tolist(),
                }
            elif pose_quat.size == 9:
                rotation_payload = {
                    "type": "rotation_matrix",
                    "value": pose_quat[:9].reshape(3, 3).tolist(),
                }
        pose_payload = {
            "rotation": rotation_payload,
            "translation": pose_trans_vec.tolist(),
            "scale": pose_scale_vec.tolist(),
        }
        with pose_json.open("w", encoding="utf-8") as f:
            import json

            json.dump(pose_payload, f, indent=2)
        print(f"SAM3D pose saved: {pose_json}")

        pose_ply = output_path.with_name(f"{output_path.stem}_pose.ply")
        save_pose_transformed_gaussian(
            output_path, pose_ply, pose_scale_vec, pose_rot_mat, pose_trans_vec
        )
        print(f"SAM3D pose PLY saved: {pose_ply}")

    if args.save_mesh:
        mesh = output.get("glb") or output.get("mesh")
        if mesh is None:
            print("SAM3D output missing mesh; skip mesh export.")
        else:
            try:
                if args.mesh_format in ("glb", "both", "all"):
                    mesh_path = output_path.with_name(f"{output_path.stem}_mesh.glb")
                    mesh.export(mesh_path)
                    print(f"SAM3D mesh saved: {mesh_path}")
                if args.mesh_format in ("ply", "both", "all"):
                    mesh_path = output_path.with_name(f"{output_path.stem}_mesh.ply")
                    mesh.export(mesh_path)
                    print(f"SAM3D mesh saved: {mesh_path}")
                if args.mesh_format in ("obj", "all"):
                    mesh_path = output_path.with_name(f"{output_path.stem}_mesh.obj")
                    mesh.export(mesh_path)
                    print(f"SAM3D mesh saved: {mesh_path}")
                if pose_rot_mat is not None and pose_scale_vec is not None and pose_trans_vec is not None:
                    if args.mesh_format in ("glb", "both", "all"):
                        mesh_path = output_path.with_name(f"{output_path.stem}_pose_mesh.glb")
                        save_pose_transformed_mesh(mesh, mesh_path, pose_scale_vec, pose_rot_mat, pose_trans_vec)
                        print(f"SAM3D pose mesh saved: {mesh_path}")
                    if args.mesh_format in ("ply", "both", "all"):
                        mesh_path = output_path.with_name(f"{output_path.stem}_pose_mesh.ply")
                        save_pose_transformed_mesh(mesh, mesh_path, pose_scale_vec, pose_rot_mat, pose_trans_vec)
                        print(f"SAM3D pose mesh saved: {mesh_path}")
                    if args.mesh_format in ("obj", "all"):
                        mesh_path = output_path.with_name(f"{output_path.stem}_pose_mesh.obj")
                        save_pose_transformed_mesh(mesh, mesh_path, pose_scale_vec, pose_rot_mat, pose_trans_vec)
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
