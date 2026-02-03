"""Mesh decimation utility for scaled SAM3D meshes.

- Input: scaled mesh file (glb/ply/obj)
  - Output: decimated mesh with fewer faces
  - Methods: open3d (quadric), trimesh (quadric), or simple vertex clustering fallback
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decimate mesh to reduce size.")
    parser.add_argument("--input", type=Path, required=True, help="Input mesh path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mesh path (default: append _decimated to input name).",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Target face ratio in (0, 1]. Ignored if --target-faces is set.",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=0,
        help="Target number of faces (overrides --ratio).",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "open3d", "trimesh", "cluster"],
        default="auto",
        help="Decimation backend (default: auto).",
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=200,
        help="Minimum face count to keep (default: 200).",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_decimated{input_path.suffix}")


def load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Scene has no geometry: {path}")
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type: {type(mesh)}")

    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


def _vertex_colors(mesh) -> np.ndarray | None:
    colors = None
    if hasattr(mesh, "visual") and mesh.visual is not None:
        if mesh.visual.kind == "vertex":
            colors = mesh.visual.vertex_colors
            if colors is not None and colors.size > 0:
                colors = colors[:, :3].astype(np.float32)
                if colors.max() > 1.0:
                    colors = colors / 255.0
    return colors


def simplify_open3d(mesh, target_faces: int):
    import open3d as o3d
    import trimesh

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    colors = _vertex_colors(mesh)
    if colors is not None and colors.shape[0] == mesh.vertices.shape[0]:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d_mesh = o3d_mesh.simplify_quadric_decimation(int(target_faces))
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_unreferenced_vertices()

    verts = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if o3d_mesh.has_vertex_colors():
        vcolors = np.asarray(o3d_mesh.vertex_colors)
        vcolors_u8 = np.clip(vcolors * 255.0, 0, 255).astype(np.uint8)
        mesh_out.visual.vertex_colors = vcolors_u8
    return mesh_out, "open3d"


def simplify_trimesh(mesh, target_faces: int):
    if not hasattr(mesh, "simplify_quadratic_decimation"):
        raise RuntimeError("trimesh.simplify_quadratic_decimation not available.")
    mesh_out = mesh.simplify_quadratic_decimation(int(target_faces))
    return mesh_out, "trimesh"


def simplify_cluster(mesh, target_faces: int):
    import trimesh

    if target_faces <= 0:
        return mesh, "cluster"

    faces = mesh.faces
    verts = mesh.vertices
    if faces.shape[0] <= target_faces:
        return mesh, "cluster"

    ratio = float(target_faces) / max(1, faces.shape[0])
    target_vertices = max(4, int(verts.shape[0] * ratio))

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    bbox = maxs - mins
    volume = float(np.prod(bbox))
    if not np.isfinite(volume) or volume <= 0:
        return mesh, "cluster"

    voxel = (volume / max(1, target_vertices)) ** (1.0 / 3.0)
    if voxel <= 0 or not np.isfinite(voxel):
        return mesh, "cluster"

    grid = np.floor((verts - mins) / voxel).astype(np.int64)
    unique, unique_idx, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
    new_verts = verts[unique_idx]

    new_faces = inverse[faces]
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[valid]

    mesh_out = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=True)
    colors = _vertex_colors(mesh)
    if colors is not None and colors.shape[0] == verts.shape[0]:
        sums = np.zeros((unique.shape[0], 3), dtype=np.float32)
        counts = np.zeros((unique.shape[0], 1), dtype=np.float32)
        np.add.at(sums, inverse, colors)
        np.add.at(counts, inverse, 1.0)
        avg = sums / np.maximum(counts, 1.0)
        mesh_out.visual.vertex_colors = np.clip(avg * 255.0, 0, 255).astype(np.uint8)
    return mesh_out, "cluster"


def pick_target_faces(face_count: int, ratio: float, target_faces: int, min_faces: int) -> int:
    if target_faces > 0:
        target = target_faces
    else:
        ratio = float(ratio)
        ratio = max(1e-6, min(1.0, ratio))
        target = int(face_count * ratio)
    target = max(min_faces, target)
    target = min(face_count, target)
    return int(target)


def decimate_mesh(mesh, target_faces: int, method: str):
    tried = []
    if method == "auto":
        methods = ["open3d", "trimesh", "cluster"]
    else:
        methods = [method]

    for name in methods:
        try:
            if name == "open3d":
                return simplify_open3d(mesh, target_faces)
            if name == "trimesh":
                return simplify_trimesh(mesh, target_faces)
            if name == "cluster":
                return simplify_cluster(mesh, target_faces)
        except Exception as exc:
            tried.append(f"{name}: {exc}")
            continue

    raise RuntimeError("All decimation methods failed:\n" + "\n".join(tried))


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input, Path.cwd())
    if args.output is None:
        output_path = default_output_path(input_path)
    else:
        output_path = resolve_path(args.output, Path.cwd())

    if not input_path.exists():
        print(f"Missing input mesh: {input_path}")
        return 1

    try:
        mesh = load_mesh(input_path)
    except Exception as exc:
        print(f"Failed to load mesh: {exc}")
        return 1

    face_count = int(mesh.faces.shape[0])
    if face_count <= 0:
        print("Mesh has no faces. Skipping decimation.")
        return 0

    target_faces = pick_target_faces(face_count, args.ratio, args.target_faces, args.min_faces)
    if target_faces >= face_count:
        print(f"Target faces >= input faces ({target_faces} >= {face_count}). Copying mesh.")
        mesh.export(output_path)
        print(f"Saved: {output_path}")
        return 0

    try:
        mesh_out, method = decimate_mesh(mesh, target_faces, args.method)
    except Exception as exc:
        print(f"Decimation failed: {exc}")
        print("Falling back to copying the original mesh.")
        mesh.export(output_path)
        print(f"Saved: {output_path}")
        return 0

    mesh_out.export(output_path)
    print(
        f"Decimated mesh with {method}: "
        f"{face_count} -> {mesh_out.faces.shape[0]} faces"
    )
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
