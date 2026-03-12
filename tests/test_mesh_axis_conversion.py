from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")

from sam3d_export import export_mesh_world_z_up, save_pose_transformed_mesh


def _make_anisotropic_box():
    return trimesh.creation.box(extents=(1.0, 2.0, 3.0))


def test_export_mesh_world_z_up_rotates_obj_x_ccw_90(tmp_path):
    mesh = _make_anisotropic_box()
    out_path = tmp_path / "axis_fixed.obj"
    export_mesh_world_z_up(mesh, out_path)

    loaded = trimesh.load(out_path, force="mesh")
    extents = loaded.bounding_box.extents

    assert np.allclose(extents, np.array([1.0, 3.0, 2.0]), atol=1e-5)


def test_save_pose_transformed_mesh_rotates_obj_x_ccw_90(tmp_path):
    mesh = _make_anisotropic_box()
    out_path = tmp_path / "pose_axis_fixed.obj"
    save_pose_transformed_mesh(
        mesh=mesh,
        output_path=out_path,
        scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
    )

    loaded = trimesh.load(out_path, force="mesh")
    extents = loaded.bounding_box.extents

    assert np.allclose(extents, np.array([1.0, 3.0, 2.0]), atol=1e-5)


def test_export_mesh_world_z_up_rotates_glb_x_ccw_90(tmp_path):
    mesh = _make_anisotropic_box()
    out_path = tmp_path / "no_fix.glb"
    export_mesh_world_z_up(mesh, out_path)

    loaded = trimesh.load(out_path, force="mesh")
    extents = loaded.bounding_box.extents

    assert np.allclose(extents, np.array([1.0, 3.0, 2.0]), atol=1e-5)


def test_export_mesh_world_z_up_rotates_obj_markers_x_ccw_90(tmp_path):
    # +Y marker should map to +Z after x-axis +90deg rotation.
    verts = np.array(
        [
            [0.0, 1.0, 0.0],  # +Y marker
            [0.0, 0.0, 2.0],  # +Z marker
            [1.0, 0.0, 0.0],  # +X marker
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out_path = tmp_path / "neg90_check.obj"
    export_mesh_world_z_up(mesh, out_path)

    loaded = trimesh.load(out_path, force="mesh")
    loaded_verts = loaded.vertices
    target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    assert np.min(np.linalg.norm(loaded_verts - target, axis=1)) < 1e-5
