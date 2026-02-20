from __future__ import annotations

import pytest

trimesh = pytest.importorskip("trimesh")

from mesh_decimate import load_mesh


def test_load_mesh_trimesh_api_compat(tmp_path):
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh_path = tmp_path / "box.ply"
    mesh.export(mesh_path)

    loaded = load_mesh(mesh_path)

    assert loaded.faces.shape[0] > 0
    assert loaded.vertices.shape[0] > 0
