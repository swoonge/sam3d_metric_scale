"""TEASER++ 기반 스케일/정합 추정."""

from __future__ import annotations

import numpy as np

from sam3d_scale_utils import apply_similarity, nearest_neighbors, sample_points

FPFH_AUTO_RATIO = 0.01
ROPS_AUTO_RATIO = 0.01
ROPS_DEFAULT_BINS = 5
ROPS_DEFAULT_ROTATIONS = 3


def _auto_fpfh_params(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, float]:
    all_points = np.vstack([src, dst])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    voxel = max(1e-6, diag * FPFH_AUTO_RATIO)
    normal_radius = voxel * 2.0
    feature_radius = voxel * 5.0
    return voxel, normal_radius, feature_radius


def _resolve_fpfh_params(
    src: np.ndarray,
    dst: np.ndarray,
    voxel_size: float,
    normal_radius: float,
    feature_radius: float,
) -> tuple[float, float, float]:
    if voxel_size <= 0 or normal_radius <= 0 or feature_radius <= 0:
        auto_voxel, auto_normal, auto_feature = _auto_fpfh_params(src, dst)
        if voxel_size <= 0:
            voxel_size = auto_voxel
        if normal_radius <= 0:
            normal_radius = auto_normal
        if feature_radius <= 0:
            feature_radius = auto_feature
    return voxel_size, normal_radius, feature_radius


def _auto_rops_params(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, float]:
    all_points = np.vstack([src, dst])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    voxel = max(1e-6, diag * ROPS_AUTO_RATIO)
    support_radius = voxel * 5.0
    mesh_radius = voxel * 2.5
    return voxel, support_radius, mesh_radius


def _resolve_rops_params(
    src: np.ndarray,
    dst: np.ndarray,
    voxel_size: float,
    support_radius: float,
    mesh_radius: float,
) -> tuple[float, float, float]:
    if voxel_size <= 0 or support_radius <= 0 or mesh_radius <= 0:
        auto_voxel, auto_support, auto_mesh = _auto_rops_params(src, dst)
        if voxel_size <= 0:
            voxel_size = auto_voxel
        if support_radius <= 0:
            support_radius = auto_support
        if mesh_radius <= 0:
            mesh_radius = auto_mesh
    return voxel_size, support_radius, mesh_radius


def _knn_features(
    src_desc: np.ndarray, dst_desc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(dst_desc)
        dists, idx = tree.query(src_desc, k=1, workers=-1)
        return idx.astype(np.int64), dists.astype(np.float32)
    except Exception:
        try:
            from sklearn.neighbors import KDTree

            tree = KDTree(dst_desc)
            dists, idx = tree.query(src_desc, k=1)
            return idx[:, 0].astype(np.int64), dists[:, 0].astype(np.float32)
        except Exception:
            return nearest_neighbors(src_desc, dst_desc, chunk=256)


def _build_correspondences_fpfh(
    src: np.ndarray,
    dst: np.ndarray,
    nn_max_points: int,
    max_correspondences: int,
    voxel_size: float,
    normal_radius: float,
    feature_radius: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required for FPFH correspondences.\n"
            "Install with: conda install -c conda-forge open3d"
        ) from exc

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src)
    dst_pcd = o3d.geometry.PointCloud()
    dst_pcd.points = o3d.utility.Vector3dVector(dst)

    if voxel_size > 0:
        src_pcd = src_pcd.voxel_down_sample(voxel_size)
        dst_pcd = dst_pcd.voxel_down_sample(voxel_size)

    src_points = np.asarray(src_pcd.points)
    dst_points = np.asarray(dst_pcd.points)

    if nn_max_points > 0 and src_points.shape[0] > nn_max_points:
        idx = np.random.default_rng(seed).choice(src_points.shape[0], nn_max_points, replace=False)
        src_points = src_points[idx]
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_points)
    if nn_max_points > 0 and dst_points.shape[0] > nn_max_points:
        idx = np.random.default_rng(seed + 1).choice(dst_points.shape[0], nn_max_points, replace=False)
        dst_points = dst_points[idx]
        dst_pcd = o3d.geometry.PointCloud()
        dst_pcd.points = o3d.utility.Vector3dVector(dst_points)

    src_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    dst_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )

    src_feat = o3d.pipelines.registration.compute_fpfh_feature(
        src_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )
    dst_feat = o3d.pipelines.registration.compute_fpfh_feature(
        dst_pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )

    src_desc = np.asarray(src_feat.data, dtype=np.float32).T
    dst_desc = np.asarray(dst_feat.data, dtype=np.float32).T
    if src_desc.size == 0 or dst_desc.size == 0:
        raise RuntimeError("FPFH produced empty descriptors.")

    src_to_dst, src_dists = _knn_features(src_desc, dst_desc)
    dst_to_src, _ = _knn_features(dst_desc, src_desc)
    src_indices = np.arange(src_desc.shape[0])
    mutual = src_indices == dst_to_src[src_to_dst]
    src_idx = src_indices[mutual]
    dst_idx = src_to_dst[src_idx]

    if src_idx.size == 0:
        raise RuntimeError("No mutual FPFH correspondences found.")

    feat_dist = src_dists[src_idx]
    if max_correspondences > 0 and feat_dist.shape[0] > max_correspondences:
        keep_n = max(3, max_correspondences)
        keep_idx = np.argpartition(feat_dist, keep_n - 1)[:keep_n]
        src_idx = src_idx[keep_idx]
        dst_idx = dst_idx[keep_idx]

    return src_points[src_idx], dst_points[dst_idx]


def _build_correspondences_pcl_rops(
    src: np.ndarray,
    dst: np.ndarray,
    nn_max_points: int,
    max_correspondences: int,
    voxel_size: float,
    support_radius: float,
    mesh_radius: float,
    rops_bins: int,
    rops_rotations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from pclpy import pcl
    except ImportError as exc:
        raise RuntimeError(
            "pclpy is required for PCL ROPS correspondences.\n"
            "Install with: conda install -c conda-forge pclpy"
        ) from exc

    rng = np.random.default_rng(seed)

    def _prepare_cloud(points: np.ndarray) -> tuple["pcl.PointCloud.PointXYZ", np.ndarray]:
        cloud = pcl.PointCloud.PointXYZ()
        cloud.from_array(points.astype(np.float32))
        if voxel_size > 0:
            vg = pcl.filters.VoxelGrid.PointXYZ()
            vg.setInputCloud(cloud)
            vg.setLeafSize(voxel_size, voxel_size, voxel_size)
            filtered = pcl.PointCloud.PointXYZ()
            vg.filter(filtered)
            cloud = filtered
        pts = cloud.to_array()
        if nn_max_points > 0 and pts.shape[0] > nn_max_points:
            idx = rng.choice(pts.shape[0], nn_max_points, replace=False)
            pts = pts[idx]
            cloud = pcl.PointCloud.PointXYZ()
            cloud.from_array(pts.astype(np.float32))
        return cloud, pts

    def _build_mesh(cloud: "pcl.PointCloud.PointXYZ") -> "pcl.PolygonMesh":
        normals = pcl.PointCloud.Normal()
        ne = pcl.features.NormalEstimationOMP.PointXYZ_Normal()
        ne.setInputCloud(cloud)
        ne.setRadiusSearch(mesh_radius)
        tree = pcl.search.KdTree.PointXYZ()
        ne.setSearchMethod(tree)
        ne.compute(normals)

        cloud_normals = pcl.PointCloud.PointNormal()
        try:
            pcl.concatenateFields(cloud, normals, cloud_normals)
        except Exception:
            pcl.concatenate_fields(cloud, normals, cloud_normals)

        gp3 = pcl.surface.GreedyProjectionTriangulation.PointNormal()
        gp3.setSearchRadius(mesh_radius)
        gp3.setMu(2.5)
        gp3.setMaximumNearestNeighbors(100)
        gp3.setMaximumSurfaceAngle(np.pi / 4.0)
        gp3.setMinimumAngle(np.pi / 18.0)
        gp3.setMaximumAngle(2.0 * np.pi / 3.0)
        gp3.setNormalConsistency(False)
        gp3.setInputCloud(cloud_normals)
        tree_norm = pcl.search.KdTree.PointNormal()
        gp3.setSearchMethod(tree_norm)
        mesh = pcl.PolygonMesh()
        gp3.reconstruct(mesh)
        if not mesh.polygons:
            raise RuntimeError("PCL ROPS mesh reconstruction failed (no polygons).")
        return mesh

    def _compute_rops(
        cloud: "pcl.PointCloud.PointXYZ", mesh: "pcl.PolygonMesh"
    ) -> np.ndarray:
        rops = pcl.features.ROPSEstimation.PointXYZ_Histogram135()
        rops.setInputCloud(cloud)
        rops.setSearchSurface(cloud)
        rops.setTriangles(mesh.polygons)
        rops.setRadiusSearch(support_radius)
        rops.setNumberOfPartitionBins(int(rops_bins))
        rops.setNumberOfRotations(int(rops_rotations))
        rops.setSupportRadius(support_radius)
        descriptors = pcl.PointCloud.Histogram135()
        rops.compute(descriptors)
        desc = descriptors.to_array().astype(np.float32)
        if desc.size == 0:
            raise RuntimeError("PCL ROPS produced empty descriptors.")
        return desc

    src_cloud, src_points = _prepare_cloud(src)
    dst_cloud, dst_points = _prepare_cloud(dst)

    src_mesh = _build_mesh(src_cloud)
    dst_mesh = _build_mesh(dst_cloud)

    src_desc = _compute_rops(src_cloud, src_mesh)
    dst_desc = _compute_rops(dst_cloud, dst_mesh)

    count = min(src_desc.shape[0], src_points.shape[0])
    src_desc = src_desc[:count]
    src_points = src_points[:count]
    count = min(dst_desc.shape[0], dst_points.shape[0])
    dst_desc = dst_desc[:count]
    dst_points = dst_points[:count]

    src_to_dst, src_dists = _knn_features(src_desc, dst_desc)
    dst_to_src, _ = _knn_features(dst_desc, src_desc)
    src_indices = np.arange(src_desc.shape[0])
    mutual = src_indices == dst_to_src[src_to_dst]
    src_idx = src_indices[mutual]
    dst_idx = src_to_dst[src_idx]

    if src_idx.size == 0:
        raise RuntimeError("No mutual PCL ROPS correspondences found.")

    feat_dist = src_dists[src_idx]
    if max_correspondences > 0 and feat_dist.shape[0] > max_correspondences:
        keep_n = max(3, max_correspondences)
        keep_idx = np.argpartition(feat_dist, keep_n - 1)[:keep_n]
        src_idx = src_idx[keep_idx]
        dst_idx = dst_idx[keep_idx]

    return src_points[src_idx], dst_points[dst_idx]


def estimate_scale(
    src: np.ndarray,
    dst: np.ndarray,
    nn_max_points: int = 8000,
    max_correspondences: int = 6000,
    noise_bound: float = 0.0,
    cbar2: float = 1.0,
    gnc_factor: float = 1.4,
    rot_max_iters: int = 100,
    estimate_scaling: bool = False,
    iterations: int = 1,
    correspondence: str = "auto",
    fpfh_voxel: float = 0.0,
    fpfh_normal_radius: float = 0.0,
    fpfh_feature_radius: float = 0.0,
    rops_voxel: float = 0.0,
    rops_support_radius: float = 0.0,
    rops_mesh_radius: float = 0.0,
    rops_bins: int = ROPS_DEFAULT_BINS,
    rops_rotations: int = ROPS_DEFAULT_ROTATIONS,
    rotation_cost_threshold: float = 1e-12,
    icp_refine: bool = False,
    icp_max_iters: int = 100,
    icp_distance: float = 0.0,
    seed: int = 0,
) -> dict:
    """TEASER++로 스케일/회전/이동을 추정."""
    try:
        import teaserpp_python
    except ImportError as exc:
        raise RuntimeError(
            "teaserpp_python is not installed. Install with:\n"
            "  pip install teaserpp-python\n"
            "or build TEASER++ from source."
        ) from exc

    mode = correspondence.lower().strip()
    if mode in ("", "auto"):
        mode = "fpfh"
    if mode not in ("fpfh", "pcl_rops", "rops"):
        raise RuntimeError(
            "Unsupported correspondence mode. "
            "Use --teaser-correspondence fpfh or pcl_rops."
        )

    iterations = max(1, int(iterations))
    scale_total = 1.0
    r = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    max_clique = None
    src_corr = None
    dst_corr = None
    last_scale_before = 1.0
    last_noise_bound = float(noise_bound)
    last_voxel = float(fpfh_voxel) if fpfh_voxel > 0 else 0.0
    last_support_radius = float(rops_support_radius) if rops_support_radius > 0 else 0.0
    last_mesh_radius = float(rops_mesh_radius) if rops_mesh_radius > 0 else 0.0

    for it in range(iterations):
        last_scale_before = scale_total
        src_scaled = src * scale_total
        if mode == "fpfh":
            voxel_size, normal_radius, feature_radius = _resolve_fpfh_params(
                src_scaled,
                dst,
                voxel_size=fpfh_voxel,
                normal_radius=fpfh_normal_radius,
                feature_radius=fpfh_feature_radius,
            )
            last_voxel = float(voxel_size)
            src_corr, dst_corr = _build_correspondences_fpfh(
                src_scaled,
                dst,
                nn_max_points=nn_max_points,
                max_correspondences=max_correspondences,
                voxel_size=voxel_size,
                normal_radius=normal_radius,
                feature_radius=feature_radius,
                seed=seed + it,
            )
        else:
            voxel_size, support_radius, mesh_radius = _resolve_rops_params(
                src_scaled,
                dst,
                voxel_size=rops_voxel,
                support_radius=rops_support_radius,
                mesh_radius=rops_mesh_radius,
            )
            last_voxel = float(voxel_size)
            last_support_radius = float(support_radius)
            last_mesh_radius = float(mesh_radius)
            src_corr, dst_corr = _build_correspondences_pcl_rops(
                src_scaled,
                dst,
                nn_max_points=nn_max_points,
                max_correspondences=max_correspondences,
                voxel_size=voxel_size,
                support_radius=support_radius,
                mesh_radius=mesh_radius,
                rops_bins=rops_bins,
                rops_rotations=rops_rotations,
                seed=seed + it,
            )

        if noise_bound <= 0.0:
            last_noise_bound = max(1e-6, float(voxel_size))
        else:
            last_noise_bound = float(noise_bound)

        params = teaserpp_python.RobustRegistrationSolver.Params()
        params.cbar2 = float(cbar2)
        params.noise_bound = float(last_noise_bound)
        params.estimate_scaling = bool(estimate_scaling)
        if hasattr(params, "inlier_selection_mode"):
            params.inlier_selection_mode = (
                teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
            )
        if hasattr(params, "rotation_tim_graph"):
            params.rotation_tim_graph = (
                teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
            )
        params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        params.rotation_gnc_factor = float(gnc_factor)
        params.rotation_max_iterations = int(rot_max_iters)
        params.rotation_cost_threshold = float(rotation_cost_threshold)

        solver = teaserpp_python.RobustRegistrationSolver(params)
        solver.solve(src_corr.T.astype(np.float64), dst_corr.T.astype(np.float64))
        solution = solver.getSolution()
        try:
            max_clique = solver.getTranslationInliersMap()
        except Exception:
            max_clique = None

        r = np.asarray(solution.rotation, dtype=np.float32)
        t = np.asarray(solution.translation, dtype=np.float32).reshape(3)
        if estimate_scaling:
            scale_total *= float(solution.scale)

    if icp_refine:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "open3d is required for ICP refinement.\n"
                "Install with: conda install -c conda-forge open3d"
            ) from exc

        src_scaled = src * scale_total
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_scaled)
        dst_pcd = o3d.geometry.PointCloud()
        dst_pcd.points = o3d.utility.Vector3dVector(dst)
        init = np.eye(4, dtype=np.float64)
        init[:3, :3] = r.astype(np.float64)
        init[:3, 3] = t.astype(np.float64)
        thresh = float(icp_distance) if icp_distance > 0.0 else float(last_noise_bound)
        thresh = max(1e-6, thresh)
        result = o3d.pipelines.registration.registration_icp(
            src_pcd,
            dst_pcd,
            thresh,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iters),
        )
        r = result.transformation[:3, :3].astype(np.float32)
        t = result.transformation[:3, 3].astype(np.float32)

    # 시각화용: 잔차 + (있다면) TEASER++ inlier 기준을 함께 사용한다.
    if src_corr is None or dst_corr is None:
        raise RuntimeError("Failed to build TEASER++ correspondences.")

    if last_scale_before <= 0:
        last_scale_before = 1.0
    matched_src = src_corr / last_scale_before
    transformed = apply_similarity(matched_src, scale_total, r, t)
    residual = np.linalg.norm(transformed - dst_corr, axis=1)
    if residual.size:
        if noise_bound > 0.0:
            thresh = max(1e-6, noise_bound * 2.0)
        else:
            median = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median)))
            thresh = max(1e-6, median + 2.0 * mad)
        match_mask = residual <= thresh
    else:
        match_mask = None

    inlier_indices = None
    if max_clique is not None:
        clique_mask = np.zeros(src_corr.shape[0], dtype=bool)
        idx = np.asarray(list(max_clique), dtype=int)
        idx = idx[(idx >= 0) & (idx < clique_mask.size)]
        if idx.size:
            clique_mask[idx] = True
            if match_mask is not None:
                combined = match_mask & clique_mask
            else:
                combined = clique_mask
            inlier_indices = np.nonzero(combined)[0].tolist()
            if match_mask is not None:
                match_mask = combined

    return {
        "scale": float(scale_total),
        "r": r,
        "t": t,
        "matched_src": matched_src,
        "matched_dst": dst_corr,
        "inlier_indices": inlier_indices,
        "match_mask": match_mask,
        "metrics": {
            "noise_bound": float(last_noise_bound),
            "corr_count": int(src_corr.shape[0]),
            "corr_mode": mode,
            "iterations": int(iterations),
            "voxel_size": float(last_voxel),
            "rops_support_radius": float(last_support_radius),
            "rops_mesh_radius": float(last_mesh_radius),
            "rops_bins": int(rops_bins),
            "rops_rotations": int(rops_rotations),
            "icp_refine": bool(icp_refine),
        },
    }
