#!/usr/bin/env bash
set -euo pipefail

# ./sam3d_metric_scale/run_full_pipeline.sh --image /home/vision/Sim2Real_Data_Augmentation_for_VLA/sam3d_metric_scale/datas/coffee_maker_sample.jpg
# ./sam3d_metric_scale/run_full_pipeline.sh --image /path/to/rgb.png --depth-image /path/to/depth.png --output-base outputs/demo

# 통합 파이프라인:
# 1) SAM2 UI로 마스크 생성
# 2) MoGe로 metric depth/포인트 추출
# 3) SAM3D로 PLY 생성
# 4) MoGe 포인트 기반으로 SAM3D 스케일 추정 (현재 비활성)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"

# 기본 conda env 이름
sam2_env="sam2"
sam3d_env="sam3d-objects"
moge_env="moge"
scale_env="teaserpp"

default_image="${repo_root}/sam2/notebooks/videos/bedroom/00031.jpg"
if [[ ! -f "${default_image}" && -f "${repo_root}/../sam2/notebooks/videos/bedroom/00031.jpg" ]]; then
  default_image="${repo_root}/../sam2/notebooks/videos/bedroom/00031.jpg"
fi
image_path="${default_image}"
depth_image_path=""
# real depth scale (e.g., 0.001 if depth is in millimeters)
depth_scale="0.001"

# 출력 베이스 디렉터리
output_base="${repo_root}/outputs"

default_sam3d_config="${repo_root}/sam-3d-objects/checkpoints/hf/pipeline.yaml"
if [[ ! -f "${default_sam3d_config}" && -f "${repo_root}/../sam-3d-objects/checkpoints/hf/pipeline.yaml" ]]; then
  default_sam3d_config="${repo_root}/../sam-3d-objects/checkpoints/hf/pipeline.yaml"
fi
sam3d_config="${default_sam3d_config}"

# SAM3D 옵션
sam3d_seed=42
sam3d_compile=0

# MoGe 옵션
moge_model="Ruicheng/moge-2-vitl-normal"
scale_method="bbox_diag"
min_pixels=100
cam_k_path="/home/vision/Sim2Real_Data_Augmentation_for_VLA/data/user_data_260123/user_data/move_box/cam_K.txt"

# Scale matching options
scale_algo="teaserpp"
scale_mode="scale_only"
fine_registration=0
icp_max_iters=1

# real depth filtering (stricter than MoGe)
real_border_margin=5
real_depth_mad=2.5
real_radius_mad=2.5
real_min_points=500

# TEASER++ 옵션(기본값)
teaser_noise_bound=0
teaser_nn_max_points=4000
teaser_max_correspondences=3000
teaser_gnc_factor=1.4
teaser_rot_max_iters=100
teaser_cbar2=1.0
teaser_iterations=2
teaser_correspondence="nn"
teaser_fpfh_voxel=0
teaser_fpfh_normal_radius=0
teaser_fpfh_feature_radius=0
teaser_icp_refine=1
teaser_icp_max_iters=100
teaser_icp_distance=0
teaser_estimate_scaling=1
scale_show_viz=0
scale_viz_method="open3d"

process_all=1

# disable open3d viz in headless environments
if [[ -z "${DISPLAY-}" ]]; then
  scale_show_viz=0
fi

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --image PATH              Input image path for SAM2 UI
  --depth-image PATH        Optional real depth image path (if provided, generate real_scale outputs)
  --depth-scale VAL         real depth scale (auto | numeric, default: auto)
  --output-base PATH        Base output directory (default: outputs)
  --latest                  Process latest mask only (default: all)
  --sam2-env NAME           Conda env for SAM2 (default: sam2)
  --sam3d-env NAME          Conda env for SAM3D (default: sam3d-objects)
  --scale-env NAME          Conda env for scale estimation (default: teaserpp)
  --sam3d-config PATH       SAM3D pipeline.yaml path
  --sam3d-seed INT          Seed for SAM3D inference
  --sam3d-compile           Enable compile flag for SAM3D
  --moge-env NAME           Conda env for MoGe (default: moge)
  --moge-model NAME         HF model id or local path
  --scale-method NAME       bbox_diag | bbox_max (default: bbox_diag)
  --min-pixels INT          Minimum valid pixels for scale
  --scale-algo NAME         icp | teaserpp (default: teaserpp)
  --scale-mode NAME         default | scale_only (default: scale_only)
  --fine-registration       After scale-only, run TEASER++ (no scale) once for R/t refine
  --icp-max-iters INT       ICP max iterations (default: 1)
  --scale-show-viz          Visualize alignment after scale matching
  --scale-viz-method NAME   open3d | matplotlib (default: open3d)
  --cam-k PATH              Camera intrinsics (3x3) for real depth backprojection
  -h, --help                Show this help
USAGE
}

resolve_path() {
  # 상대 경로를 레포 루트 기준으로 해석
  local input="$1"
  if [[ "${input}" = /* ]]; then
    echo "${input}"
  else
    echo "${repo_root}/${input}"
  fi
}

make_output_root() {
  # 동일 이름 폴더가 있을 경우 _001, _002 형태로 새 디렉터리 생성
  local base_dir="$1"
  local stem="$2"
  local root="${base_dir}/${stem}"
  local idx=1
  while [[ -e "${root}" ]]; do
    root="${base_dir}/${stem}_$(printf "%03d" "${idx}")"
    idx=$((idx + 1))
  done
  echo "${root}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      image_path="$2"
      shift 2
      ;;
    --depth-image)
      depth_image_path="$2"
      shift 2
      ;;
    --depth-scale)
      depth_scale="$2"
      shift 2
      ;;
    --output-base)
      output_base="$2"
      shift 2
      ;;
    --latest)
      process_all=0
      shift
      ;;
    --sam2-env)
      sam2_env="$2"
      shift 2
      ;;
    --sam3d-env)
      sam3d_env="$2"
      shift 2
      ;;
    --scale-env)
      scale_env="$2"
      shift 2
      ;;
    --sam3d-config)
      sam3d_config="$2"
      shift 2
      ;;
    --sam3d-seed)
      sam3d_seed="$2"
      shift 2
      ;;
    --sam3d-compile)
      sam3d_compile=1
      shift
      ;;
    --moge-env)
      moge_env="$2"
      shift 2
      ;;
    --moge-model)
      moge_model="$2"
      shift 2
      ;;
    --scale-method)
      scale_method="$2"
      shift 2
      ;;
    --min-pixels)
      min_pixels="$2"
      shift 2
      ;;
    --scale-algo)
      scale_algo="$2"
      shift 2
      ;;
    --scale-mode)
      scale_mode="$2"
      shift 2
      ;;
    --fine-registration)
      fine_registration=1
      shift 1
      ;;
    --icp-max-iters)
      icp_max_iters="$2"
      shift 2
      ;;
    --scale-show-viz)
      scale_show_viz=1
      shift 1
      ;;
    --scale-viz-method)
      scale_viz_method="$2"
      shift 2
      ;;
    --cam-k)
      cam_k_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi

# 최종 경로 정규화
image_path="$(resolve_path "${image_path}")"
output_base="$(resolve_path "${output_base}")"
sam3d_config="$(resolve_path "${sam3d_config}")"
if [[ -n "${depth_image_path}" ]]; then
  depth_image_path="$(resolve_path "${depth_image_path}")"
fi
if [[ -n "${cam_k_path}" ]]; then
  cam_k_path="$(resolve_path "${cam_k_path}")"
fi

image_stem="$(basename "${image_path}")"
image_stem="${image_stem%.*}"
output_root="$(make_output_root "${output_base}" "${image_stem}")"
mask_dir="${output_root}/sam2_masks"
sam3d_out_dir="${output_root}/sam3d"
moge_out_dir="${output_root}/moge_scale"
real_out_dir="${output_root}/real_scale"
scale_out_dir="${output_root}/sam3d_scale"

mkdir -p "${output_root}"
echo "Output root: ${output_root}"
# 원본 이미지는 결과 루트에 복사(재현성/추적 목적)
cp -n "${image_path}" "${output_root}/" 2>/dev/null || true
if [[ -n "${depth_image_path}" ]]; then
  cp -n "${depth_image_path}" "${output_root}/" 2>/dev/null || true
fi

# 1) SAM2 UI 실행(마스크 생성)
conda run -n "${sam2_env}" python "${repo_root}/src/image_point.py" \
  --image "${image_path}" \
  --output-dir "${mask_dir}"

mkdir -p "${sam3d_out_dir}" "${moge_out_dir}" "${scale_out_dir}"
if [[ -n "${depth_image_path}" ]]; then
  mkdir -p "${real_out_dir}"
fi

if [[ ${process_all} -eq 1 ]]; then
  mapfile -t masks < <(ls -1 "${mask_dir}/${image_stem}"_*.png 2>/dev/null || true)
else
  latest_mask="$(ls -t "${mask_dir}/${image_stem}"_*.png 2>/dev/null | head -n 1 || true)"
  masks=()
  if [[ -n "${latest_mask}" ]]; then
    masks=("${latest_mask}")
  fi
fi

if [[ ${#masks[@]} -eq 0 ]]; then
  echo "No masks found in ${mask_dir} for ${image_stem}_*.png"
  exit 1
fi

for mask_path in "${masks[@]}"; do
  mask_name="$(basename "${mask_path}")"
  mask_stem="${mask_name%.png}"
  output_path="${sam3d_out_dir}/${mask_stem}.ply"
  moge_npz="${moge_out_dir}/${image_stem}_${mask_stem}.npz"
  real_npz=""
  real_json=""
  real_ply=""
  scale_txt="${scale_out_dir}/${mask_stem}_scale.txt"
  scaled_ply="${scale_out_dir}/${mask_stem}_scaled.ply"

  # 2) MoGe 실행(마스크 영역 metric depth/points)
  conda run -n "${moge_env}" python "${repo_root}/src/moge_scale.py" \
    --image "${image_path}" \
    --mask "${mask_path}" \
    --model "${moge_model}" \
    --output-dir "${moge_out_dir}" \
    --scale-method "${scale_method}" \
    --min-pixels "${min_pixels}"

  if [[ -n "${cam_k_path}" ]]; then
    conda run -n "${moge_env}" python "${repo_root}/src/moge_scale.py" \
      --image "${image_path}" \
      --mask "${mask_path}" \
      --model "${moge_model}" \
      --output-dir "${moge_out_dir}" \
      --scale-method "${scale_method}" \
      --min-pixels "${min_pixels}" \
      --cam-k "${cam_k_path}" \
      --output-suffix "camk"
  fi

  # 2-1) Real depth 기반 outputs (옵션)
  if [[ -n "${depth_image_path}" ]]; then
    real_npz="${real_out_dir}/${image_stem}_${mask_stem}.npz"
    real_json="${real_out_dir}/${image_stem}_${mask_stem}.json"
    real_ply="${real_out_dir}/${image_stem}_${mask_stem}.ply"

    real_tmp_script="$(mktemp "${output_root}/real_depth_XXXX.py")"
    cat <<'PY' > "${real_tmp_script}"
import json
import os
from pathlib import Path

import cv2
import numpy as np

def mad_keep_mask(values: np.ndarray, thresh: float) -> np.ndarray:
    if thresh <= 0:
        return np.ones(values.shape[0], dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-8:
        return np.ones(values.shape[0], dtype=bool)
    z_score = 0.6745 * (values - median) / mad
    return np.abs(z_score) <= thresh

def build_filter_keep_mask(points: np.ndarray, ys: np.ndarray, xs: np.ndarray, shape: tuple[int, int],
                           border_margin: int, depth_mad: float, radius_mad: float) -> np.ndarray:
    keep = np.ones(points.shape[0], dtype=bool)
    if border_margin > 0 and ys.shape[0] == points.shape[0]:
        height, width = shape
        keep &= (ys >= border_margin) & (ys < height - border_margin) & (xs >= border_margin) & (xs < width - border_margin)
    filtered = points[keep]
    if filtered.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    if depth_mad > 0 or radius_mad > 0:
        sub_keep = np.ones(filtered.shape[0], dtype=bool)
        if depth_mad > 0:
            sub_keep &= mad_keep_mask(filtered[:, 2], depth_mad)
        if radius_mad > 0:
            center = np.median(filtered, axis=0)
            radius = np.linalg.norm(filtered - center, axis=1)
            sub_keep &= mad_keep_mask(radius, radius_mad)
        indices = np.where(keep)[0]
        keep_final = np.zeros(points.shape[0], dtype=bool)
        keep_final[indices[sub_keep]] = True
        return keep_final
    return keep

image_path = Path(os.environ["IMAGE_PATH"])
depth_path = Path(os.environ["DEPTH_PATH"])
mask_path = Path(os.environ["MASK_PATH"])
output_npz = Path(os.environ["OUTPUT_NPZ"])
output_json = Path(os.environ["OUTPUT_JSON"])
output_ply = Path(os.environ["OUTPUT_PLY"])
output_full_ply = Path(os.environ["OUTPUT_FULL_PLY"])
cam_k_path = Path(os.environ.get("CAM_K_PATH", ""))
min_pixels = int(os.environ.get("MIN_PIXELS", "100"))

border_margin = int(os.environ.get("REAL_BORDER_MARGIN", "5"))
depth_mad = float(os.environ.get("REAL_DEPTH_MAD", "2.5"))
radius_mad = float(os.environ.get("REAL_RADIUS_MAD", "2.5"))
min_points = int(os.environ.get("REAL_MIN_POINTS", "500"))

if not depth_path.exists():
    raise SystemExit(f"Missing depth image: {depth_path}")
if not mask_path.exists():
    raise SystemExit(f"Missing mask: {mask_path}")
if cam_k_path and not cam_k_path.exists():
    raise SystemExit(f"Missing camera intrinsics: {cam_k_path}")

depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
if depth is None:
    raise SystemExit(f"Failed to read depth image: {depth_path}")
if depth.ndim == 3:
    depth = depth[:, :, 0]
depth = depth.astype(np.float32)

depth_scale_raw = os.environ.get("DEPTH_SCALE", "auto")
if depth_scale_raw == "auto":
    # heuristic: uint16 or large values -> millimeters
    if depth.dtype.kind in ("u", "i") and depth.max() > 50:
        depth_scale_value = 0.001
    elif depth.max() > 20 and depth.max() < 10000:
        depth_scale_value = 0.001
    else:
        depth_scale_value = 1.0
else:
    depth_scale_value = float(depth_scale_raw)
depth = depth * depth_scale_value

if cam_k_path:
    k_mat = np.loadtxt(str(cam_k_path)).astype(np.float32)
    if k_mat.size == 4:
        fx, fy, cx, cy = [float(v) for v in k_mat.ravel()]
    else:
        k_mat = k_mat.reshape(3, 3)
        fx = float(k_mat[0, 0])
        fy = float(k_mat[1, 1])
        cx = float(k_mat[0, 2])
        cy = float(k_mat[1, 2])
else:
    fx = fy = None
    cx = cy = None

mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
if mask is None:
    raise SystemExit(f"Failed to read mask: {mask_path}")
if mask.ndim == 3:
    mask = mask[:, :, -1]
mask = mask > 0

if mask.shape != depth.shape:
    mask = cv2.resize(
        mask.astype(np.uint8),
        (depth.shape[1], depth.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

valid = mask & np.isfinite(depth) & (depth > 0)
if valid.sum() < min_pixels:
    raise SystemExit(f"Not enough valid pixels: {int(valid.sum())}")

ys, xs = np.where(valid)
depth_masked = depth[valid]

height, width = depth.shape
if fx is None or fy is None:
    fx = fy = float(max(height, width))
if cx is None or cy is None:
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
z = depth_masked
x = (xs - cx) * z / fx
y = (ys - cy) * z / fy
points = np.stack([x, y, z], axis=1).astype(np.float32)

valid_full = np.isfinite(depth) & (depth > 0)
ys_full, xs_full = np.where(valid_full)
z_full = depth[valid_full]
x_full = (xs_full - cx) * z_full / fx
y_full = (ys_full - cy) * z_full / fy
points_full = np.stack([x_full, y_full, z_full], axis=1).astype(np.float32)

keep = build_filter_keep_mask(points, ys, xs, depth.shape, border_margin, depth_mad, radius_mad)
filter_applied = int(keep.sum()) >= min_points
if not filter_applied:
    keep = np.ones(points.shape[0], dtype=bool)
filtered_points = points[keep]
filtered_depth = depth_masked[keep]
filtered_valid = valid.copy()
if keep.shape[0] == ys.shape[0]:
    filtered_valid[ys[~keep], xs[~keep]] = False
if filtered_points.size == 0:
    filtered_points = points
    filtered_depth = depth_masked
    filtered_valid = valid
    filter_applied = False

points = filtered_points
depth_masked = filtered_depth

def compute_scale(points_np: np.ndarray) -> dict:
    mins = points_np.min(axis=0)
    maxs = points_np.max(axis=0)
    size = maxs - mins
    diag = float(np.linalg.norm(size))
    max_dim = float(size.max())
    return {
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
        "bbox_size": size.tolist(),
        "bbox_diag": diag,
        "bbox_max_dim": max_dim,
        "scale_method": "bbox_diag",
    }

stats = {
    "image": str(image_path),
    "depth_image": str(depth_path),
    "mask": str(mask_path),
    "model": "real_depth",
    "depth_scale": float(depth_scale_value),
    "camera_fx": float(fx),
    "camera_fy": float(fy),
    "camera_cx": float(cx),
    "camera_cy": float(cy),
    "points_count_raw": int(points.shape[0]),
    "points_count": int(points.shape[0]),
    "filter_applied": bool(filter_applied),
    "filter_border_margin": int(border_margin),
    "filter_depth_mad": float(depth_mad),
    "filter_radius_mad": float(radius_mad),
    "filter_min_points": int(min_points),
    "depth_mean": float(depth_masked.mean()),
    "depth_median": float(np.median(depth_masked)),
    "depth_min": float(depth_masked.min()),
    "depth_max": float(depth_masked.max()),
}
stats.update(compute_scale(points))

output_npz.parent.mkdir(parents=True, exist_ok=True)
output_json.parent.mkdir(parents=True, exist_ok=True)
output_ply.parent.mkdir(parents=True, exist_ok=True)

with output_json.open("w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

np.savez_compressed(
    output_npz,
    mask=mask.astype(np.uint8),
    valid_mask=filtered_valid.astype(np.uint8),
    points_masked=points.astype(np.float32),
    depth_masked=depth_masked.astype(np.float32),
    valid_full=valid_full.astype(np.uint8),
    points_full=points_full.astype(np.float32),
    depth_full=depth.astype(np.float32),
)

with output_ply.open("w", encoding="utf-8") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {points.shape[0]}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")
    for px, py, pz in points:
        f.write(f"{px} {py} {pz}\n")

with output_full_ply.open("w", encoding="utf-8") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {points_full.shape[0]}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")
    for px, py, pz in points_full:
        f.write(f"{px} {py} {pz}\n")

print(f"Saved real depth stats: {output_json}")
print(f"Saved real depth npz: {output_npz}")
print(f"Saved real depth ply: {output_ply}")
print(f"Saved real depth full ply: {output_full_ply}")
PY
    IMAGE_PATH="${image_path}" \
    DEPTH_PATH="${depth_image_path}" \
    MASK_PATH="${mask_path}" \
    OUTPUT_NPZ="${real_npz}" \
    OUTPUT_JSON="${real_json}" \
    OUTPUT_PLY="${real_ply}" \
    OUTPUT_FULL_PLY="${real_out_dir}/${image_stem}_${mask_stem}_full.ply" \
    CAM_K_PATH="${cam_k_path}" \
    MIN_PIXELS="${min_pixels}" \
    DEPTH_SCALE="${depth_scale}" \
    REAL_BORDER_MARGIN="${real_border_margin}" \
    REAL_DEPTH_MAD="${real_depth_mad}" \
    REAL_RADIUS_MAD="${real_radius_mad}" \
    REAL_MIN_POINTS="${real_min_points}" \
    conda run -n "${moge_env}" python "${real_tmp_script}"
    rm -f "${real_tmp_script}"

  fi

  if [[ -n "${depth_image_path}" ]]; then
    if [[ ! -f "${real_npz}" ]]; then
      echo "Real depth outputs missing: ${real_npz}. Falling back to MoGe for scaling."
      real_npz=""
    fi
  fi

  # 3) SAM3D 실행(PLY 생성)
  compile_flag=""
  if [[ ${sam3d_compile} -eq 1 ]]; then
    compile_flag="--compile"
  fi

  sam3d_pointmap_flags=()
  if [[ -n "${depth_image_path}" && -n "${cam_k_path}" ]]; then
    sam3d_pointmap_flags+=(
      "--pointmap-from-depth"
      "--depth-image" "${depth_image_path}"
      "--cam-k" "${cam_k_path}"
      "--depth-scale" "${depth_scale}"
    )
  fi

  conda run -n "${sam3d_env}" python "${repo_root}/src/sam3d_export.py" \
    --image "${image_path}" \
    --mask "${mask_path}" \
    --sam3d-config "${sam3d_config}" \
    --output "${output_path}" \
    --seed "${sam3d_seed}" \
    --pose-rot-transpose \
    "${sam3d_pointmap_flags[@]}" \
    ${compile_flag}

  # 4) 스케일 정합 단계는 본 실험에서 제외
  echo "Skipping sam3d_scale (scale matching disabled for pointmap-depth experiment)."

done
