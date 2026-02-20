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
scale_env=""
scale_env_set=0

default_image="${repo_root}/sam2/notebooks/videos/bedroom/00031.jpg"
if [[ ! -f "${default_image}" && -f "${repo_root}/../sam2/notebooks/videos/bedroom/00031.jpg" ]]; then
  default_image="${repo_root}/../sam2/notebooks/videos/bedroom/00031.jpg"
fi
image_path="${default_image}"
depth_image_path=""
# real depth scale (e.g., 0.001 if depth is in millimeters)
depth_scale="auto"

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
cam_k_path=""
run_moge=0

# Scale matching options
scale_algo="icp"
scale_mode="scale_only"
fine_registration=0
estimate_scale=0
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
process_all=1

# Mesh decimation options (scaled mesh size reduction)
mesh_decimate=1
mesh_decimate_ratio=0.02
mesh_target_faces=20000
mesh_decimate_method="auto"
mesh_decimate_min_faces=200

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
  --scale-env NAME          Conda env for scale estimation (default: sam3d-objects for icp, teaserpp for teaserpp)
  --sam3d-config PATH       SAM3D pipeline.yaml path
  --sam3d-seed INT          Seed for SAM3D inference
  --sam3d-compile           Enable compile flag for SAM3D
  --moge-env NAME           Conda env for MoGe (default: moge)
  --run-moge                Run MoGe depth inference (default: off)
  --moge-model NAME         HF model id or local path
  --scale-method NAME       bbox_diag | bbox_max (default: bbox_diag)
  --min-pixels INT          Minimum valid pixels for scale
  --scale-algo NAME         icp | teaserpp (default: icp)
  --scale-mode NAME         default | scale_only (default: scale_only)
  --estimate-scale          Enable scale estimation against target points
  --fine-registration       After scale-only, run ICP once for R/t refine
  --icp-max-iters INT       ICP max iterations (default: 1)
  --cam-k PATH              Camera intrinsics (3x3) for real depth backprojection
  --mesh-decimate           Enable mesh decimation after scaling (default: on)
  --no-mesh-decimate        Disable mesh decimation
  --mesh-decimate-ratio VAL Target face ratio (0<r<=1, default: 0.02)
  --mesh-target-faces INT   Target number of faces (overrides ratio, default: 20000)
  --mesh-decimate-method M  auto | open3d | trimesh | cluster (default: auto)
  --mesh-min-faces INT      Minimum faces to keep (default: 200)
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
      scale_env_set=1
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
    --run-moge)
      run_moge=1
      shift
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
    --estimate-scale)
      estimate_scale=1
      shift 1
      ;;
    --fine-registration)
      fine_registration=1
      shift 1
      ;;
    --icp-max-iters)
      icp_max_iters="$2"
      shift 2
      ;;
    --cam-k)
      cam_k_path="$2"
      shift 2
      ;;
    --mesh-decimate)
      mesh_decimate=1
      shift
      ;;
    --no-mesh-decimate)
      mesh_decimate=0
      shift
      ;;
    --mesh-decimate-ratio)
      mesh_decimate_ratio="$2"
      shift 2
      ;;
    --mesh-target-faces)
      mesh_target_faces="$2"
      shift 2
      ;;
    --mesh-decimate-method)
      mesh_decimate_method="$2"
      shift 2
      ;;
    --mesh-min-faces)
      mesh_decimate_min_faces="$2"
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

# scale env 기본값 결정(알고리즘별)
if [[ ${scale_env_set} -eq 0 ]]; then
  if [[ "${scale_algo}" == "teaserpp" ]]; then
    scale_env="teaserpp"
  else
    scale_env="${sam3d_env}"
  fi
fi

# 최종 경로 정규화
image_path="$(resolve_path "${image_path}")"
output_base="$(resolve_path "${output_base}")"
sam3d_config="$(resolve_path "${sam3d_config}")"
if [[ -n "${depth_image_path}" ]]; then
  if [[ -z "${cam_k_path}" ]]; then
    echo "--depth-image requires --cam-k (3x3 intrinsics txt)."
    exit 1
  fi
  depth_image_path="$(resolve_path "${depth_image_path}")"
fi
if [[ -n "${cam_k_path}" ]]; then
  cam_k_path="$(resolve_path "${cam_k_path}")"
fi
if [[ -n "${cam_k_path}" && ! -f "${cam_k_path}" ]]; then
  echo "Missing camera intrinsics: ${cam_k_path}"
  exit 1
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

mkdir -p "${sam3d_out_dir}" "${scale_out_dir}"
if [[ ${run_moge} -eq 1 ]]; then
  mkdir -p "${moge_out_dir}"
fi
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
  if [[ ${run_moge} -eq 1 ]]; then
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
  else
    echo "Skipping MoGe (use --run-moge to enable)."
  fi

  # 2-1) Real depth 기반 outputs (옵션)
  if [[ -n "${depth_image_path}" ]]; then
    real_npz="${real_out_dir}/${image_stem}_${mask_stem}.npz"
    real_json="${real_out_dir}/${image_stem}_${mask_stem}.json"
    real_ply="${real_out_dir}/${image_stem}_${mask_stem}.ply"
    conda run -n "${sam3d_env}" python "${repo_root}/src/real_depth_scale.py" \
      --image "${image_path}" \
      --depth-image "${depth_image_path}" \
      --mask "${mask_path}" \
      --output-npz "${real_npz}" \
      --output-json "${real_json}" \
      --output-ply "${real_ply}" \
      --output-full-ply "${real_out_dir}/${image_stem}_${mask_stem}_full.ply" \
      --cam-k "${cam_k_path}" \
      --min-pixels "${min_pixels}" \
      --depth-scale "${depth_scale}" \
      --border-margin "${real_border_margin}" \
      --depth-mad "${real_depth_mad}" \
      --radius-mad "${real_radius_mad}" \
      --min-points "${real_min_points}"

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

  # 4) 스케일 정합 (real depth 우선, 없으면 MoGe)
  target_npz=""
  if [[ -n "${real_npz}" && -f "${real_npz}" ]]; then
    target_npz="${real_npz}"
  elif [[ ${run_moge} -eq 1 && -f "${moge_npz}" ]]; then
    target_npz="${moge_npz}"
  fi

  if [[ -n "${target_npz}" ]]; then
    sam3d_pose_ply="${sam3d_out_dir}/${mask_stem}_pose.ply"

    if ! conda run -n "${scale_env}" python -c "import trimesh" >/dev/null 2>&1; then
      echo "Installing trimesh into ${scale_env} for mesh scaling outputs..."
      conda run -n "${scale_env}" python -m pip install trimesh
    fi

    scale_flags=()
    if [[ ${estimate_scale} -eq 1 ]]; then
      scale_flags+=("--estimate-scale")
    fi
    if [[ ${fine_registration} -eq 1 ]]; then
      scale_flags+=("--fine-registration")
    fi

    conda run -n "${scale_env}" python "${repo_root}/src/sam3d_scale.py" \
      --sam3d-ply "${sam3d_pose_ply}" \
      --moge-npz "${target_npz}" \
      --algo "${scale_algo}" \
      --output-dir "${scale_out_dir}" \
      --output-scale "${scale_txt}" \
      --output-scaled-ply "${scaled_ply}" \
      --icp-max-iters "${icp_max_iters}" \
      --mode "${scale_mode}" \
      --teaser-estimate-scaling \
      --teaser-noise-bound "${teaser_noise_bound}" \
      --teaser-nn-max-points "${teaser_nn_max_points}" \
      --teaser-max-correspondences "${teaser_max_correspondences}" \
      --teaser-gnc-factor "${teaser_gnc_factor}" \
      --teaser-rot-max-iters "${teaser_rot_max_iters}" \
      --teaser-cbar2 "${teaser_cbar2}" \
      --teaser-iterations "${teaser_iterations}" \
      --teaser-correspondence "${teaser_correspondence}" \
      --teaser-fpfh-voxel "${teaser_fpfh_voxel}" \
      --teaser-fpfh-normal-radius "${teaser_fpfh_normal_radius}" \
      --teaser-fpfh-feature-radius "${teaser_fpfh_feature_radius}" \
      --teaser-icp-refine \
      --teaser-icp-max-iters "${teaser_icp_max_iters}" \
      --teaser-icp-distance "${teaser_icp_distance}" \
      "${scale_flags[@]}"

    if [[ ${mesh_decimate} -eq 1 ]]; then
      mesh_inputs=()
      for ext in glb ply obj; do
        mesh_path="${scale_out_dir}/${mask_stem}_scaled_mesh.${ext}"
        if [[ -f "${mesh_path}" ]]; then
          mesh_inputs+=("${mesh_path}")
        fi
      done

      if [[ ${#mesh_inputs[@]} -eq 0 ]]; then
        echo "No scaled mesh found for decimation: ${scale_out_dir}/${mask_stem}_scaled_mesh.*"
      else
        for mesh_path in "${mesh_inputs[@]}"; do
          mesh_name="$(basename "${mesh_path}")"
          mesh_ext="${mesh_name##*.}"
          mesh_base="${mesh_name%.*}"
          decimated_path="${scale_out_dir}/${mesh_base}_decimated.${mesh_ext}"

          decimate_args=(
            --input "${mesh_path}"
            --output "${decimated_path}"
            --ratio "${mesh_decimate_ratio}"
            --method "${mesh_decimate_method}"
            --min-faces "${mesh_decimate_min_faces}"
          )
          if [[ ${mesh_target_faces} -gt 0 ]]; then
            decimate_args+=(--target-faces "${mesh_target_faces}")
          fi

          if ! conda run -n "${scale_env}" python "${repo_root}/src/mesh_decimate.py" "${decimate_args[@]}"; then
            echo "Mesh decimation failed: ${mesh_path}"
          fi
        done
      fi
    fi
  else
    echo "Skipping sam3d_scale (no target point cloud)."
  fi

done
