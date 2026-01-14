#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"

sam2_env="sam2"
sam3d_env="sam3d-objects"
moge_env="moge"

default_image="${repo_root}/sam2/notebooks/videos/bedroom/00031.jpg"
if [[ ! -f "${default_image}" && -f "${repo_root}/../sam2/notebooks/videos/bedroom/00031.jpg" ]]; then
  default_image="${repo_root}/../sam2/notebooks/videos/bedroom/00031.jpg"
fi
image_path="${default_image}"

output_base="${repo_root}/outputs"

default_sam3d_config="${repo_root}/sam-3d-objects/checkpoints/hf/pipeline.yaml"
if [[ ! -f "${default_sam3d_config}" && -f "${repo_root}/../sam-3d-objects/checkpoints/hf/pipeline.yaml" ]]; then
  default_sam3d_config="${repo_root}/../sam-3d-objects/checkpoints/hf/pipeline.yaml"
fi
sam3d_config="${default_sam3d_config}"

sam3d_seed=42
sam3d_compile=0

moge_model="Ruicheng/moge-2-vitl-normal"
scale_method="bbox_diag"
min_pixels=100

process_all=1

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --image PATH              Input image path for SAM2 UI
  --output-base PATH        Base output directory (default: outputs)
  --latest                  Process latest mask only (default: all)
  --sam2-env NAME           Conda env for SAM2 (default: sam2)
  --sam3d-env NAME          Conda env for SAM3D (default: sam3d-objects)
  --sam3d-config PATH       SAM3D pipeline.yaml path
  --sam3d-seed INT          Seed for SAM3D inference
  --sam3d-compile           Enable compile flag for SAM3D
  --moge-env NAME           Conda env for MoGe (default: moge)
  --moge-model NAME         HF model id or local path
  --scale-method NAME       bbox_diag | bbox_max (default: bbox_diag)
  --min-pixels INT          Minimum valid pixels for scale
  -h, --help                Show this help
USAGE
}

resolve_path() {
  local input="$1"
  if [[ "${input}" = /* ]]; then
    echo "${input}"
  else
    echo "${repo_root}/${input}"
  fi
}

make_output_root() {
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

image_path="$(resolve_path "${image_path}")"
output_base="$(resolve_path "${output_base}")"
sam3d_config="$(resolve_path "${sam3d_config}")"

image_stem="$(basename "${image_path}")"
image_stem="${image_stem%.*}"
output_root="$(make_output_root "${output_base}" "${image_stem}")"
mask_dir="${output_root}/sam2_masks"
sam3d_out_dir="${output_root}/sam3d"
moge_out_dir="${output_root}/moge_scale"

mkdir -p "${output_root}"
echo "Output root: ${output_root}"
cp -n "${image_path}" "${output_root}/" 2>/dev/null || true

conda run -n "${sam2_env}" python "${repo_root}/image_point.py" \
  --image "${image_path}" \
  --output-dir "${mask_dir}"

mkdir -p "${sam3d_out_dir}" "${moge_out_dir}"

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
  output_path="${sam3d_out_dir}/${mask_name%.png}.ply"

  conda run -n "${moge_env}" python "${repo_root}/moge_scale.py" \
    --image "${image_path}" \
    --mask "${mask_path}" \
    --model "${moge_model}" \
    --output-dir "${moge_out_dir}" \
    --scale-method "${scale_method}" \
    --min-pixels "${min_pixels}"

  compile_flag=""
  if [[ ${sam3d_compile} -eq 1 ]]; then
    compile_flag="--compile"
  fi

  conda run -n "${sam3d_env}" python "${repo_root}/sam3d_export.py" \
    --image "${image_path}" \
    --mask "${mask_path}" \
    --sam3d-config "${sam3d_config}" \
    --output "${output_path}" \
    --seed "${sam3d_seed}" \
    ${compile_flag}

done
