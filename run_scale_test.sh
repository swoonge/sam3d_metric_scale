# ./sam3d_metric_scale/run_scale_test.sh   --base_root /home/vision/Sim2Real_Data_Augmentation_for_VLA/sam3d_metric_scale/outputs/coffee_maker_sample   --sam3d_file /sam3d/coffee_maker_sample_000.ply   --moge_file /moge_scale/coffee_maker_sample_coffee_maker_sample_000.npz   --algo teaserpp   --teaser-estimate-scaling   --teaser-noise-bound 0   --teaser-nn-max-points 4000   --teaser-max-correspondences 3000   --teaser-gnc-factor 1.4   --teaser-rot-max-iters 100   --teaser-cbar2 1.0   --teaser-iterations 2   --teaser-correspondence fpfh   --teaser-fpfh-voxel 0   --teaser-fpfh-normal-radius 0   --teaser-fpfh-feature-radius 0   --teaser-icp-refine   --teaser-icp-max-iters 100   --teaser-icp-distance 0   --sam3d-env teaserpp   --show-viz --viz-method open3d

#!/usr/bin/env bash
set -euo pipefail

# SAM3D 스케일 알고리즘 테스트 러너

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"

sam3d_env="teaserpp"
output_root=""
base_root=""
moge_file=""
sam3d_file=""
mask_stem=""
algo="teaserpp"
output_dir=""
use_all=0

show_viz=1
save_viz=0
debug_viz=0
viz_path=""
viz_max_points=""
viz_max_pairs=""
viz_dpi=""
viz_method="open3d"

seed=""
max_points=""

icp_max_iters=""
icp_tolerance=""
icp_nn_max_points=""
icp_trim_ratio=""

ransac_iters=""
ransac_sample=""
ransac_inlier_thresh=""
ransac_nn_max_points=""

rms_nn_max_points=""
teaser_noise_bound="0"
teaser_nn_max_points="4000"
teaser_max_correspondences="3000"
teaser_gnc_factor="1.4"
teaser_rot_max_iters="100"
teaser_cbar2="1.0"
teaser_iterations=""
teaser_correspondence="fpfh"
teaser_fpfh_voxel="0"
teaser_fpfh_normal_radius="0"
teaser_fpfh_feature_radius="0"
teaser_estimate_scaling=1
teaser_icp_refine=0
teaser_icp_max_iters=""
teaser_icp_distance=""
super4pcs_bin=""
super4pcs_overlap=""
super4pcs_delta=""
super4pcs_timeout=""
run_args=()

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --base_root PATH       Base output root (same as --output-root)
  --output-root PATH     Output root directory
  --moge_file PATH       MoGe npz path (relative to base_root if starts with /)
  --sam3d_file PATH      SAM3D ply path (relative to base_root if starts with /)
  --mask-stem NAME       Mask stem (e.g. coffee_maker_sample_000)
  --all                 Run for all masks in output_root/sam3d
  --algo NAME            icp | ransac | rms | teaserpp | super4pcs (default: icp)
  --output-dir PATH      Override output dir for scale results
  --sam3d-env NAME       Conda env for sam3d scale (default: sam3d-objects)
  --show-viz             Show matplotlib window
  --save-viz             Save match visualization png
  --debug-viz            Show debug 5-panel visualization
  --viz-path PATH        Custom path for visualization (single run only)
  --viz-max-points N     Max points for visualization
  --viz-max-pairs N      Max pair lines for visualization
  --viz-dpi N            DPI for saved visualization
  --viz-method NAME      matplotlib | open3d (default: matplotlib)
  --seed N               Random seed
  --max-points N         Max points for algorithm input
  --icp-max-iters N      ICP max iterations
  --icp-tolerance F      ICP tolerance
  --icp-nn-max-points N  ICP NN sample size
  --icp-trim-ratio F     ICP trim ratio (0<r<=1)
  --ransac-iters N       RANSAC iterations
  --ransac-sample N      RANSAC sample size
  --ransac-inlier-thresh F RANSAC inlier threshold
  --ransac-nn-max-points N RANSAC NN sample size
  --rms-nn-max-points N  RMS NN sample size
  --teaser-noise-bound F TEASER++ noise bound (<=0 auto)
  --teaser-nn-max-points N TEASER++ NN sample size
  --teaser-max-correspondences N TEASER++ max correspondences
  --teaser-gnc-factor F  TEASER++ GNC factor
  --teaser-rot-max-iters N TEASER++ rotation max iters
  --teaser-cbar2 F       TEASER++ cbar2
  --teaser-iterations N  TEASER++ repeat passes (default: 1)
  --teaser-correspondence MODE fpfh (default: fpfh)
  --teaser-fpfh-voxel F  FPFH voxel size (<=0 auto)
  --teaser-fpfh-normal-radius F FPFH normal radius (<=0 auto)
  --teaser-fpfh-feature-radius F FPFH feature radius (<=0 auto)
  --teaser-estimate-scaling Enable TEASER++ scaling
  --teaser-icp-refine    Enable ICP refinement after TEASER++
  --teaser-icp-max-iters N ICP max iterations
  --teaser-icp-distance F ICP max correspondence distance (<=0 auto)
  --super4pcs-bin PATH   Super4PCS binary path
  --super4pcs-overlap F  Super4PCS overlap
  --super4pcs-delta F    Super4PCS delta (<=0 auto)
  --super4pcs-timeout N  Super4PCS timeout (tool-specific)
  -h, --help             Show this help
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --base_root)
      base_root="$2"
      shift 2
      ;;
    --moge_file)
      moge_file="$2"
      shift 2
      ;;
    --sam3d_file)
      sam3d_file="$2"
      shift 2
      ;;
    --mask-stem)
      mask_stem="$2"
      shift 2
      ;;
    --all)
      use_all=1
      shift
      ;;
    --algo)
      algo="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --sam3d-env)
      sam3d_env="$2"
      shift 2
      ;;
    --show-viz)
      show_viz=1
      shift
      ;;
    --save-viz)
      save_viz=1
      shift
      ;;
    --debug-viz)
      debug_viz=1
      shift
      ;;
    --viz-path)
      viz_path="$2"
      shift 2
      ;;
    --viz-max-points)
      viz_max_points="$2"
      shift 2
      ;;
    --viz-max-pairs)
      viz_max_pairs="$2"
      shift 2
      ;;
    --viz-dpi)
      viz_dpi="$2"
      shift 2
      ;;
    --viz-method)
      viz_method="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --max-points)
      max_points="$2"
      shift 2
      ;;
    --icp-max-iters)
      icp_max_iters="$2"
      shift 2
      ;;
    --icp-tolerance)
      icp_tolerance="$2"
      shift 2
      ;;
    --icp-nn-max-points)
      icp_nn_max_points="$2"
      shift 2
      ;;
    --icp-trim-ratio)
      icp_trim_ratio="$2"
      shift 2
      ;;
    --ransac-iters)
      ransac_iters="$2"
      shift 2
      ;;
    --ransac-sample)
      ransac_sample="$2"
      shift 2
      ;;
    --ransac-inlier-thresh)
      ransac_inlier_thresh="$2"
      shift 2
      ;;
    --ransac-nn-max-points)
      ransac_nn_max_points="$2"
      shift 2
      ;;
    --rms-nn-max-points)
      rms_nn_max_points="$2"
      shift 2
      ;;
    --teaser-noise-bound)
      teaser_noise_bound="$2"
      shift 2
      ;;
    --teaser-nn-max-points)
      teaser_nn_max_points="$2"
      shift 2
      ;;
    --teaser-max-correspondences)
      teaser_max_correspondences="$2"
      shift 2
      ;;
    --teaser-gnc-factor)
      teaser_gnc_factor="$2"
      shift 2
      ;;
    --teaser-rot-max-iters)
      teaser_rot_max_iters="$2"
      shift 2
      ;;
    --teaser-cbar2)
      teaser_cbar2="$2"
      shift 2
      ;;
    --teaser-iterations)
      teaser_iterations="$2"
      shift 2
      ;;
    --teaser-correspondence)
      teaser_correspondence="$2"
      shift 2
      ;;
    --teaser-fpfh-voxel)
      teaser_fpfh_voxel="$2"
      shift 2
      ;;
    --teaser-fpfh-normal-radius)
      teaser_fpfh_normal_radius="$2"
      shift 2
      ;;
    --teaser-fpfh-feature-radius)
      teaser_fpfh_feature_radius="$2"
      shift 2
      ;;
    --teaser-estimate-scaling)
      teaser_estimate_scaling=1
      shift
      ;;
    --teaser-icp-refine)
      teaser_icp_refine=1
      shift
      ;;
    --teaser-icp-max-iters)
      teaser_icp_max_iters="$2"
      shift 2
      ;;
    --teaser-icp-distance)
      teaser_icp_distance="$2"
      shift 2
      ;;
    --super4pcs-bin)
      super4pcs_bin="$2"
      shift 2
      ;;
    --super4pcs-overlap)
      super4pcs_overlap="$2"
      shift 2
      ;;
    --super4pcs-delta)
      super4pcs_delta="$2"
      shift 2
      ;;
    --super4pcs-timeout)
      super4pcs_timeout="$2"
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

if [[ -n "${base_root}" ]]; then
  if [[ -n "${output_root}" && "${output_root}" != "${base_root}" ]]; then
    echo "--output-root and --base_root are both set but different"
    exit 1
  fi
  output_root="${base_root}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi

if [[ -n "${output_root}" ]]; then
  output_root="$(resolve_path "${output_root}")"
fi
sam3d_dir="${output_root}/sam3d"
moge_dir="${output_root}/moge_scale"

resolve_file() {
  local base="$1"
  local path="$2"
  if [[ -z "${base}" ]]; then
    echo "$(resolve_path "${path}")"
    return 0
  fi
  if [[ "${path}" = /* ]]; then
    if [[ "${path}" == "${base}"* ]]; then
      echo "${path}"
    else
      echo "${base%/}${path}"
    fi
    return 0
  fi
  echo "${base%/}/${path}"
}

find_moge_npz() {
  local stem="$1"
  local candidate=""
  candidate="$(ls -1 "${moge_dir}"/*_"${stem}".npz 2>/dev/null | head -n 1 || true)"
  if [[ -z "${candidate}" ]]; then
    candidate="$(ls -1 "${moge_dir}/${stem}.npz" 2>/dev/null | head -n 1 || true)"
  fi
  echo "${candidate}"
}

build_args() {
  local sam3d_ply="$1"
  local moge_npz="$2"
  run_args=(
    --sam3d-ply "${sam3d_ply}"
    --moge-npz "${moge_npz}"
    --algo "${algo}"
  )
  if [[ -n "${output_dir}" ]]; then
    run_args+=(--output-dir "$(resolve_path "${output_dir}")")
  fi
  if [[ ${show_viz} -eq 1 ]]; then
    run_args+=(--show-viz)
  fi
  if [[ ${save_viz} -eq 1 ]]; then
    run_args+=(--save-viz)
  fi
  if [[ ${debug_viz} -eq 1 ]]; then
    run_args+=(--debug-viz)
  fi
  if [[ -n "${viz_path}" ]]; then
    run_args+=(--viz-path "$(resolve_path "${viz_path}")")
  fi
  if [[ -n "${viz_max_points}" ]]; then
    run_args+=(--viz-max-points "${viz_max_points}")
  fi
  if [[ -n "${viz_max_pairs}" ]]; then
    run_args+=(--viz-max-pairs "${viz_max_pairs}")
  fi
  if [[ -n "${viz_dpi}" ]]; then
    run_args+=(--viz-dpi "${viz_dpi}")
  fi
  if [[ -n "${viz_method}" ]]; then
    run_args+=(--viz-method "${viz_method}")
  fi
  if [[ -n "${seed}" ]]; then
    run_args+=(--seed "${seed}")
  fi
  if [[ -n "${max_points}" ]]; then
    run_args+=(--max-points "${max_points}")
  fi

  if [[ -n "${icp_max_iters}" ]]; then
    run_args+=(--icp-max-iters "${icp_max_iters}")
  fi
  if [[ -n "${icp_tolerance}" ]]; then
    run_args+=(--icp-tolerance "${icp_tolerance}")
  fi
  if [[ -n "${icp_nn_max_points}" ]]; then
    run_args+=(--icp-nn-max-points "${icp_nn_max_points}")
  fi
  if [[ -n "${icp_trim_ratio}" ]]; then
    run_args+=(--icp-trim-ratio "${icp_trim_ratio}")
  fi

  if [[ -n "${ransac_iters}" ]]; then
    run_args+=(--ransac-iters "${ransac_iters}")
  fi
  if [[ -n "${ransac_sample}" ]]; then
    run_args+=(--ransac-sample "${ransac_sample}")
  fi
  if [[ -n "${ransac_inlier_thresh}" ]]; then
    run_args+=(--ransac-inlier-thresh "${ransac_inlier_thresh}")
  fi
  if [[ -n "${ransac_nn_max_points}" ]]; then
    run_args+=(--ransac-nn-max-points "${ransac_nn_max_points}")
  fi

  if [[ -n "${rms_nn_max_points}" ]]; then
    run_args+=(--rms-nn-max-points "${rms_nn_max_points}")
  fi
  if [[ -n "${teaser_noise_bound}" ]]; then
    run_args+=(--teaser-noise-bound "${teaser_noise_bound}")
  fi
  if [[ -n "${teaser_nn_max_points}" ]]; then
    run_args+=(--teaser-nn-max-points "${teaser_nn_max_points}")
  fi
  if [[ -n "${teaser_max_correspondences}" ]]; then
    run_args+=(--teaser-max-correspondences "${teaser_max_correspondences}")
  fi
  if [[ -n "${teaser_gnc_factor}" ]]; then
    run_args+=(--teaser-gnc-factor "${teaser_gnc_factor}")
  fi
  if [[ -n "${teaser_rot_max_iters}" ]]; then
    run_args+=(--teaser-rot-max-iters "${teaser_rot_max_iters}")
  fi
  if [[ -n "${teaser_cbar2}" ]]; then
    run_args+=(--teaser-cbar2 "${teaser_cbar2}")
  fi
  if [[ -n "${teaser_iterations}" ]]; then
    run_args+=(--teaser-iterations "${teaser_iterations}")
  fi
  if [[ -n "${teaser_correspondence}" ]]; then
    run_args+=(--teaser-correspondence "${teaser_correspondence}")
  fi
  if [[ -n "${teaser_fpfh_voxel}" ]]; then
    run_args+=(--teaser-fpfh-voxel "${teaser_fpfh_voxel}")
  fi
  if [[ -n "${teaser_fpfh_normal_radius}" ]]; then
    run_args+=(--teaser-fpfh-normal-radius "${teaser_fpfh_normal_radius}")
  fi
  if [[ -n "${teaser_fpfh_feature_radius}" ]]; then
    run_args+=(--teaser-fpfh-feature-radius "${teaser_fpfh_feature_radius}")
  fi
  if [[ ${teaser_estimate_scaling} -eq 1 ]]; then
    run_args+=(--teaser-estimate-scaling)
  fi
  if [[ ${teaser_icp_refine} -eq 1 ]]; then
    run_args+=(--teaser-icp-refine)
  fi
  if [[ -n "${teaser_icp_max_iters}" ]]; then
    run_args+=(--teaser-icp-max-iters "${teaser_icp_max_iters}")
  fi
  if [[ -n "${teaser_icp_distance}" ]]; then
    run_args+=(--teaser-icp-distance "${teaser_icp_distance}")
  fi
  if [[ -n "${super4pcs_bin}" ]]; then
    run_args+=(--super4pcs-bin "${super4pcs_bin}")
  fi
  if [[ -n "${super4pcs_overlap}" ]]; then
    run_args+=(--super4pcs-overlap "${super4pcs_overlap}")
  fi
  if [[ -n "${super4pcs_delta}" ]]; then
    run_args+=(--super4pcs-delta "${super4pcs_delta}")
  fi
  if [[ -n "${super4pcs_timeout}" ]]; then
    run_args+=(--super4pcs-timeout "${super4pcs_timeout}")
  fi

  return 0
}

run_one() {
  local stem="$1"
  local sam3d_ply="${sam3d_dir}/${stem}.ply"
  if [[ ! -f "${sam3d_ply}" ]]; then
    echo "Missing SAM3D ply: ${sam3d_ply}"
    return 1
  fi
  local moge_npz
  moge_npz="$(find_moge_npz "${stem}")"
  if [[ -z "${moge_npz}" ]]; then
    echo "Missing MoGe npz for ${stem} under ${moge_dir}"
    return 1
  fi
  build_args "${sam3d_ply}" "${moge_npz}"
  conda run -n "${sam3d_env}" python -u "${repo_root}/src/sam3d_scale.py" "${run_args[@]}"
}

if [[ -n "${moge_file}" || -n "${sam3d_file}" ]]; then
  if [[ -z "${moge_file}" || -z "${sam3d_file}" ]]; then
    echo "--moge_file and --sam3d_file must be provided together"
    exit 1
  fi
  if [[ -z "${output_root}" ]]; then
    echo "--base_root (or --output-root) is required with --moge_file/--sam3d_file"
    exit 1
  fi
  if [[ ${use_all} -eq 1 || -n "${mask_stem}" ]]; then
    echo "--all/--mask-stem cannot be used with --moge_file/--sam3d_file"
    exit 1
  fi
  sam3d_ply="$(resolve_file "${output_root}" "${sam3d_file}")"
  moge_npz="$(resolve_file "${output_root}" "${moge_file}")"
  if [[ ! -f "${sam3d_ply}" ]]; then
    echo "Missing SAM3D ply: ${sam3d_ply}"
    exit 1
  fi
  if [[ ! -f "${moge_npz}" ]]; then
    echo "Missing MoGe npz: ${moge_npz}"
    exit 1
  fi
  build_args "${sam3d_ply}" "${moge_npz}"
  conda run -n "${sam3d_env}" python -u "${repo_root}/src/sam3d_scale.py" "${run_args[@]}"
  exit 0
fi

if [[ -z "${output_root}" ]]; then
  echo "--output-root is required"
  usage
  exit 1
fi

if [[ ! -d "${sam3d_dir}" ]]; then
  echo "Missing sam3d directory: ${sam3d_dir}"
  exit 1
fi
if [[ ! -d "${moge_dir}" ]]; then
  echo "Missing moge_scale directory: ${moge_dir}"
  exit 1
fi

if [[ ${use_all} -eq 1 && -n "${mask_stem}" ]]; then
  echo "--mask-stem cannot be used with --all"
  exit 1
fi

if [[ ${use_all} -eq 1 && -n "${viz_path}" ]]; then
  echo "--viz-path cannot be used with --all"
  exit 1
fi

if [[ ${use_all} -eq 1 ]]; then
  mapfile -t plys < <(ls -1 "${sam3d_dir}"/*.ply 2>/dev/null || true)
  if [[ ${#plys[@]} -eq 0 ]]; then
    echo "No PLY files found under ${sam3d_dir}"
    exit 1
  fi
  for ply in "${plys[@]}"; do
    stem="$(basename "${ply%.ply}")"
    run_one "${stem}"
  done
  exit 0
fi

if [[ -z "${mask_stem}" ]]; then
  latest_ply="$(ls -t "${sam3d_dir}"/*.ply 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_ply}" ]]; then
    echo "No PLY files found under ${sam3d_dir}"
    exit 1
  fi
  mask_stem="$(basename "${latest_ply%.ply}")"
fi

run_one "${mask_stem}"
