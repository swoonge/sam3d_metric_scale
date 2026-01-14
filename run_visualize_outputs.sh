#!/usr/bin/env bash
set -euo pipefail

# 통합 시각화 UI 실행 스크립트

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"

# 기본 conda env
sam3d_env="sam3d-objects"
output_root=""
image_path=""
server_port=""
share=0
no_browser=0
moge_max_points=""
moge_axis_fraction=""
moge_axis_steps=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --output-root PATH   Output root directory (required)
  --image PATH         Source image path (optional)
  --sam3d-env NAME     Conda env for visualization (default: sam3d-objects)
  --server-port INT    Gradio server port
  --share              Enable gradio share link
  --no-browser         Do not auto-open a browser window
  --moge-max-points N  Max points for MoGe point cloud
  --moge-axis-fraction F Axis length as fraction of bbox max dim
  --moge-axis-steps N  Points per axis for MoGe axes
  -h, --help           Show this help
USAGE
}

resolve_path() {
  # 상대 경로를 레포 루트 기준으로 변환
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
    --image)
      image_path="$2"
      shift 2
      ;;
    --sam3d-env)
      sam3d_env="$2"
      shift 2
      ;;
    --server-port)
      server_port="$2"
      shift 2
      ;;
    --share)
      share=1
      shift
      ;;
    --no-browser)
      no_browser=1
      shift
      ;;
    --moge-max-points)
      moge_max_points="$2"
      shift 2
      ;;
    --moge-axis-fraction)
      moge_axis_fraction="$2"
      shift 2
      ;;
    --moge-axis-steps)
      moge_axis_steps="$2"
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

if [[ -z "${output_root}" ]]; then
  # 필수 인자 체크
  echo "--output-root is required"
  usage
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi

output_root="$(resolve_path "${output_root}")"
args=(--output-root "${output_root}")

if [[ -n "${image_path}" ]]; then
  image_path="$(resolve_path "${image_path}")"
  args+=(--image "${image_path}")
fi

if [[ -n "${server_port}" ]]; then
  args+=(--server-port "${server_port}")
fi

if [[ ${share} -eq 1 ]]; then
  args+=(--share)
fi

if [[ ${no_browser} -eq 1 ]]; then
  args+=(--no-browser)
fi

if [[ -n "${moge_max_points}" ]]; then
  args+=(--moge-max-points "${moge_max_points}")
fi

if [[ -n "${moge_axis_fraction}" ]]; then
  args+=(--moge-axis-fraction "${moge_axis_fraction}")
fi

if [[ -n "${moge_axis_steps}" ]]; then
  args+=(--moge-axis-steps "${moge_axis_steps}")
fi

conda run -n "${sam3d_env}" python "${repo_root}/src/visualize_outputs.py" "${args[@]}"
