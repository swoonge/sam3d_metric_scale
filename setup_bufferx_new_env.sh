#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bufferx_root="${script_dir}/BUFFER-X"
env_name="bufferx_new"
python_version="3.11"
cuda_arch="8.9"
skip_apt=0
skip_download=0
recreate=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --env NAME           Conda env name (default: bufferx_new)
  --python VER         Python version (default: 3.11)
  --bufferx-root PATH  BUFFER-X root (default: ./BUFFER-X)
  --cuda-arch ARCH     CUDA arch list for build (default: 8.9)
  --skip-apt           Skip apt-get build deps
  --skip-download      Skip pretrained model download
  --recreate           Remove existing env before creation
  -h, --help           Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      env_name="$2"
      shift 2
      ;;
    --python)
      python_version="$2"
      shift 2
      ;;
    --bufferx-root)
      bufferx_root="$2"
      shift 2
      ;;
    --cuda-arch)
      cuda_arch="$2"
      shift 2
      ;;
    --skip-apt)
      skip_apt=1
      shift
      ;;
    --skip-download)
      skip_download=1
      shift
      ;;
    --recreate)
      recreate=1
      shift
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
  echo "conda is required but not found in PATH."
  exit 1
fi

if [[ ! -d "${bufferx_root}" ]]; then
  echo "BUFFER-X root not found: ${bufferx_root}"
  exit 1
fi

if [[ ! -d "${bufferx_root}/Pointnet2_PyTorch" ]]; then
  echo "Pointnet2_PyTorch not found: ${bufferx_root}/Pointnet2_PyTorch"
  exit 1
fi

if [[ ${skip_apt} -eq 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y \
      gcc-12 g++-12 build-essential cmake git ninja-build unzip \
      libgl1 libtbb-dev libeigen3-dev
  else
    echo "sudo not available; skipping apt-get."
  fi
fi

if conda env list | awk '{print $1}' | grep -q "^${env_name}$"; then
  if [[ ${recreate} -eq 1 ]]; then
    conda env remove -n "${env_name}" -y
  else
    echo "Conda env ${env_name} already exists. Re-run with --recreate to rebuild."
    exit 1
  fi
fi

conda create -n "${env_name}" \
  -c conda-forge -c defaults --override-channels --strict-channel-priority \
  python="${python_version}" pip -y

conda install -n "${env_name}" \
  -c nvidia/label/cuda-12.4.0 -c defaults --override-channels --strict-channel-priority -y \
  cuda-nvcc=12.4 \
  cuda-cccl=12.4 \
  cuda-cudart=12.4 cuda-cudart-dev=12.4 \
  cuda-nvrtc=12.4 cuda-nvrtc-dev=12.4 \
  cuda-libraries-dev=12.4

conda install -n "${env_name}" \
  -c conda-forge --override-channels --strict-channel-priority -y \
  numpy=1.26.4 open3d scikit-learn scipy

conda run -n "${env_name}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${env_name}" python -m pip install \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
conda run -n "${env_name}" python -m pip install \
  ninja kornia einops easydict tensorboard tensorboardX tabulate pathlib nibabel

conda run -n "${env_name}" bash -c " \
  export CUDA_HOME=\$CONDA_PREFIX; \
  export PATH=\$CUDA_HOME/bin:\$PATH; \
  export LD_LIBRARY_PATH=\$CUDA_HOME/lib:\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH; \
  export CUDACXX=\$CUDA_HOME/bin/nvcc; \
  export CPATH=\$CUDA_HOME/include:\$CPATH; \
  export C_INCLUDE_PATH=\$CUDA_HOME/include; \
  export CPLUS_INCLUDE_PATH=\$CUDA_HOME/include; \
  export CC=/usr/bin/gcc-12; \
  export CXX=/usr/bin/g++-12; \
  export CUDAHOSTCXX=/usr/bin/g++-12; \
  export TORCH_CUDA_ARCH_LIST=${cuda_arch}; \
  python -m pip install -v --no-build-isolation knn-cuda \
"

conda run -n "${env_name}" bash -c " \
  export CUDA_HOME=\$CONDA_PREFIX; \
  export PATH=\$CUDA_HOME/bin:\$PATH; \
  export LD_LIBRARY_PATH=\$CUDA_HOME/lib:\$CUDA_HOME/lib64:\$CUDA_HOME/targets/x86_64-linux/lib:\$CUDA_HOME/targets/x86_64-linux/lib64:\$LD_LIBRARY_PATH; \
  export CUDACXX=\$CUDA_HOME/bin/nvcc; \
  export CPATH=\$CUDA_HOME/include:\$CUDA_HOME/targets/x86_64-linux/include:\$CPATH; \
  export C_INCLUDE_PATH=\$CUDA_HOME/include:\$CUDA_HOME/targets/x86_64-linux/include; \
  export CPLUS_INCLUDE_PATH=\$CUDA_HOME/include:\$CUDA_HOME/targets/x86_64-linux/include; \
  export CC=/usr/bin/gcc-12; \
  export CXX=/usr/bin/g++-12; \
  export CUDAHOSTCXX=/usr/bin/g++-12; \
  export TORCH_CUDA_ARCH_LIST=${cuda_arch}; \
  export MAX_JOBS=8; \
  cd \"${bufferx_root}/Pointnet2_PyTorch\" && \
  rm -rf pointnet2_ops_lib/build && \
  python -m pip install -v --no-build-isolation pointnet2_ops_lib/. \
"

conda run -n "${env_name}" bash -c " \
  export CC=/usr/bin/gcc-12; \
  export CXX=/usr/bin/g++-12; \
  export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu; \
  export CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu; \
  cd \"${bufferx_root}/cpp_wrappers\" && \
  rm -rf cpp_subsampling/build cpp_neighbors/build && \
  sh compile_wrappers.sh \
"

if [[ ! -d "${bufferx_root}/torch-batch-svd" ]]; then
  git clone https://github.com/KinglittleQ/torch-batch-svd.git "${bufferx_root}/torch-batch-svd"
fi
conda run -n "${env_name}" bash -c " \
  cd \"${bufferx_root}/torch-batch-svd\" && \
  python -m pip install . \
"

if [[ ${skip_download} -eq 0 ]]; then
  bash "${bufferx_root}/scripts/download_pretrained_models.sh"
fi

echo ""
echo "BUFFER-X environment ${env_name} is ready."
