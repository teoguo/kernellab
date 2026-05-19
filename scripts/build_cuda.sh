#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_root="${CUDA_ROOT:-/usr/local/cuda-13.0}"
build_dir="${1:-build-cuda}"

if [[ ! -x "${cuda_root}/bin/nvcc" ]]; then
  echo "nvcc not found at ${cuda_root}/bin/nvcc" >&2
  exit 1
fi

cmake -S "${repo_root}" -B "${repo_root}/${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DKERNELLAB_ENABLE_CUDA=ON \
  -DKERNELLAB_BUILD_TESTS=ON \
  -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${cuda_root}"

cmake --build "${repo_root}/${build_dir}"
ctest --test-dir "${repo_root}/${build_dir}" --output-on-failure
