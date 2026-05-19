#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-build}"

cmake -S "${repo_root}" -B "${repo_root}/${build_dir}" \
  -DKERNELLAB_ENABLE_CUDA=OFF \
  -DKERNELLAB_BUILD_TESTS=ON

cmake --build "${repo_root}/${build_dir}"
ctest --test-dir "${repo_root}/${build_dir}" --output-on-failure
