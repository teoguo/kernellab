#include "kernellab/backend/cuda_support.hpp"

#include <limits>
#include <memory>
#include <string>

#if defined(KERNELLAB_HAS_CUDA) && KERNELLAB_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace kernellab {

namespace {

bool CheckedFloatMatrixBytes(const std::int64_t rows, const std::int64_t cols,
                             std::size_t& bytes) noexcept {
  if (rows < 0 || cols < 0) {
    return false;
  }
  const auto row_count = static_cast<std::size_t>(rows);
  const auto col_count = static_cast<std::size_t>(cols);
  if (row_count != 0 && col_count > (std::numeric_limits<std::size_t>::max() / row_count)) {
    return false;
  }
  const std::size_t elements = row_count * col_count;
  if (elements > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
    return false;
  }
  bytes = elements * sizeof(float);
  return true;
}

std::string BackendPrefix(const std::string_view backend_id) {
  return std::string(backend_id);
}

class UnavailableBackend final : public Backend {
public:
  explicit UnavailableBackend(std::string backend_id) : backend_id_(std::move(backend_id)) {}

  std::string_view id() const noexcept override {
    return backend_id_;
  }

  bool is_available() const noexcept override {
    return false;
  }

  std::string unavailable_reason() const override {
    return "CUDA support is not compiled";
  }

  Status Prepare(const Matrix&, const Matrix&, BackendSessionPtr&) const override {
    return Status::Error(backend_id_ +
                         " backend is unavailable because CUDA support is not compiled");
  }

private:
  std::string backend_id_;
};

} // namespace

Status ValidateCudaGemmInputs(const Matrix& a, const Matrix& b, const std::string_view backend_id) {
  const std::string prefix = BackendPrefix(backend_id);
  if (a.cols() != b.rows()) {
    return Status::Error(prefix + " requires a.cols() == b.rows()");
  }
  constexpr auto max_int = static_cast<std::int64_t>(std::numeric_limits<int>::max());
  if (a.rows() > max_int || b.cols() > max_int || a.cols() > max_int) {
    return Status::Error(prefix + " dimensions exceed int32 range");
  }

  std::size_t ignored = 0;
  if (!CheckedFloatMatrixBytes(a.rows(), a.cols(), ignored) ||
      !CheckedFloatMatrixBytes(b.rows(), b.cols(), ignored) ||
      !CheckedFloatMatrixBytes(a.rows(), b.cols(), ignored)) {
    return Status::Error(prefix + " matrix byte size overflow");
  }
  return Status::Ok();
}

std::size_t CudaFloatMatrixBytes(const std::int64_t rows, const std::int64_t cols) noexcept {
  std::size_t bytes = 0;
  const bool ok = CheckedFloatMatrixBytes(rows, cols, bytes);
  return ok ? bytes : 0;
}

#if defined(KERNELLAB_HAS_CUDA) && KERNELLAB_HAS_CUDA

bool CudaBackendsCompiled() noexcept {
  return true;
}

const CudaRuntimeInfo& GetCudaRuntimeInfo() noexcept {
  static const CudaRuntimeInfo info = [] {
    CudaRuntimeInfo runtime_info;
    runtime_info.compiled = true;

    cudaError_t status = cudaDriverGetVersion(&runtime_info.driver_version);
    if (status != cudaSuccess) {
      runtime_info.error_message =
          "cudaDriverGetVersion failed: " + std::string(cudaGetErrorString(status));
      return runtime_info;
    }

    status = cudaRuntimeGetVersion(&runtime_info.runtime_version);
    if (status != cudaSuccess) {
      runtime_info.error_message =
          "cudaRuntimeGetVersion failed: " + std::string(cudaGetErrorString(status));
      return runtime_info;
    }

    status = cudaGetDeviceCount(&runtime_info.device_count);
    if (status != cudaSuccess) {
      runtime_info.error_message =
          "cudaGetDeviceCount failed: " + std::string(cudaGetErrorString(status));
      return runtime_info;
    }

    if (runtime_info.driver_version < runtime_info.runtime_version) {
      runtime_info.error_message = "CUDA driver version is older than CUDA runtime version";
      return runtime_info;
    }

    if (runtime_info.device_count <= 0) {
      runtime_info.error_message = "no CUDA devices available";
      return runtime_info;
    }

    runtime_info.available = true;
    return runtime_info;
  }();

  return info;
}

#else

bool CudaBackendsCompiled() noexcept {
  return false;
}

const CudaRuntimeInfo& GetCudaRuntimeInfo() noexcept {
  static const CudaRuntimeInfo info = [] {
    CudaRuntimeInfo runtime_info;
    runtime_info.error_message = "CUDA support is not compiled";
    return runtime_info;
  }();

  return info;
}

BackendPtr MakeCudaNaiveBackend() {
  return std::make_unique<UnavailableBackend>("cuda_naive");
}

BackendPtr MakeCudaSmemBackend() {
  return std::make_unique<UnavailableBackend>("cuda_smem");
}

BackendPtr MakeCudaRegBackend() {
  return std::make_unique<UnavailableBackend>("cuda_reg");
}

BackendPtr MakeCublasBackend() {
  return std::make_unique<UnavailableBackend>("cublas");
}

#endif

} // namespace kernellab
