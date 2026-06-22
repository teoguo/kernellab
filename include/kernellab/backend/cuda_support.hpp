#pragma once

#include "kernellab/backend/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace kernellab {

struct CudaRuntimeInfo {
  bool compiled = false;
  bool available = false;
  int device_count = 0;
  int driver_version = 0;
  int runtime_version = 0;
  std::string error_message;
};

bool CudaBackendsCompiled() noexcept;
const CudaRuntimeInfo& GetCudaRuntimeInfo() noexcept;

Status ValidateCudaGemmInputs(const Matrix& a, const Matrix& b, std::string_view backend_id);
std::size_t CudaFloatMatrixBytes(std::int64_t rows, std::int64_t cols) noexcept;

BackendPtr MakeCudaNaiveBackend();
BackendPtr MakeCudaSmemBackend();
BackendPtr MakeCudaRegBackend();
BackendPtr MakeCudaRegV2Backend();
BackendPtr MakeCublasBackend();

} // namespace kernellab
