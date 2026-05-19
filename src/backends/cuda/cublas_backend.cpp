#include "kernellab/backend/cuda_support.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <memory>
#include <string>

namespace kernellab {

namespace {

const char* CublasStatusString(const cublasStatus_t status) {
  return cublasGetStatusString(status);
}

std::string FormatCudaError(const char* operation, const cudaError_t status) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

class CublasBackend final : public Backend {
private:
  class CublasSession;

public:
  std::string_view id() const noexcept override {
    return "cublas";
  }

  bool is_available() const noexcept override {
    return GetCudaRuntimeInfo().available;
  }

  std::string unavailable_reason() const override {
    const auto& info = GetCudaRuntimeInfo();
    return info.error_message.empty() ? "CUDA unavailable" : info.error_message;
  }

  Status Prepare(const Matrix& a, const Matrix& b, BackendSessionPtr& session) const override {
    const auto validation_status = ValidateCudaGemmInputs(a, b, id());
    if (!validation_status.ok) {
      return validation_status;
    }
    auto fresh = std::make_unique<CublasSession>(a, b);
    const auto init_status = fresh->Initialize();
    if (!init_status.ok) {
      return init_status;
    }
    session = std::move(fresh);
    return Status::Ok();
  }

private:
  class CublasSession final : public BackendSession {
  public:
    CublasSession(const Matrix& a, const Matrix& b)
        : a_(a), b_(b), m_(static_cast<int>(a.rows())), n_(static_cast<int>(b.cols())),
          k_(static_cast<int>(a.cols())), a_bytes_(CudaFloatMatrixBytes(a.rows(), a.cols())),
          b_bytes_(CudaFloatMatrixBytes(b.rows(), b.cols())),
          c_bytes_(CudaFloatMatrixBytes(a.rows(), b.cols())) {}

    ~CublasSession() override {
      if (handle_ != nullptr) {
        cublasDestroy(handle_);
      }
      if (start_event_ != nullptr) {
        cudaEventDestroy(start_event_);
      }
      if (stop_event_ != nullptr) {
        cudaEventDestroy(stop_event_);
      }
      if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
      }
      cudaFree(device_a_);
      cudaFree(device_b_);
      cudaFree(device_c_);
    }

    Status Run(Matrix& c, IterationMeasurement& measurement) override {
      // Initialize() runs in Prepare(), so first Run() iteration is no
      // longer contaminated by allocation / H2D / handle creation cost.
      if (c.rows() != a_.rows() || c.cols() != b_.cols()) {
        return Status::Error("cublas output matrix has wrong shape");
      }

      const auto wall_start = std::chrono::steady_clock::now();
      const cudaError_t record_start = cudaEventRecord(start_event_, stream_);
      if (record_start != cudaSuccess) {
        return Status::Error(FormatCudaError("cublas cudaEventRecord(start)", record_start));
      }

      const float alpha = 1.0F;
      const float beta = 0.0F;
      const cublasStatus_t gemm_status =
          cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n_, m_, k_, &alpha, device_b_, n_,
                      device_a_, k_, &beta, device_c_, n_);
      if (gemm_status != CUBLAS_STATUS_SUCCESS) {
        return Status::Error("cublasSgemm failed: " + std::string(CublasStatusString(gemm_status)));
      }

      const cudaError_t record_stop = cudaEventRecord(stop_event_, stream_);
      if (record_stop != cudaSuccess) {
        return Status::Error(FormatCudaError("cublas cudaEventRecord(stop)", record_stop));
      }
      const cudaError_t event_sync = cudaEventSynchronize(stop_event_);
      if (event_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cublas cudaEventSynchronize", event_sync));
      }

      float kernel_ms = 0.0F;
      const cudaError_t elapsed_status =
          cudaEventElapsedTime(&kernel_ms, start_event_, stop_event_);
      if (elapsed_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cublas cudaEventElapsedTime", elapsed_status));
      }

      const cudaError_t copy_out =
          cudaMemcpyAsync(c.data(), device_c_, c_bytes_, cudaMemcpyDeviceToHost, stream_);
      if (copy_out != cudaSuccess) {
        return Status::Error(FormatCudaError("cublas cudaMemcpyAsync D2H", copy_out));
      }
      const cudaError_t stream_sync = cudaStreamSynchronize(stream_);
      if (stream_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cublas cudaStreamSynchronize", stream_sync));
      }

      const auto wall_finish = std::chrono::steady_clock::now();
      measurement.kernel_ms = static_cast<double>(kernel_ms);
      measurement.e2e_ms =
          std::chrono::duration<double, std::milli>(wall_finish - wall_start).count();
      return Status::Ok();
    }

    // Public so CublasBackend::Prepare can drive setup before handing
    // the session to the executor.
    Status Initialize() {
      const cudaError_t stream_status = cudaStreamCreate(&stream_);
      if (stream_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cublas cudaStreamCreate", stream_status));
      }

      const cudaError_t start_status = cudaEventCreate(&start_event_);
      if (start_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cublas cudaEventCreate(start)", start_status));
      }
      const cudaError_t stop_status = cudaEventCreate(&stop_event_);
      if (stop_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cublas cudaEventCreate(stop)", stop_status));
      }

      const cudaError_t alloc_a = cudaMalloc(&device_a_, a_bytes_);
      const cudaError_t alloc_b =
          (alloc_a == cudaSuccess) ? cudaMalloc(&device_b_, b_bytes_) : alloc_a;
      const cudaError_t alloc_c = (alloc_a == cudaSuccess && alloc_b == cudaSuccess)
                                      ? cudaMalloc(&device_c_, c_bytes_)
                                      : alloc_b;
      if (alloc_a != cudaSuccess || alloc_b != cudaSuccess || alloc_c != cudaSuccess) {
        const cudaError_t failure =
            (alloc_a != cudaSuccess) ? alloc_a : ((alloc_b != cudaSuccess) ? alloc_b : alloc_c);
        return Status::Error(FormatCudaError("cublas cudaMalloc", failure));
      }

      const cublasStatus_t create_status = cublasCreate(&handle_);
      if (create_status != CUBLAS_STATUS_SUCCESS) {
        return Status::Error("cublasCreate failed: " +
                             std::string(CublasStatusString(create_status)));
      }
      const cublasStatus_t math_mode_status = cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH);
      if (math_mode_status != CUBLAS_STATUS_SUCCESS) {
        return Status::Error("cublasSetMathMode failed: " +
                             std::string(CublasStatusString(math_mode_status)));
      }
      const cublasStatus_t stream_bind_status = cublasSetStream(handle_, stream_);
      if (stream_bind_status != CUBLAS_STATUS_SUCCESS) {
        return Status::Error("cublasSetStream failed: " +
                             std::string(CublasStatusString(stream_bind_status)));
      }

      const cudaError_t copy_a =
          cudaMemcpyAsync(device_a_, a_.data(), a_bytes_, cudaMemcpyHostToDevice, stream_);
      const cudaError_t copy_b =
          (copy_a == cudaSuccess)
              ? cudaMemcpyAsync(device_b_, b_.data(), b_bytes_, cudaMemcpyHostToDevice, stream_)
              : copy_a;
      if (copy_a != cudaSuccess || copy_b != cudaSuccess) {
        const cudaError_t failure = (copy_a != cudaSuccess) ? copy_a : copy_b;
        return Status::Error(FormatCudaError("cublas cudaMemcpyAsync H2D", failure));
      }

      const cudaError_t stream_sync = cudaStreamSynchronize(stream_);
      if (stream_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cublas cudaStreamSynchronize", stream_sync));
      }

      return Status::Ok();
    }

  private:
    const Matrix& a_;
    const Matrix& b_;
    int m_ = 0;
    int n_ = 0;
    int k_ = 0;
    std::size_t a_bytes_ = 0;
    std::size_t b_bytes_ = 0;
    std::size_t c_bytes_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
    cublasHandle_t handle_ = nullptr;
    float* device_a_ = nullptr;
    float* device_b_ = nullptr;
    float* device_c_ = nullptr;
  };
};

} // namespace

BackendPtr MakeCublasBackend() {
  return std::make_unique<CublasBackend>();
}

} // namespace kernellab
