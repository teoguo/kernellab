#include "kernellab/backend/cuda_support.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <memory>
#include <string>

namespace kernellab {

namespace {

std::string FormatCudaError(const char* operation, const cudaError_t status) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

// 32x32 shared-memory tiled GEMM.
//
// Goal: drop the bytes/flop ratio of the naive kernel by reusing each
// global load BM x (or BN x) times before it leaves smem. With BM = BN
// = BK = 32 and one thread per output element, the inner-K loop reads
// BK = 32 elements per FMA pair from smem instead of from gmem -- so the
// global-memory traffic per output is roughly K/BM + K/BN, not 2K.
//
// Bank-conflict note: warps inside this block are rows of the 32x32
// thread grid (varying threadIdx.x, fixed threadIdx.y). For the inner
// accumulator `as[ty][inner] * bs[inner][tx]`:
//   - `as[ty][inner]` is broadcast across the warp (same address) -- no
//     conflict.
//   - `bs[inner][tx]` ranges tx = 0..31 across 32 banks -- one element
//     per bank, no conflict at 4-byte bank size.
// So this layout is conflict-free without padding.
constexpr int kTile = 32;

__global__ void SmemKernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, int m, int n, int k) {
  __shared__ float as[kTile][kTile];
  __shared__ float bs[kTile][kTile];

  const int block_row = blockIdx.y * kTile;
  const int block_col = blockIdx.x * kTile;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int row = block_row + ty;
  const int col = block_col + tx;

  float acc = 0.0F;
  for (int k_tile = 0; k_tile < k; k_tile += kTile) {
    const int a_col = k_tile + tx;
    const int b_row = k_tile + ty;
    // Zero-pad on the edge tiles so the unrolled inner loop doesn't need
    // its own bounds check.
    as[ty][tx] = (row < m && a_col < k) ? a[(row * k) + a_col] : 0.0F;
    bs[ty][tx] = (b_row < k && col < n) ? b[(b_row * n) + col] : 0.0F;
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kTile; ++inner) {
      acc += as[ty][inner] * bs[inner][tx];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[(row * n) + col] = acc;
  }
}

class CudaSmemBackend final : public Backend {
private:
  class CudaSmemSession;

public:
  std::string_view id() const noexcept override {
    return "cuda_smem";
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
    auto fresh = std::make_unique<CudaSmemSession>(a, b);
    const auto init_status = fresh->Initialize();
    if (!init_status.ok) {
      return init_status;
    }
    session = std::move(fresh);
    return Status::Ok();
  }

private:
  class CudaSmemSession final : public BackendSession {
  public:
    CudaSmemSession(const Matrix& a, const Matrix& b)
        : a_(a), b_(b), m_(static_cast<int>(a.rows())), n_(static_cast<int>(b.cols())),
          k_(static_cast<int>(a.cols())), a_bytes_(CudaFloatMatrixBytes(a.rows(), a.cols())),
          b_bytes_(CudaFloatMatrixBytes(b.rows(), b.cols())),
          c_bytes_(CudaFloatMatrixBytes(a.rows(), b.cols())) {}

    ~CudaSmemSession() override {
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
      if (c.rows() != a_.rows() || c.cols() != b_.cols()) {
        return Status::Error("cuda_smem output matrix has wrong shape");
      }

      const auto wall_start = std::chrono::steady_clock::now();
      const cudaError_t record_start = cudaEventRecord(start_event_, stream_);
      if (record_start != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem cudaEventRecord(start)", record_start));
      }

      const dim3 block(kTile, kTile);
      const dim3 grid(static_cast<unsigned int>((c.cols() + block.x - 1) / block.x),
                      static_cast<unsigned int>((c.rows() + block.y - 1) / block.y));
      SmemKernel<<<grid, block, 0, stream_>>>(device_a_, device_b_, device_c_, m_, n_, k_);

      const cudaError_t launch_status = cudaGetLastError();
      if (launch_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem kernel launch", launch_status));
      }

      const cudaError_t record_stop = cudaEventRecord(stop_event_, stream_);
      if (record_stop != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem cudaEventRecord(stop)", record_stop));
      }
      const cudaError_t event_sync = cudaEventSynchronize(stop_event_);
      if (event_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem cudaEventSynchronize", event_sync));
      }

      float kernel_ms = 0.0F;
      const cudaError_t elapsed_status =
          cudaEventElapsedTime(&kernel_ms, start_event_, stop_event_);
      if (elapsed_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem cudaEventElapsedTime", elapsed_status));
      }

      const cudaError_t copy_out =
          cudaMemcpyAsync(c.data(), device_c_, c_bytes_, cudaMemcpyDeviceToHost, stream_);
      if (copy_out != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem cudaMemcpyAsync D2H", copy_out));
      }
      const cudaError_t stream_sync = cudaStreamSynchronize(stream_);
      if (stream_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem cudaStreamSynchronize", stream_sync));
      }

      const auto wall_finish = std::chrono::steady_clock::now();
      measurement.kernel_ms = static_cast<double>(kernel_ms);
      measurement.e2e_ms =
          std::chrono::duration<double, std::milli>(wall_finish - wall_start).count();
      return Status::Ok();
    }

    Status Initialize() {
      const cudaError_t stream_status = cudaStreamCreate(&stream_);
      if (stream_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem cudaStreamCreate", stream_status));
      }

      const cudaError_t start_status = cudaEventCreate(&start_event_);
      if (start_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem cudaEventCreate(start)", start_status));
      }
      const cudaError_t stop_status = cudaEventCreate(&stop_event_);
      if (stop_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem cudaEventCreate(stop)", stop_status));
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
        return Status::Error(FormatCudaError("cuda_smem cudaMalloc", failure));
      }

      const cudaError_t copy_a =
          cudaMemcpyAsync(device_a_, a_.data(), a_bytes_, cudaMemcpyHostToDevice, stream_);
      const cudaError_t copy_b =
          (copy_a == cudaSuccess)
              ? cudaMemcpyAsync(device_b_, b_.data(), b_bytes_, cudaMemcpyHostToDevice, stream_)
              : copy_a;
      if (copy_a != cudaSuccess || copy_b != cudaSuccess) {
        const cudaError_t failure = (copy_a != cudaSuccess) ? copy_a : copy_b;
        return Status::Error(FormatCudaError("cuda_smem cudaMemcpyAsync H2D", failure));
      }

      const cudaError_t stream_sync = cudaStreamSynchronize(stream_);
      if (stream_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_smem cudaStreamSynchronize", stream_sync));
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
    float* device_a_ = nullptr;
    float* device_b_ = nullptr;
    float* device_c_ = nullptr;
  };
};

} // namespace

BackendPtr MakeCudaSmemBackend() {
  return std::make_unique<CudaSmemBackend>();
}

} // namespace kernellab
