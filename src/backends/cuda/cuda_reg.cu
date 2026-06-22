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

// cuda_reg: 128x128 block tile + 8x8 per-thread register tile.
//
// One block computes a 128x128 tile of C.  The block has 256 threads;
// each thread owns one 8x8 micro-tile, so 16x16 thread tiles cover the
// whole block tile.  This is the next rung after cuda_smem: instead of
// one thread producing one C element, each thread keeps 64 partial sums
// in registers and reuses each smem operand across a small outer product.
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kThreadM = 8;
constexpr int kThreadN = 8;
constexpr int kThreadsPerBlock = 256;
constexpr int kThreadTilesN = kBlockN / kThreadN;

__global__ void RegTiledKernel(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ c, int m, int n, int k) {
  // As is stored transposed: original A tile [BM][BK] becomes [BK][BM].
  // In the compute loop, all threads read a fixed bk row and nearby M
  // columns, which turns the per-thread A fragment into bank-friendly
  // row reads.  The +1 padding breaks store/load bank-alignment patterns
  // caused by BM=128 being a multiple of 32 banks.
  __shared__ float as[kBlockK][kBlockM + 1];

  // Bs is already consumed as [BK][BN], so row-major shared memory keeps
  // both cooperative loads and compute-time reads simple and coalesced.
  __shared__ float bs[kBlockK][kBlockN];

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * kBlockM;
  const int block_col = blockIdx.x * kBlockN;

  // Map the 1D thread id to a 16x16 grid of micro-tiles.  Consecutive
  // thread ids move across N first, matching CUDA's warp layout and
  // giving adjacent threads adjacent B/C columns.
  const int thread_tile_row = tid / kThreadTilesN;
  const int thread_tile_col = tid % kThreadTilesN;
  const int row_base = thread_tile_row * kThreadM;
  const int col_base = thread_tile_col * kThreadN;

  // 64 fp32 accumulators live in registers.  This is the whole point of
  // register tiling: after one A value and one B value are fetched from
  // smem, they are reused for an 8x8 outer product before touching C.
  float acc[kThreadM][kThreadN];
#pragma unroll
  for (int tm = 0; tm < kThreadM; ++tm) {
#pragma unroll
    for (int tn = 0; tn < kThreadN; ++tn) {
      acc[tm][tn] = 0.0F;
    }
  }

  for (int k_tile = 0; k_tile < k; k_tile += kBlockK) {
    // Cooperative A load.  We iterate the original [BM][BK] tile in
    // row-major order so each warp reads short contiguous chunks from A.
    // The store is transposed to as[bk][row] for bank-friendly compute.
    for (int linear = tid; linear < kBlockM * kBlockK; linear += kThreadsPerBlock) {
      const int tile_row = linear / kBlockK;
      const int tile_k = linear % kBlockK;
      const int global_row = block_row + tile_row;
      const int global_k = k_tile + tile_k;
      as[tile_k][tile_row] =
          (global_row < m && global_k < k) ? a[(global_row * k) + global_k] : 0.0F;
    }

    // Cooperative B load.  B's tile layout [BK][BN] matches global
    // row-major order for each bk row, so consecutive threads load and
    // store adjacent columns.
    for (int linear = tid; linear < kBlockK * kBlockN; linear += kThreadsPerBlock) {
      const int tile_k = linear / kBlockN;
      const int tile_col = linear % kBlockN;
      const int global_k = k_tile + tile_k;
      const int global_col = block_col + tile_col;
      bs[tile_k][tile_col] =
          (global_k < k && global_col < n) ? b[(global_k * n) + global_col] : 0.0F;
    }

    // Every thread must wait until the whole A/B K-slice is resident in
    // shared memory before any thread starts reusing it for its 8x8 tile.
    __syncthreads();

#pragma unroll
    for (int bk = 0; bk < kBlockK; ++bk) {
      float a_frag[kThreadM];
      float b_frag[kThreadN];

      // Pull one A column fragment and one B row fragment from smem into
      // registers.  Each a_frag value is reused across TN columns; each
      // b_frag value is reused across TM rows.
#pragma unroll
      for (int tm = 0; tm < kThreadM; ++tm) {
        a_frag[tm] = as[bk][row_base + tm];
      }
#pragma unroll
      for (int tn = 0; tn < kThreadN; ++tn) {
        b_frag[tn] = bs[bk][col_base + tn];
      }

      // Register-resident outer product.  These 64 FMAs are why smem
      // arithmetic intensity rises: one A smem read feeds 8 columns, and
      // one B smem read feeds 8 rows.
#pragma unroll
      for (int tm = 0; tm < kThreadM; ++tm) {
#pragma unroll
        for (int tn = 0; tn < kThreadN; ++tn) {
          acc[tm][tn] += a_frag[tm] * b_frag[tn];
        }
      }
    }

    // The same shared-memory buffers are reused for the next K-slice.
    // This barrier prevents a fast warp from overwriting As/Bs while a
    // slower warp is still consuming the current slice.
    __syncthreads();
  }

  // Store the 8x8 micro-tile.  Guards make non-multiple M/N/K sizes
  // correct; out-of-range A/B elements were zero-padded during loads.
#pragma unroll
  for (int tm = 0; tm < kThreadM; ++tm) {
    const int global_row = block_row + row_base + tm;
    if (global_row < m) {
#pragma unroll
      for (int tn = 0; tn < kThreadN; ++tn) {
        const int global_col = block_col + col_base + tn;
        if (global_col < n) {
          c[(global_row * n) + global_col] = acc[tm][tn];
        }
      }
    }
  }
}

class CudaRegBackend final : public Backend {
private:
  class CudaRegSession;

public:
  std::string_view id() const noexcept override {
    return "cuda_reg";
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
    auto fresh = std::make_unique<CudaRegSession>(a, b);
    const auto init_status = fresh->Initialize();
    if (!init_status.ok) {
      return init_status;
    }
    session = std::move(fresh);
    return Status::Ok();
  }

private:
  class CudaRegSession final : public BackendSession {
  public:
    CudaRegSession(const Matrix& a, const Matrix& b)
        : a_(a), b_(b), m_(static_cast<int>(a.rows())), n_(static_cast<int>(b.cols())),
          k_(static_cast<int>(a.cols())), a_bytes_(CudaFloatMatrixBytes(a.rows(), a.cols())),
          b_bytes_(CudaFloatMatrixBytes(b.rows(), b.cols())),
          c_bytes_(CudaFloatMatrixBytes(a.rows(), b.cols())) {}

    ~CudaRegSession() override {
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
      // Prepare()/Initialize() already did allocation and H2D copies.
      // Run() keeps the same timing contract as cuda_naive/cuda_smem:
      // cudaEvent measures only the kernel, while e2e also includes D2H.
      if (c.rows() != a_.rows() || c.cols() != b_.cols()) {
        return Status::Error("cuda_reg output matrix has wrong shape");
      }

      const auto wall_start = std::chrono::steady_clock::now();
      const cudaError_t record_start = cudaEventRecord(start_event_, stream_);
      if (record_start != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg cudaEventRecord(start)", record_start));
      }

      const dim3 block(kThreadsPerBlock);
      const dim3 grid(static_cast<unsigned int>((c.cols() + kBlockN - 1) / kBlockN),
                      static_cast<unsigned int>((c.rows() + kBlockM - 1) / kBlockM));
      RegTiledKernel<<<grid, block, 0, stream_>>>(device_a_, device_b_, device_c_, m_, n_, k_);

      const cudaError_t launch_status = cudaGetLastError();
      if (launch_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg kernel launch", launch_status));
      }

      const cudaError_t record_stop = cudaEventRecord(stop_event_, stream_);
      if (record_stop != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg cudaEventRecord(stop)", record_stop));
      }
      const cudaError_t event_sync = cudaEventSynchronize(stop_event_);
      if (event_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg cudaEventSynchronize", event_sync));
      }

      float kernel_ms = 0.0F;
      const cudaError_t elapsed_status =
          cudaEventElapsedTime(&kernel_ms, start_event_, stop_event_);
      if (elapsed_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg cudaEventElapsedTime", elapsed_status));
      }

      const cudaError_t copy_out =
          cudaMemcpyAsync(c.data(), device_c_, c_bytes_, cudaMemcpyDeviceToHost, stream_);
      if (copy_out != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg cudaMemcpyAsync D2H", copy_out));
      }
      const cudaError_t stream_sync = cudaStreamSynchronize(stream_);
      if (stream_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg cudaStreamSynchronize", stream_sync));
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
        return Status::Error(FormatCudaError("cuda_reg cudaStreamCreate", stream_status));
      }

      const cudaError_t start_status = cudaEventCreate(&start_event_);
      if (start_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg cudaEventCreate(start)", start_status));
      }
      const cudaError_t stop_status = cudaEventCreate(&stop_event_);
      if (stop_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg cudaEventCreate(stop)", stop_status));
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
        return Status::Error(FormatCudaError("cuda_reg cudaMalloc", failure));
      }

      const cudaError_t copy_a =
          cudaMemcpyAsync(device_a_, a_.data(), a_bytes_, cudaMemcpyHostToDevice, stream_);
      const cudaError_t copy_b =
          (copy_a == cudaSuccess)
              ? cudaMemcpyAsync(device_b_, b_.data(), b_bytes_, cudaMemcpyHostToDevice, stream_)
              : copy_a;
      if (copy_a != cudaSuccess || copy_b != cudaSuccess) {
        const cudaError_t failure = (copy_a != cudaSuccess) ? copy_a : copy_b;
        return Status::Error(FormatCudaError("cuda_reg cudaMemcpyAsync H2D", failure));
      }

      const cudaError_t stream_sync = cudaStreamSynchronize(stream_);
      if (stream_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg cudaStreamSynchronize", stream_sync));
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

BackendPtr MakeCudaRegBackend() {
  return std::make_unique<CudaRegBackend>();
}

} // namespace kernellab
