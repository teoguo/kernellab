#include "kernellab/backend/cuda_support.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace kernellab {

namespace {

std::string FormatCudaError(const char* operation, const cudaError_t status) {
  return std::string(operation) + " failed: " + cudaGetErrorString(status);
}

// cuda_reg_v2 Layer 2: cuda_reg's 128x128 register-tiled GEMM plus
// float4 vectorized movement and ordinary-load shared-memory double
// buffering.
//
// We intentionally keep the same BM/BN/BK/TM/TN shape as cuda_reg so
// this layer isolates one idea: reduce load instruction count and make
// aligned 128-bit transactions where the data layout naturally permits
// them.  Correctness still wins over vectorization: edge tiles and
// unaligned rows fall back to scalar loads/stores.
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kThreadM = 8;
constexpr int kThreadN = 8;
constexpr int kThreadsPerBlock = 256;
constexpr int kThreadTilesN = kBlockN / kThreadN;
constexpr int kFloat4Width = 4;
constexpr int kAsStride = kBlockM + 4;

__device__ bool IsAligned16(const float* pointer) {
  return (reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4)) == 0;
}

__device__ void StoreFloat4(float* pointer, const float4 value) {
  *reinterpret_cast<float4*>(pointer) = value;
}

__device__ void LoadTileFloat4(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ as_buffer, float* __restrict__ bs_buffer,
                               int block_row, int block_col, int k_tile, int m, int n, int k) {
  const int tid = threadIdx.x;

  // Cooperative A load, vectorized across the K dimension.  Each thread
  // owns one float4 from a row of A because BM*BK/4 = 256.  A is still
  // stored transposed in smem, so the float4 global load is unpacked into
  // four scalar stores to As[tile_k + lane][tile_row].
  for (int linear = tid; linear < (kBlockM * kBlockK) / kFloat4Width;
       linear += kThreadsPerBlock) {
      const int tile_row = linear / (kBlockK / kFloat4Width);
      const int tile_k = (linear % (kBlockK / kFloat4Width)) * kFloat4Width;
      const int global_row = block_row + tile_row;
      const int global_k = k_tile + tile_k;
      float4 packed{0.0F, 0.0F, 0.0F, 0.0F};
      if (global_row < m) {
        const float* source = a + (global_row * k) + global_k;
        if ((global_k + 3) < k && IsAligned16(source)) {
          packed = *reinterpret_cast<const float4*>(source);
        } else {
          // Partial K tiles, such as K=17, still contain valid lanes.
          // Load each in-range lane rather than zeroing the whole vector.
          packed.x = (global_k + 0 < k) ? source[0] : 0.0F;
          packed.y = (global_k + 1 < k) ? source[1] : 0.0F;
          packed.z = (global_k + 2 < k) ? source[2] : 0.0F;
          packed.w = (global_k + 3 < k) ? source[3] : 0.0F;
        }
      }
      as_buffer[((tile_k + 0) * kAsStride) + tile_row] = packed.x;
      as_buffer[((tile_k + 1) * kAsStride) + tile_row] = packed.y;
      as_buffer[((tile_k + 2) * kAsStride) + tile_row] = packed.z;
      as_buffer[((tile_k + 3) * kAsStride) + tile_row] = packed.w;
  }

  // Cooperative B load, vectorized across N.  Bs is row-major [BK][BN],
  // so a float4 from global B can be written as a float4 into smem.
  for (int linear = tid; linear < (kBlockK * kBlockN) / kFloat4Width;
       linear += kThreadsPerBlock) {
      const int tile_k = linear / (kBlockN / kFloat4Width);
      const int tile_col = (linear % (kBlockN / kFloat4Width)) * kFloat4Width;
      const int global_k = k_tile + tile_k;
      const int global_col = block_col + tile_col;
      float4 packed{0.0F, 0.0F, 0.0F, 0.0F};
      if (global_k < k) {
        const float* source = b + (global_k * n) + global_col;
        if ((global_col + 3) < n && IsAligned16(source)) {
          packed = *reinterpret_cast<const float4*>(source);
        } else {
          // Partial N edge tiles keep their valid lanes.  This is slower
          // than float4 but only happens on boundary blocks.
          packed.x = (global_col + 0 < n) ? source[0] : 0.0F;
          packed.y = (global_col + 1 < n) ? source[1] : 0.0F;
          packed.z = (global_col + 2 < n) ? source[2] : 0.0F;
          packed.w = (global_col + 3 < n) ? source[3] : 0.0F;
        }
      }
      StoreFloat4(&bs_buffer[(tile_k * kBlockN) + tile_col], packed);
  }
}

__device__ void ComputeTileFromSmem(const float* __restrict__ as_buffer,
                                    const float* __restrict__ bs_buffer, int row_base,
                                    int col_base, float acc[kThreadM][kThreadN]) {
#pragma unroll
    for (int bk = 0; bk < kBlockK; ++bk) {
      float a_frag[kThreadM];
      float b_frag[kThreadN];

      // Pull one A column fragment and one B row fragment from smem into
      // registers using two float4 loads each.  row_base and col_base are
      // multiples of 8, and kAsStride is 132, so these shared-memory
      // addresses are 16-byte aligned.  The fragment is then unpacked
      // into scalar arrays to keep the outer-product loop readable.
      const float4 a0 =
          *reinterpret_cast<const float4*>(&as_buffer[(bk * kAsStride) + row_base + 0]);
      const float4 a1 =
          *reinterpret_cast<const float4*>(&as_buffer[(bk * kAsStride) + row_base + 4]);
      const float4 b0 =
          *reinterpret_cast<const float4*>(&bs_buffer[(bk * kBlockN) + col_base + 0]);
      const float4 b1 =
          *reinterpret_cast<const float4*>(&bs_buffer[(bk * kBlockN) + col_base + 4]);
      a_frag[0] = a0.x;
      a_frag[1] = a0.y;
      a_frag[2] = a0.z;
      a_frag[3] = a0.w;
      a_frag[4] = a1.x;
      a_frag[5] = a1.y;
      a_frag[6] = a1.z;
      a_frag[7] = a1.w;
      b_frag[0] = b0.x;
      b_frag[1] = b0.y;
      b_frag[2] = b0.z;
      b_frag[3] = b0.w;
      b_frag[4] = b1.x;
      b_frag[5] = b1.y;
      b_frag[6] = b1.z;
      b_frag[7] = b1.w;

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
}

__global__ void RegV2DoubleBufferedKernel(const float* __restrict__ a,
                                          const float* __restrict__ b, float* __restrict__ c,
                                          int m, int n, int k) {
  // Layer 2 owns two copies of each smem tile.  While buffer 0 is being
  // consumed by the outer-product compute loop, buffer 1 can hold the
  // next K tile, and vice versa.  This doubles smem from ~8 KB to ~16 KB
  // but keeps the same register-tiled math and timing contract.
  __shared__ __align__(16) float as[2][kBlockK][kAsStride];
  __shared__ __align__(16) float bs[2][kBlockK][kBlockN];

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * kBlockM;
  const int block_col = blockIdx.x * kBlockN;

  // Map the 1D thread id to a 16x16 grid of 8x8 micro-tiles.
  const int thread_tile_row = tid / kThreadTilesN;
  const int thread_tile_col = tid % kThreadTilesN;
  const int row_base = thread_tile_row * kThreadM;
  const int col_base = thread_tile_col * kThreadN;

  float acc[kThreadM][kThreadN];
#pragma unroll
  for (int tm = 0; tm < kThreadM; ++tm) {
#pragma unroll
    for (int tn = 0; tn < kThreadN; ++tn) {
      acc[tm][tn] = 0.0F;
    }
  }

  // Prologue: every thread cooperatively loads the first K tile into
  // buffer 0.  The barrier makes that buffer visible before compute.
  LoadTileFloat4(a, b, &as[0][0][0], &bs[0][0][0], block_row, block_col, 0, m, n, k);
  __syncthreads();

  int current_buffer = 0;
  for (int k_tile = 0; k_tile < k; k_tile += kBlockK) {
    const int next_k_tile = k_tile + kBlockK;
    const int next_buffer = 1 - current_buffer;

    // Ordinary global loads are still synchronous inside a thread, but
    // putting the next tile in the opposite buffer before computing the
    // current tile lets the scheduler overlap some load stalls from one
    // warp with arithmetic from another warp.  The following barrier is
    // the handoff: after compute, all next-buffer loads must be visible.
    if (next_k_tile < k) {
      LoadTileFloat4(a, b, &as[next_buffer][0][0], &bs[next_buffer][0][0], block_row, block_col,
                     next_k_tile, m, n, k);
    }

    ComputeTileFromSmem(&as[current_buffer][0][0], &bs[current_buffer][0][0], row_base, col_base,
                        acc);

    // Do not let any warp start consuming next_buffer until all warps
    // have finished loading it.  The same barrier also prevents the next
    // loop iteration from overwriting current_buffer too early.
    __syncthreads();
    current_buffer = next_buffer;
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

class CudaRegV2Backend final : public Backend {
private:
  class CudaRegV2Session;

public:
  std::string_view id() const noexcept override {
    return "cuda_reg_v2";
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
    auto fresh = std::make_unique<CudaRegV2Session>(a, b);
    const auto init_status = fresh->Initialize();
    if (!init_status.ok) {
      return init_status;
    }
    session = std::move(fresh);
    return Status::Ok();
  }

private:
  class CudaRegV2Session final : public BackendSession {
  public:
    CudaRegV2Session(const Matrix& a, const Matrix& b)
        : a_(a), b_(b), m_(static_cast<int>(a.rows())), n_(static_cast<int>(b.cols())),
          k_(static_cast<int>(a.cols())), a_bytes_(CudaFloatMatrixBytes(a.rows(), a.cols())),
          b_bytes_(CudaFloatMatrixBytes(b.rows(), b.cols())),
          c_bytes_(CudaFloatMatrixBytes(a.rows(), b.cols())) {}

    ~CudaRegV2Session() override {
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
        return Status::Error("cuda_reg_v2 output matrix has wrong shape");
      }

      const auto wall_start = std::chrono::steady_clock::now();
      const cudaError_t record_start = cudaEventRecord(start_event_, stream_);
      if (record_start != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaEventRecord(start)", record_start));
      }

      const dim3 block(kThreadsPerBlock);
      const dim3 grid(static_cast<unsigned int>((c.cols() + kBlockN - 1) / kBlockN),
                      static_cast<unsigned int>((c.rows() + kBlockM - 1) / kBlockM));
      RegV2DoubleBufferedKernel<<<grid, block, 0, stream_>>>(device_a_, device_b_, device_c_, m_,
                                                             n_, k_);

      const cudaError_t launch_status = cudaGetLastError();
      if (launch_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg_v2 kernel launch", launch_status));
      }

      const cudaError_t record_stop = cudaEventRecord(stop_event_, stream_);
      if (record_stop != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaEventRecord(stop)", record_stop));
      }
      const cudaError_t event_sync = cudaEventSynchronize(stop_event_);
      if (event_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaEventSynchronize", event_sync));
      }

      float kernel_ms = 0.0F;
      const cudaError_t elapsed_status =
          cudaEventElapsedTime(&kernel_ms, start_event_, stop_event_);
      if (elapsed_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaEventElapsedTime", elapsed_status));
      }

      const cudaError_t copy_out =
          cudaMemcpyAsync(c.data(), device_c_, c_bytes_, cudaMemcpyDeviceToHost, stream_);
      if (copy_out != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaMemcpyAsync D2H", copy_out));
      }
      const cudaError_t stream_sync = cudaStreamSynchronize(stream_);
      if (stream_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaStreamSynchronize", stream_sync));
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
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaStreamCreate", stream_status));
      }

      const cudaError_t start_status = cudaEventCreate(&start_event_);
      if (start_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaEventCreate(start)", start_status));
      }
      const cudaError_t stop_status = cudaEventCreate(&stop_event_);
      if (stop_status != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaEventCreate(stop)", stop_status));
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
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaMalloc", failure));
      }

      const cudaError_t copy_a =
          cudaMemcpyAsync(device_a_, a_.data(), a_bytes_, cudaMemcpyHostToDevice, stream_);
      const cudaError_t copy_b =
          (copy_a == cudaSuccess)
              ? cudaMemcpyAsync(device_b_, b_.data(), b_bytes_, cudaMemcpyHostToDevice, stream_)
              : copy_a;
      if (copy_a != cudaSuccess || copy_b != cudaSuccess) {
        const cudaError_t failure = (copy_a != cudaSuccess) ? copy_a : copy_b;
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaMemcpyAsync H2D", failure));
      }

      const cudaError_t stream_sync = cudaStreamSynchronize(stream_);
      if (stream_sync != cudaSuccess) {
        return Status::Error(FormatCudaError("cuda_reg_v2 cudaStreamSynchronize", stream_sync));
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

BackendPtr MakeCudaRegV2Backend() {
  return std::make_unique<CudaRegV2Backend>();
}

} // namespace kernellab
