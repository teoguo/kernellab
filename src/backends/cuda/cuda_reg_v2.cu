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

// cuda_reg_v2 Layer 3: cuda_reg's 128x128 register-tiled GEMM plus
// float4 vectorized movement, double-buffered smem, and cp.async
// global-to-smem copies on the hot tile load path.
//
// We intentionally keep the same BM/BN/BK/TM/TN shape as cuda_reg so the
// layer isolates one idea: replace ordinary global loads with the
// sm_80+ async copy pipeline.  Correctness still wins over speed: edge
// tiles use cp.async's zero-fill source-size operand instead of reading
// outside A/B.
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kThreadM = 8;
constexpr int kThreadN = 8;
constexpr int kThreadsPerBlock = 256;
constexpr int kThreadTilesN = kBlockN / kThreadN;
constexpr int kFloat4Width = 4;

// Layer 3 changes A's smem layout from transposed As[bk][row] to
// row-major As[row][bk].  That is the key cp.async enabler: one A row
// contains BK=8 contiguous floats in global memory, so each thread can
// copy one 16-byte half-row directly into smem without unpacking.
//
// A plain stride of BK+1=9 would reduce bank conflicts, but row starts
// would not all be 16-byte aligned, which is unsafe for 16-byte cp.async.
// A stride of 12 floats keeps every row start 16-byte aligned.  The
// extra 4-float group skew makes rows 8 apart land on different banks;
// one warp currently covers two 8-row groups, so this avoids a steady
// two-address/same-bank pattern when loading a_frag.
constexpr int kAsStride = kBlockK + 4;
constexpr int kAsGroupRows = kThreadM;
constexpr int kAsGroupSkew = 4;
constexpr int kAsSmemElements = (kBlockM * kAsStride) + kAsGroupSkew;

__device__ bool IsAligned16(const float* pointer) {
  return (reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4)) == 0;
}

__device__ int ClampCopyBytes(const bool row_or_col_valid, const int remaining_floats) {
  if (!row_or_col_valid || remaining_floats <= 0) {
    return 0;
  }
  const int valid_floats = (remaining_floats < kFloat4Width) ? remaining_floats : kFloat4Width;
  return valid_floats * static_cast<int>(sizeof(float));
}

__device__ unsigned int SharedAddress(const void* pointer) {
  unsigned int address = 0;
  asm("{ .reg .u64 shared_address; cvta.to.shared.u64 shared_address, %1; "
      "cvt.u32.u64 %0, shared_address; }"
      : "=r"(address)
      : "l"(pointer));
  return address;
}

__device__ void CpAsyncCopy16(float* smem_destination, const float* global_source,
                              int valid_bytes) {
  // cp.async copies 16 bytes from global memory to shared memory without
  // first materializing the data in general-purpose registers.  The
  // fourth operand is the number of valid source bytes.  When it is less
  // than 16, the hardware zero-fills the remaining bytes, which is the
  // clean way to keep edge tiles correct.
  const unsigned int shared_destination = SharedAddress(smem_destination);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(shared_destination),
               "l"(global_source), "r"(valid_bytes));
}

__device__ void StorePartialFloat4(float* smem_destination, const float* global_source,
                                   int valid_bytes) {
  // cp.async's fast path requires the source and destination to be
  // aligned for the 16-byte copy.  Odd matrix widths such as K=17 or
  // N=129 make later rows unaligned, so boundary tests must use a scalar
  // zero-fill fallback rather than issuing undefined async copies.
  const int valid_floats = valid_bytes / static_cast<int>(sizeof(float));
#pragma unroll
  for (int lane = 0; lane < kFloat4Width; ++lane) {
    smem_destination[lane] = (lane < valid_floats) ? global_source[lane] : 0.0F;
  }
}

__device__ void CopyFloat4ToSmem(float* smem_destination, const float* global_source,
                                 int valid_bytes) {
  if (valid_bytes == 16 && IsAligned16(global_source)) {
    CpAsyncCopy16(smem_destination, global_source, valid_bytes);
  } else {
    StorePartialFloat4(smem_destination, global_source, valid_bytes);
  }
}

__device__ void CpAsyncCommitGroup() {
  // Commit closes the current batch of cp.async instructions.  Later
  // wait_group calls reason about committed groups, not individual copy
  // instructions.
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ void CpAsyncWaitAll() {
  // wait_group 0 means no committed cp.async group may remain pending.
  // We pair this with __syncthreads() before consuming the smem buffer,
  // because wait_group is per-thread while the tile is cooperative.
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ int AsOffset(const int row, const int bk) {
  const int group_skew = ((row / kAsGroupRows) & 1) * kAsGroupSkew;
  return (row * kAsStride) + group_skew + bk;
}

__device__ void LoadTileCpAsync(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ as_buffer, float* __restrict__ bs_buffer,
                                int block_row, int block_col, int k_tile, int m, int n, int k) {
  const int tid = threadIdx.x;

  // Cooperative A load: BM*BK floats are exactly 256 float4 chunks, so
  // each thread issues one cp.async.  This is why switching As to
  // row-major matters: A[global_row][global_k:global_k+4] is contiguous
  // in global memory and contiguous in shared memory.
  const int a_tile_row = tid / (kBlockK / kFloat4Width);
  const int a_tile_k = (tid % (kBlockK / kFloat4Width)) * kFloat4Width;
  const int a_global_row = block_row + a_tile_row;
  const int a_global_k = k_tile + a_tile_k;
  const bool a_row_valid = a_global_row < m;
  const int a_valid_bytes = ClampCopyBytes(a_row_valid, k - a_global_k);
  const float* a_source =
      (a_valid_bytes > 0) ? (a + (a_global_row * k) + a_global_k) : a;
  CopyFloat4ToSmem(&as_buffer[AsOffset(a_tile_row, a_tile_k)], a_source, a_valid_bytes);

  // Cooperative B load: B is row-major across N, and Bs keeps the same
  // [BK][BN] row-major layout, so the previous float4 path maps directly
  // to one cp.async per thread.
  const int b_tile_k = tid / (kBlockN / kFloat4Width);
  const int b_tile_col = (tid % (kBlockN / kFloat4Width)) * kFloat4Width;
  const int b_global_k = k_tile + b_tile_k;
  const int b_global_col = block_col + b_tile_col;
  const bool b_row_valid = b_global_k < k;
  const int b_valid_bytes = ClampCopyBytes(b_row_valid, n - b_global_col);
  const float* b_source =
      (b_valid_bytes > 0) ? (b + (b_global_k * n) + b_global_col) : b;
  CopyFloat4ToSmem(&bs_buffer[(b_tile_k * kBlockN) + b_tile_col], b_source, b_valid_bytes);
}

__device__ void ComputeTileFromSmem(const float* __restrict__ as_buffer,
                                    const float* __restrict__ bs_buffer, int row_base,
                                    int col_base, float acc[kThreadM][kThreadN]) {
#pragma unroll
    for (int bk = 0; bk < kBlockK; ++bk) {
      float a_frag[kThreadM];
      float b_frag[kThreadN];

      // Layer 3 reads A from row-major As[row][bk].  For a fixed bk, the
      // 16 column-tiles in a half warp all read the same A element, which
      // shared memory can broadcast.  The stride+skew above keeps the two
      // row groups inside one warp from repeatedly landing on identical
      // banks.
#pragma unroll
      for (int tm = 0; tm < kThreadM; ++tm) {
        a_frag[tm] = as_buffer[AsOffset(row_base + tm, bk)];
      }

      // B remains contiguous across N, so each thread still uses two
      // float4 smem reads for its 8-column fragment.
      const float4 b0 =
          *reinterpret_cast<const float4*>(&bs_buffer[(bk * kBlockN) + col_base + 0]);
      const float4 b1 =
          *reinterpret_cast<const float4*>(&bs_buffer[(bk * kBlockN) + col_base + 4]);
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

__global__ void RegV2CpAsyncKernel(const float* __restrict__ a,
                                   const float* __restrict__ b, float* __restrict__ c, int m,
                                   int n, int k) {
  // Layer 3 still owns two copies of each smem tile, but global-to-smem
  // movement now uses cp.async.  A is flat because each row has an
  // explicit padded/skewed offset rather than a simple rectangular C
  // array layout.
  __shared__ __align__(16) float as[2][kAsSmemElements];
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

  // Prologue: issue the first async copy group, wait for it, then use a
  // CTA-wide barrier.  cp.async wait is per-thread; __syncthreads makes
  // the cooperatively filled tile visible to every thread.
  LoadTileCpAsync(a, b, &as[0][0], &bs[0][0][0], block_row, block_col, 0, m, n, k);
  CpAsyncCommitGroup();
  CpAsyncWaitAll();
  __syncthreads();

  int current_buffer = 0;
  for (int k_tile = 0; k_tile < k; k_tile += kBlockK) {
    const int next_k_tile = k_tile + kBlockK;
    const int next_buffer = 1 - current_buffer;

    // Pipeline body: issue async copies for K+BK into the opposite
    // buffer, commit that group, then compute the current buffer while
    // the copy engine moves the next tile.  This is the difference from
    // Layer 2: the load is not a normal register-producing instruction.
    if (next_k_tile < k) {
      LoadTileCpAsync(a, b, &as[next_buffer][0], &bs[next_buffer][0][0], block_row, block_col,
                      next_k_tile, m, n, k);
      CpAsyncCommitGroup();
    }

    ComputeTileFromSmem(&as[current_buffer][0], &bs[current_buffer][0][0], row_base, col_base,
                        acc);

    // Wait only when a next tile was issued.  wait_group 0 proves the
    // async group has finished; the barrier proves all threads are done
    // reading current_buffer before the next loop can reuse it.
    if (next_k_tile < k) {
      CpAsyncWaitAll();
      __syncthreads();
    }
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
      RegV2CpAsyncKernel<<<grid, block, 0, stream_>>>(device_a_, device_b_, device_c_, m_, n_,
                                                      k_);

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
