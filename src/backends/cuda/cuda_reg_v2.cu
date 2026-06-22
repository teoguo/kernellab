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

// Autotune knobs.  The defaults are a sensible occupancy-first candidate,
// but the remote sweep overrides these with nvcc -D flags so every
// configuration is built as a separate compile-time specialization.
#ifndef KLAB_REG_V2_BM
#define KLAB_REG_V2_BM 128
#endif
#ifndef KLAB_REG_V2_BN
#define KLAB_REG_V2_BN 128
#endif
#ifndef KLAB_REG_V2_BK
#define KLAB_REG_V2_BK 16
#endif
#ifndef KLAB_REG_V2_TM
#define KLAB_REG_V2_TM 8
#endif
#ifndef KLAB_REG_V2_TN
#define KLAB_REG_V2_TN 8
#endif
#ifndef KLAB_REG_V2_WARPTILE
#define KLAB_REG_V2_WARPTILE 1
#endif

constexpr int kBlockM = KLAB_REG_V2_BM;
constexpr int kBlockN = KLAB_REG_V2_BN;
constexpr int kBlockK = KLAB_REG_V2_BK;
constexpr int kThreadM = KLAB_REG_V2_TM;
constexpr int kThreadN = KLAB_REG_V2_TN;
constexpr int kFloat4Width = 4;
constexpr int kThreadsPerBlock = (kBlockM * kBlockN) / (kThreadM * kThreadN);

static_assert(kBlockM % kThreadM == 0, "BM must be divisible by TM");
static_assert(kBlockN % kThreadN == 0, "BN must be divisible by TN");
static_assert(kBlockK % kFloat4Width == 0, "BK must be divisible by 4 for float4 loads");
static_assert(kThreadM % kFloat4Width == 0, "TM must be 4 or 8 in this sweep");
static_assert(kThreadN % kFloat4Width == 0, "TN must be 4 or 8 in this sweep");
static_assert(kThreadsPerBlock >= 128 && kThreadsPerBlock <= 512,
              "autotune grid keeps blocks between 128 and 512 threads");

__device__ bool IsAligned16(const float* pointer) {
  return (reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4)) == 0;
}

template <int BM, int BN, int BK, int TM, int TN>
struct RegV2Config {
  static constexpr int kThreads = (BM * BN) / (TM * TN);
  static constexpr int kThreadTilesN = BN / TN;
  // As is intentionally transposed: As[bk][row].  That keeps each
  // thread's A fragment contiguous during compute, which was the winning
  // L1 float4 design.  +4 preserves 16-byte alignment for float4 smem
  // reads while avoiding a plain 128-float bank stride.
  static constexpr int kAsStride = BM + 4;
};

template <int BM, int BN, int BK, int TM, int TN>
__device__ void LoadTileFloat4(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ as_buffer, float* __restrict__ bs_buffer,
                               int block_row, int block_col, int k_tile, int m, int n, int k) {
  using Config = RegV2Config<BM, BN, BK, TM, TN>;
  const int tid = threadIdx.x;

  // A is row-major in global memory, so a thread reads a contiguous
  // float4 along K.  It then scatters those four lanes into transposed
  // shared memory As[bk][row].  This scatter is the price we pay to make
  // compute-side A reads contiguous and cheap.
  for (int linear = tid; linear < (BM * BK) / kFloat4Width; linear += Config::kThreads) {
    const int tile_row = linear / (BK / kFloat4Width);
    const int tile_k = (linear % (BK / kFloat4Width)) * kFloat4Width;
    const int global_row = block_row + tile_row;
    const int global_k = k_tile + tile_k;
    float4 packed{0.0F, 0.0F, 0.0F, 0.0F};
    if (global_row < m) {
      const float* source = a + (global_row * k) + global_k;
      if ((global_k + 3) < k && IsAligned16(source)) {
        packed = *reinterpret_cast<const float4*>(source);
      } else {
        packed.x = (global_k + 0 < k) ? source[0] : 0.0F;
        packed.y = (global_k + 1 < k) ? source[1] : 0.0F;
        packed.z = (global_k + 2 < k) ? source[2] : 0.0F;
        packed.w = (global_k + 3 < k) ? source[3] : 0.0F;
      }
    }
    as_buffer[((tile_k + 0) * Config::kAsStride) + tile_row] = packed.x;
    as_buffer[((tile_k + 1) * Config::kAsStride) + tile_row] = packed.y;
    as_buffer[((tile_k + 2) * Config::kAsStride) + tile_row] = packed.z;
    as_buffer[((tile_k + 3) * Config::kAsStride) + tile_row] = packed.w;
  }

  // B stays row-major in shared memory: Bs[bk][col].  Global B is also
  // contiguous along N, so full tiles use a direct float4 load/store.
  for (int linear = tid; linear < (BK * BN) / kFloat4Width; linear += Config::kThreads) {
    const int tile_k = linear / (BN / kFloat4Width);
    const int tile_col = (linear % (BN / kFloat4Width)) * kFloat4Width;
    const int global_k = k_tile + tile_k;
    const int global_col = block_col + tile_col;
    float4 packed{0.0F, 0.0F, 0.0F, 0.0F};
    if (global_k < k) {
      const float* source = b + (global_k * n) + global_col;
      if ((global_col + 3) < n && IsAligned16(source)) {
        packed = *reinterpret_cast<const float4*>(source);
      } else {
        packed.x = (global_col + 0 < n) ? source[0] : 0.0F;
        packed.y = (global_col + 1 < n) ? source[1] : 0.0F;
        packed.z = (global_col + 2 < n) ? source[2] : 0.0F;
        packed.w = (global_col + 3 < n) ? source[3] : 0.0F;
      }
    }
    *reinterpret_cast<float4*>(&bs_buffer[(tile_k * BN) + tile_col]) = packed;
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__device__ void ComputeTileFromSmem(const float* __restrict__ as_buffer,
                                    const float* __restrict__ bs_buffer, int row_base,
                                    int col_base, float acc[TM][TN]) {
  using Config = RegV2Config<BM, BN, BK, TM, TN>;
#pragma unroll
  for (int bk = 0; bk < BK; ++bk) {
    float a_frag[TM];
    float b_frag[TN];

    // Fragment loads stay vectorized for all sweeped TM/TN values
    // (4 or 8).  This keeps the L1 float4 benefit while the sweep changes
    // accumulator count and therefore register pressure.
#pragma unroll
    for (int tm = 0; tm < TM; tm += kFloat4Width) {
      const float4 packed =
          *reinterpret_cast<const float4*>(&as_buffer[(bk * Config::kAsStride) + row_base + tm]);
      a_frag[tm + 0] = packed.x;
      a_frag[tm + 1] = packed.y;
      a_frag[tm + 2] = packed.z;
      a_frag[tm + 3] = packed.w;
    }
#pragma unroll
    for (int tn = 0; tn < TN; tn += kFloat4Width) {
      const float4 packed = *reinterpret_cast<const float4*>(&bs_buffer[(bk * BN) + col_base + tn]);
      b_frag[tn + 0] = packed.x;
      b_frag[tn + 1] = packed.y;
      b_frag[tn + 2] = packed.z;
      b_frag[tn + 3] = packed.w;
    }

#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
        acc[tm][tn] += a_frag[tm] * b_frag[tn];
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(RegV2Config<BM, BN, BK, TM, TN>::kThreads, 2)
    void RegV2AutotunedKernel(const float* __restrict__ a, const float* __restrict__ b,
                              float* __restrict__ c, int m, int n, int k) {
  using Config = RegV2Config<BM, BN, BK, TM, TN>;
  __shared__ __align__(16) float as[BK][Config::kAsStride];
  __shared__ __align__(16) float bs[BK][BN];

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
#if KLAB_REG_V2_WARPTILE
  // Optional controlled warptiling experiment.  It keeps the same
  // transposed-As/float4 compute path and only changes ownership of
  // thread tiles: eight warps cover a 2x4 grid of 64x32 warp tiles.
  // This is deliberately guarded by a macro because it must earn its
  // place by benchmark without pushing registers past the 2-block/SM
  // budget.
  static_assert(BM == 128 && BN == 128 && TM == 8 && TN == 8,
                "current warptile mapping is defined for the 128x128 8x8 finalist");
  constexpr int kWarpSize = 32;
  constexpr int kWarpTilesM = 2;
  constexpr int kWarpTilesN = 4;
  constexpr int kWarpTileM = BM / kWarpTilesM;
  constexpr int kWarpTileN = BN / kWarpTilesN;
  constexpr int kWarpLaneTilesN = kWarpTileN / TN;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int warp_tile_row = warp_id / kWarpTilesN;
  const int warp_tile_col = warp_id % kWarpTilesN;
  const int lane_tile_row = lane_id / kWarpLaneTilesN;
  const int lane_tile_col = lane_id % kWarpLaneTilesN;
  const int row_base = (warp_tile_row * kWarpTileM) + (lane_tile_row * TM);
  const int col_base = (warp_tile_col * kWarpTileN) + (lane_tile_col * TN);
#else
  const int thread_tile_row = tid / Config::kThreadTilesN;
  const int thread_tile_col = tid % Config::kThreadTilesN;
  const int row_base = thread_tile_row * TM;
  const int col_base = thread_tile_col * TN;
#endif

  float acc[TM][TN];
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      acc[tm][tn] = 0.0F;
    }
  }

  for (int k_tile = 0; k_tile < k; k_tile += BK) {
    LoadTileFloat4<BM, BN, BK, TM, TN>(a, b, &as[0][0], &bs[0][0], block_row, block_col, k_tile,
                                       m, n, k);
    __syncthreads();

    ComputeTileFromSmem<BM, BN, BK, TM, TN>(&as[0][0], &bs[0][0], row_base, col_base, acc);
    __syncthreads();
  }

  // Store this thread's TMxTN micro-tile.  Guards make non-multiple M/N/K sizes
  // correct; out-of-range A/B elements were zero-padded during loads.
#pragma unroll
  for (int tm = 0; tm < TM; ++tm) {
    const int global_row = block_row + row_base + tm;
    if (global_row < m) {
#pragma unroll
      for (int tn = 0; tn < TN; ++tn) {
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
      RegV2AutotunedKernel<kBlockM, kBlockN, kBlockK, kThreadM, kThreadN>
          <<<grid, block, 0, stream_>>>(device_a_, device_b_, device_c_, m_, n_, k_);

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
