# kernellab

A C++17 / CUDA testbed for reproducible fp32 GEMM measurement: kernel
time is separated from end-to-end time, correctness is checked against
an fp64-accumulating CPU oracle, allocation / H2D / handle setup live in
`Backend::Prepare()`, and reports use a stable versioned schema. The
CUDA backends form a step-by-step optimization path from a naive kernel
to shared-memory tiling, register tiling, autotuning, and cuBLAS.

## Backends

| Backend       | Idea                                                                              |
| ------------- | --------------------------------------------------------------------------------- |
| `cpu_ref`     | Triple-nested loop accumulating in `double`. **Correctness oracle**, not a baseline. |
| `cpu_naive`   | Same shape, fp32 throughout. Scalar CPU baseline.                                 |
| `cpu_omp`     | Adds `#pragma omp parallel for collapse(2)`. Built only when OpenMP is found.     |
| `cuda_naive`  | One thread per output element. No reuse, no shared memory.                        |
| `cuda_smem`   | 32×32 shared-memory tile + per-thread accumulator. Bank-conflict-free layout.     |
| `cuda_reg`    | 128×128 block tile + 8×8 per-thread register tile. FP32 CUDA, no Tensor Cores.    |
| `cuda_reg_v2` | Autotuned register-tiled CUDA SGEMM. Current default: BK=16 + warp-tiled mapping. |
| `cublas`      | `cublasSgemm` on a persistent handle, pinned to `CUBLAS_PEDANTIC_MATH`. Production fp32 reference. |

CPU-only builds work by default when CUDA is unavailable. CUDA backends are enabled automatically when CMake finds a usable CUDA toolchain and `KERNELLAB_ENABLE_CUDA=ON`.

## Validated Environments

- CPU-only macOS development build with AppleClang 17 and CMake 4.3.1.
- Linux `x86_64` CUDA build on the perf-test host: CUDA 12.1, CMake
  3.28.3, `g++ 11.4.0`, RTX 6000 Ada (sm_89). Build paths in
  `scripts/build_cuda.sh` accept any `CUDA_ROOT` (defaults to
  `/usr/local/cuda-13.0`; override via env var as shown in the build
  section).

On systems where CUDA is installed but `nvcc` is not on `PATH`, pass an explicit CUDA compiler path during configure or use the helper script below.

## Results

Measured on **NVIDIA RTX 6000 Ada Generation** (sm_89, 142 SMs, 48 GB ECC
GDDR6, ~91 TFLOPs theoretical fp32 peak), Linux + CUDA 12.1, AMD Ryzen
9 7950X host, Release build. cuBLAS is pinned to
`CUBLAS_PEDANTIC_MATH` so the denominator is a strict fp32 SGEMM
baseline. Reproduce with the [CLI examples below](#cli) or by sweeping
multiple sizes through `kernellab compare`.

### Headline — M = N = K = 4096, fp32, 10 timed iterations after 3 warmup

All CUDA backend rows are verified against the fp64 `cpu_ref` oracle.
Nsight Compute hardware counters were unavailable on the measured host
because of `ERR_NVGPUCTRPERM`; register/shared-memory/spill data for the
custom CUDA kernels comes from `ptxas --verbose` logs.

| Backend       | Kernel (ms) | E2E (ms) | GFLOPs | % of cuBLAS | Verified |
| ------------- | ----------: | -------: | -----: | ----------: | :------: |
| `cuda_naive`  |       26.28 |    31.36 |  5231  |        11 % |    ✓     |
| `cuda_smem`   |       18.50 |    23.57 |  7428  |        16 % |    ✓     |
| `cuda_reg`    |        4.09 |     9.17 | 33628  |        72 % |    ✓     |
| `cuda_reg_v2` Layer 1 float4 |        3.81 |     8.89 | 36095  |        77 % |    ✓     |
| `cuda_reg_v2` Layer 2 double buffer |        3.81 |     9.90 | 36057  |        77 % |    ✓     |
| `cuda_reg_v2` Layer 3 cp.async |        3.96 |     9.52 | 34726  |        74 % |    ✓     |
| `cuda_reg_v2` Layer 4 warptiling |        3.86 |     9.04 | 35647  |        76 % |    ✓     |
| `cuda_reg_v2` autotuned BK16 warptile |        3.51 |     8.60 | 39108  |        83 % |    ✓     |
| `cublas`      |        2.94 |     8.01 | 46747  |       100 % |    ✓     |

`cuda_smem` is **42 % faster than `cuda_naive`** because it reuses each
global-memory load through shared memory. `cuda_reg` adds an 8×8
per-thread register tile, so each shared-memory operand feeds a small
outer product before leaving registers; at 4096³ it is **4.5× faster
than `cuda_smem`** and reaches **72 % of strict fp32 cuBLAS**.
`cuda_reg_v2` keeps the same fp32 math and adds vectorized tile
movement, occupancy-aware autotuning, BK=16, and a warp-tiled mapping;
the best measured configuration reaches **83 % of cuBLAS** at 4096³.

### Size sweep

`kernel_ms` is the CUDA-event-measured device time. `gflops = 2*M*N*K /
(kernel_ms * 1e6)`. `% cuBLAS` is purely descriptive — a higher
percentage at smaller sizes mostly means cuBLAS hasn't fully warmed up
its tile-selection heuristics.

| M=N=K | `cuda_naive` (GFLOPs / % cuBLAS) | `cuda_smem` (GFLOPs / % cuBLAS) | `cuda_reg` (GFLOPs / % cuBLAS) | `cuda_reg_v2` L1 float4 (GFLOPs / % cuBLAS) | `cuda_reg_v2` L2 double buffer (GFLOPs / % cuBLAS) | `cuda_reg_v2` L3 cp.async (GFLOPs / % cuBLAS) | `cuda_reg_v2` L4 warptiling (GFLOPs / % cuBLAS) | `cuda_reg_v2` autotuned (GFLOPs / % cuBLAS) | `cublas` (GFLOPs) |
| ----: | -------------------------------: | ------------------------------: | -----------------------------: | ------------------------------------------: | --------------------------------------------------: | ----------------------------------------------: | ------------------------------------------------: | --------------------------------------------: | ----------------: |
|  1024 |                  5383 / **15 %** |                 6241 / **17 %** |               11983 / **36 %** |                         12468 / **38 %** |                                   12451 / **37 %** |                               15796 / **47 %** |                                 16252 / **49 %** |                                 not measured |             33245 |
|  2048 |                  5537 / **11 %** |                 6945 / **14 %** |               32794 / **64 %** |                         34248 / **68 %** |                                   34979 / **69 %** |                               32828 / **65 %** |                                 34071 / **67 %** |                           37469 / **73 %** |             50986 |
|  4096 |                  5231 / **11 %** |                 7428 / **16 %** |               33451 / **71 %** |                         36095 / **77 %** |                                   36057 / **77 %** |                               34726 / **74 %** |                                 35647 / **76 %** |                           39108 / **83 %** |             47086 |

CPU baselines for context (1024³): `cpu_ref` (fp64 oracle, single-thread)
0.29 GFLOPs; `cpu_omp` (parallel fp32) ~3.9 GFLOPs. cuBLAS at 4096³ hits
~51 % of theoretical fp32 peak on this device.

### Measurement notes

This is the value of the v2 foundations:

1. **`kernel_ms` is kernel-only time**. Because `Backend::Prepare()` runs once and
   sinks `cudaMalloc` + H2D + cuBLAS handle creation into setup, every
   `kernel_ms` reported is just `cudaEvent`-bracketed kernel time — no
   amortized allocation contamination. cuBLAS at 4096³ shows
   `kernel_ms = 2.95` vs `e2e_ms = 8.03`; that 5 ms gap is the D2H copy
   and stream sync, which is a real production cost worth reporting
   separately rather than hiding inside "kernel time".
2. **Verification is K-aware**. `DefaultTolerance(problem)` scales by
   `sqrt(K)` so an fp32 GEMM with random inputs that diverges from the
   fp64 `cpu_ref` oracle by `~1e-4` at K=2048 still passes, while a
   real kernel bug ten times that magnitude still fails. The same code
   path catches both.
3. **`cuda_smem` was written, verified against the fp64 oracle at K up
   to 4096, and benchmarked across the size sweep without touching the
   framework.** That's the whole point of the session abstraction:
   new backends plug in, the rest is data flow.

## Build

```bash
./scripts/build_cpu_only.sh
```

Equivalent manual command:

```bash
cmake -S . -B build -DKERNELLAB_ENABLE_CUDA=OFF -DKERNELLAB_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Release build:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DKERNELLAB_ENABLE_CUDA=OFF \
  -DKERNELLAB_BUILD_TESTS=ON
cmake --build build-release
ctest --test-dir build-release --output-on-failure
```

CUDA-enabled build:

```bash
./scripts/build_cuda.sh
```

Override the CUDA install path via `CUDA_ROOT` when the toolkit isn't
at the script default. Equivalent manual command:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DKERNELLAB_ENABLE_CUDA=ON \
  -DKERNELLAB_BUILD_TESTS=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDA_ROOT
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

## CLI

```bash
./build/kernellab run --backend cpu_ref --m 256 --n 256 --k 256 --iterations 5 --warmup 1
./build/kernellab verify --backend cpu_omp --m 128 --n 128 --k 128
./build/kernellab compare --backends cpu_ref,cpu_omp,cublas --m 256 --n 256 --k 256 --iterations 5
./build/kernellab --list-backends
./build/kernellab --version
./build/kernellab --print-device-info
```

Optional exports:

```bash
./build/kernellab compare --backends cpu_ref,cpu_omp --m 64 --n 64 --k 64 --json-out results.json --csv-out results.csv
```

## Notes

- `cpu_ref` is the higher-precision correctness oracle used by `verify` and `compare`.
- `cpu_naive` is the scalar fp32 CPU baseline.
- `cpu_omp` is compiled and enabled only when OpenMP is found.
- `cuda_naive`, `cuda_smem`, and `cublas` are compiled only when CUDA is enabled and the toolkit is detected.
- `cublas` explicitly uses `CUBLAS_PEDANTIC_MATH` so the cuBLAS row is a strict fp32 SGEMM baseline rather than CUDA's default math mode.
- On CPU-only builds, CUDA backends remain visible to explicit requests but report themselves as unavailable.
- `verify` is intentionally a single-shot correctness command in v2 foundations; benchmarking-oriented iteration controls live on `run` and `compare`.
- `run` and `compare` default to `10` timed iterations and `2` warmups.
- Reports now distinguish `kernel_ms` from end-to-end `mean_ms`.

## LLM GEMM Atlas

`atlas/` extends kernellab from square GEMM kernels to transformer-shaped GEMMs
and end-to-end LLM inference profiling. It measures cuBLAS efficiency for
prefill and decode projection shapes, then maps those microbench results to
Nsight Systems kernel summaries from a real `Qwen/Qwen2.5-1.5B` inference run.

Start here:

```bash
python -m pip install -r atlas/requirements.txt

python atlas/bench_transformer_gemm.py \
  --hardware RTX6000Ada \
  --out results/transformer_gemm_atlas.csv

nsys profile \
  -o traces/trace_qwen \
  --capture-range=cudaProfilerApi \
  --cuda-memory-usage=true \
  python atlas/infer_with_profile.py \
    --model Qwen/Qwen2.5-1.5B \
    --max-new-tokens 128
```

See [LLM GEMM Atlas](docs/atlas.md) and [atlas/README.md](atlas/README.md) for
the full workflow. On RTX 6000 Ada, the committed fp16 run measured prefill
GEMMs at **84.3% peak** versus decode GEMMs at **0.9% peak** (**89.9x** gap).
Nsight Systems then showed the largest decode GEMV/GEMM kernels accounting for
about **70.2%** of GPU kernel time in a 128-token `Qwen/Qwen2.5-1.5B` run.

## Docs

- [LLM GEMM Atlas](docs/atlas.md)
- [Methodology](docs/methodology.md)
- [Adding a Backend](docs/adding-a-backend.md)
- [Register Tiling Design](docs/register_tiling_design.md)
- [Register Tiling Explained](docs/register_tiling_explained.md)
- [cuda_reg_v2 Design](docs/cuda_reg_v2_design.md)
- [cuda_reg_v2 Explained](docs/cuda_reg_v2_explained.md)
- [cuda_reg_v2 Autotune Design](docs/cuda_reg_autotune_design.md)
- [cuda_reg_v2 Autotune Explained](docs/cuda_reg_autotune_explained.md)

## Developer Guardrails

- `KERNELLAB_ENABLE_WARNINGS=ON` enables strict compiler warnings.
- `KERNELLAB_ENABLE_SANITIZERS=ON` enables ASan/UBSan for supported CPU-only debug-style builds.
- GitHub Actions currently covers CPU-only Debug/Release validation.
- A future self-hosted Linux `aarch64` CUDA 13 runner is the intended next CI extension.
