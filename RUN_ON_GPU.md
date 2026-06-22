# RUN_ON_GPU.md

在远程 Linux + RTX 6000 Ada 机器上执行以下命令。假设你已经在 `kernellab-atlas-work` repo 根目录，并且分支是 `feat/cuda-reg-tiling`。

## 0. 确认环境

```bash
git status --short --branch
nvidia-smi
/usr/local/cuda-12.1/bin/nvcc --version
```

如果 CUDA 12.1 不在 `/usr/local/cuda-12.1`，把下面命令里的 `CUDA_ROOT` 改成实际路径。

## 1. Build: CUDA 12.1

基于现有 `scripts/build_cuda.sh`：

```bash
CUDA_ROOT=/usr/local/cuda-12.1 ./scripts/build_cuda.sh build-cuda
```

如果 workstation 上系统 CMake 低于项目要求的 3.28，不要用 sudo。可以把新版 CMake 装到 NAS user base：

```bash
mkdir -p /home/mnt/nas/c2smarter/python-userbase
PYTHONUSERBASE=/home/mnt/nas/c2smarter/python-userbase \
  python3 -m pip install --user "cmake>=3.28,<4"

export PYTHONUSERBASE=/home/mnt/nas/c2smarter/python-userbase
export PATH=/home/mnt/nas/c2smarter/python-userbase/bin:$PATH
CUDA_ROOT=/usr/local/cuda-12.1 ./scripts/build_cuda.sh build-cuda
```

确认 backend 注册：

```bash
./build-cuda/kernellab --list-backends
./build-cuda/kernellab --print-device-info
```

`--list-backends` 里应该能看到 `cuda_reg`，且在 CUDA 机器上 available 为 `true`。

## 2. 正确性: 先小尺寸过 oracle

先跑 tile 整除尺寸：

```bash
./build-cuda/kernellab verify --backend cuda_reg --m 128 --n 128 --k 128
```

再跑非整除边界尺寸，确认 guard 没问题：

```bash
./build-cuda/kernellab verify --backend cuda_reg --m 130 --n 129 --k 17
```

最后跑一个中等尺寸：

```bash
./build-cuda/kernellab verify --backend cuda_reg --m 512 --n 512 --k 512
```

如果任何一步失败，把完整 stdout/stderr 贴回给我，不要继续填 README 数字。

## 3. Benchmark: 1024 / 2048 / 4096 sweep

逐个尺寸跑 compare，拿 `cuda_reg` 的 GFLOPs 和 `% cuBLAS`：

```bash
./build-cuda/kernellab compare --backends cuda_naive,cuda_smem,cuda_reg,cublas --m 1024 --n 1024 --k 1024 --warmup 3 --iterations 10
./build-cuda/kernellab compare --backends cuda_naive,cuda_smem,cuda_reg,cublas --m 2048 --n 2048 --k 2048 --warmup 3 --iterations 10
./build-cuda/kernellab compare --backends cuda_naive,cuda_smem,cuda_reg,cublas --m 4096 --n 4096 --k 4096 --warmup 3 --iterations 10
```

可选：同时导出 CSV，方便回填 README：

```bash
mkdir -p results
./build-cuda/kernellab compare --backends cuda_naive,cuda_smem,cuda_reg,cublas --m 1024 --n 1024 --k 1024 --warmup 3 --iterations 10 --csv-out results/cuda_reg_1024.csv
./build-cuda/kernellab compare --backends cuda_naive,cuda_smem,cuda_reg,cublas --m 2048 --n 2048 --k 2048 --warmup 3 --iterations 10 --csv-out results/cuda_reg_2048.csv
./build-cuda/kernellab compare --backends cuda_naive,cuda_smem,cuda_reg,cublas --m 4096 --n 4096 --k 4096 --warmup 3 --iterations 10 --csv-out results/cuda_reg_4096.csv
```

## 4. NCU: cuda_reg @4096^3

先跑 section-based profile，通常最稳：

```bash
mkdir -p results
/usr/local/cuda-12.1/bin/ncu --set full \
  --target-processes all \
  --kernel-name regex:RegTiledKernel \
  --launch-skip 3 \
  --launch-count 1 \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SourceCounters \
  --export results/ncu_cuda_reg_4096 \
  ./build-cuda/kernellab run --backend cuda_reg --m 4096 --n 4096 --k 4096 --warmup 3 --iterations 1
```

再跑 metric-focused profile，直接抓答辩需要的指标：

```bash
/usr/local/cuda-12.1/bin/ncu \
  --target-processes all \
  --kernel-name regex:RegTiledKernel \
  --launch-skip 3 \
  --launch-count 1 \
  --metrics launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  ./build-cuda/kernellab run --backend cuda_reg --m 4096 --n 4096 --k 4096 --warmup 3 --iterations 1
```

重点看：

- `launch__registers_per_thread`: registers/thread；
- `sm__warps_active...`: achieved occupancy；
- `l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st`: local memory load/store，若明显大于 0，说明可能 spill；
- `dram__throughput...` 和 `lts__throughput...`: DRAM/L2 throughput；
- `l1tex__data_bank_conflicts...shared...`: shared-memory bank conflict。

如果 metric 名称因 NCU 版本不兼容，先查可用 metric：

```bash
/usr/local/cuda-12.1/bin/ncu --query-metrics | grep -E "registers_per_thread|warps_active|mem_local|throughput|bank_conflicts"
```

如果 NCU 返回 `ERR_NVGPUCTRPERM`，说明普通用户没有 performance counter 权限。不要用 sudo 或改 driver 设置；记录这个限制，并用 `--ptxas-options=-v` 补充 registers/spill 的编译证据：

```bash
cmake -S . -B build-cuda-ptxas \
  -DCMAKE_BUILD_TYPE=Release \
  -DKERNELLAB_ENABLE_CUDA=ON \
  -DKERNELLAB_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.1 \
  -DCMAKE_CUDA_FLAGS="--ptxas-options=-v"
cmake --build build-cuda-ptxas --target kernellab_core --clean-first 2>&1 | tee results/ptxas_cuda_reg_build.log
grep -A4 -B2 RegTiledKernel results/ptxas_cuda_reg_build.log
```

## 5. 如果看到 spill

如果 local memory load/store 明显大于 0，把 NCU 输出贴回给我。下一步不要先改 README 数字；优先比较两个回退方案：

1. 把 `TM/TN` 从 `8x8` 调小到 `4x8` 或 `4x4`，降低 accumulator 数量；
2. 尝试 `__launch_bounds__` 或 compiler register cap，但必须同时看 spill 和 GFLOPs，不能只看 occupancy。
