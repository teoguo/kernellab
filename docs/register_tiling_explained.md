# cuda_reg register tiling 逐步讲解

这份文档是给作者学习和面试答辩用的。`cuda_reg` 的目标不是“赢 cuBLAS”，而是在 `cuda_naive -> cuda_smem -> cuda_reg -> cublas` 这条优化阶梯里，把“为什么 register tiling 是下一步”讲清楚，并且让代码能逐行辩护。

## 1. 这个 kernel 在算什么

GEMM 是 `C[M,N] = A[M,K] * B[K,N]`。`cuda_naive` 是一个线程算一个 `C[row,col]`，每次 FMA 都从 global memory 读 A/B。`cuda_smem` 先把 A/B 的一小块搬进 shared memory，一个线程仍然只算一个输出。`cuda_reg` 再进一步：一个线程一次算 `8x8 = 64` 个输出，把 64 个 partial sum 放在寄存器里。

这样做的核心变化是：从 shared memory 读到的一个 A 值，不再只服务一个输出，而是服务同一行的 8 个输出；一个 B 值也服务同一列方向的 8 个输出。一次 smem 读取，换来更多 FMA。

## 2. 为什么是 BM=BN=128, BK=8, TM=TN=8

`BM=BN=128` 表示一个 block 负责 C 的 `128x128` 大 tile。`TM=TN=8` 表示一个线程负责自己的 `8x8` micro-tile。于是 block 内需要的线程数是：

```text
(BM * BN) / (TM * TN) = (128 * 128) / 64 = 256 threads
```

256 threads/block 是 CUDA 上很常见的规模：有足够多的 warp 并行工作，也不会让单个 block 太大。`BK=8` 表示每轮只沿 K 方向推进 8 个元素。shared memory 用量约为：

```text
As: BK * BM floats = 8 * 128 * 4 bytes
Bs: BK * BN floats = 8 * 128 * 4 bytes
total ~= 8 KB
```

代码里 `As` 多了 1 列 padding：`[BK][BM + 1]`。这是为了打破 `128` 正好是 32 个 smem bank 的整数倍时可能出现的 bank 对齐模式。用量仍然约 8 KB，不会成为 occupancy 的主要限制。

这个配置的权衡是：`8x8` micro-tile 给了足够高的寄存器复用，但每线程 64 个 accumulator 会带来明显 register pressure。真实 occupancy、registers/thread、spill 必须看 NCU，不能靠嘴估。

## 3. 为什么 As 要转置存 shared memory

A 在 global memory 里是 row-major 的 `[M][K]`。加载时，线程按原始 A tile 的 row-major 顺序读，这样每个 warp 会读短的连续 K 段。写入 shared memory 时，代码把它放成：

```text
原始 A tile: A[row][bk]
shared As:  As[bk][row]
```

compute 阶段每个线程固定一个 `bk`，然后取自己 8 行的 A fragment：`As[bk][row_base + 0..7]`。这让内层读变成沿着 shared-memory row 的连续访问。对 warp 来说，很多线程会读相同或相邻的 A 地址，硬件可以 broadcast/multicast，bank conflict 风险比直接用 `[BM][BK]` 布局低。

一句话答辩：As 转置不是为了改变数学，而是为了让“计算时的访问模式”更顺。GEMM 的热路径是 K 内层反复读 smem，应该优先照顾这条路径。

## 4. register tiling 为什么提高 arithmetic intensity

`cuda_smem` 里，一个线程每次从 smem 读一个 A 和一个 B，只更新一个 accumulator：

```text
1 A + 1 B -> 1 FMA
```

`cuda_reg` 里，一个线程每个 `bk` 读 8 个 A 和 8 个 B，然后做外积：

```text
8 A + 8 B -> 8 * 8 = 64 FMA
```

从“每个 smem 值能产生多少计算”看，A fragment 的每个值会被 8 个 B 列复用，B fragment 的每个值会被 8 个 A 行复用。smem traffic 没有按 64 倍增加，但 FMA 数量大幅增加，所以 arithmetic intensity 上升。瓶颈更可能从 shared-memory 带宽转向 FP32 compute 或 register/occupancy。

## 5. `__syncthreads()` 为什么放在两个位置

第一处 barrier 在 cooperative load 后面。原因很直接：一个 block 的 256 个线程一起把 As/Bs 搬进 smem，任何线程开始计算前，都必须等整个 K-slice 准备好。

第二处 barrier 在 compute 之后、下一轮 load 之前。因为下一轮会复用同一块 shared memory。如果某些 warp 先跑完并开始写下一轮 As/Bs，而其他 warp 还在读当前 As/Bs，就会读到被覆盖的数据。这个 barrier 是 correctness 必需，不是性能装饰。

## 6. 边界尺寸怎么保证正确

加载 A/B 时对 `global_row < M`、`global_col < N`、`global_k < K` 做 guard，越界位置写 0 到 shared memory。最后写 C 时也 guard `row < M && col < N`。所以即使 M/N/K 不是 128/8 的整数倍，数学上仍然正确。

正确性口径仍然是项目现有的 `cpu_ref` fp64 oracle + K-aware `DefaultTolerance`。`cuda_reg` 全程 float 输入、float 寄存器累加，不使用 Tensor Core，也不改变 GFLOPs 公式或 kernel timing 口径。

## 7. register pressure、occupancy 和 spill 风险

每个线程至少有：

- 64 个 `acc[8][8]` accumulator；
- 8 个 `a_frag`；
- 8 个 `b_frag`；
- 若干 index / pointer / loop register。

所以实际 registers/thread 可能在 80+，也可能因为编译器展开和地址计算更高。寄存器越多，每个 SM 能同时驻留的 warp 越少，achieved occupancy 可能下降。更严重的是 spill：如果寄存器不够，编译器会把本该在寄存器里的值放到 local memory，而 local memory 实际走 global/L2 路径，性能会掉。

判断方式不要猜，跑 NCU：

- `registers/thread` 看 LaunchStats；
- `achieved occupancy` 看 Occupancy；
- `local memory load/store` 大于 0 就要警惕 spill；
- 同时看 DRAM/L2 throughput 和 shared-memory bank conflict。

如果 NCU 显示明显 spill，优先方案是把 `TM/TN` 从 `8x8` 调成 `4x8` 或 `4x4`，降低 accumulator 数量；另一个方案是尝试 `__launch_bounds__` 或 `--maxrregcount`，但这可能强行降寄存器并制造更多 spill，需要实测比较，不能只看 occupancy 变高。

## 8. 为什么仍然不及 cuBLAS

`cuda_reg` 是教学/简历项目里的清晰优化台阶，不是工业 SGEMM。它仍然没有：

- double buffering：不能把下一块 global->smem 的加载和当前块计算重叠；
- vectorized load/store：没有显式 `float4` 或更强的内存搬运优化；
- warp-level / instruction-level scheduling：没有手写 SASS 或复杂 pipeline；
- autotuning：没有按不同 shape、不同 GPU 自动搜索 BM/BN/BK/TM/TN；
- Tensor Core：本项目硬约束是 FP32 CUDA core 口径。

所以预期是 4096³ 下达到 cuBLAS 的 40-70% 是合理目标，但最终必须填真实测量。如果低于预期，也要用 NCU 数据解释是 register pressure、spill、memory throughput、bank conflict，还是 instruction scheduling 问题。

## 9. 当前实测状态

远程实测环境：`c2smarter-MS-7D69`，CUDA 12.1 build，NVIDIA driver 580.126.09，RTX 6000 Ada。由于系统 CMake 是 3.22.1，而项目要求 3.28+，实测时在 NAS user base 安装了 `cmake 3.31.10`，没有使用 sudo。

正确性：

```text
cuda_reg 128x128x128: verification_passed=true
cuda_reg 130x129x17:  verification_passed=true
cuda_reg 512x512x512: verification_passed=true
```

benchmark 结果，均由 `kernellab compare --backends cuda_naive,cuda_smem,cuda_reg,cublas` 跑出，且每行 verification passed：

| Size | cuda_reg GFLOPs | % cuBLAS | Verification | Notes |
| ---: | --------------: | -------: | :----------: | ----- |
| 1024³ | 12468.9 | 34.3% | passed | 小尺寸 cuBLAS kernel time 只有 0.059 ms，固定开销影响更明显 |
| 2048³ | 32884.0 | 64.8% | passed | 进入设计目标 40-70% 区间 |
| 4096³ | 33628.2 | 71.9% | passed | 略高于预期区间上沿，约 4.5x `cuda_smem` |

NCU 状态：普通用户运行 Nsight Compute 时遇到 `ERR_NVGPUCTRPERM`，系统不允许访问 NVIDIA performance counters。按照 workstation access guide，没有使用 sudo、没有改 driver/perf-counter 设置。因此 DRAM/L2 throughput、achieved occupancy、smem bank conflict 不能假装有数。

可替代的静态编译证据来自 `--ptxas-options=-v`：

```text
ptxas info : Function properties for ...RegTiledKernel...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 128 registers, 8224 bytes smem, 388 bytes cmem[0]
```

这说明当前编译版本没有 ptxas-level spill。128 registers/thread 和 256 threads/block 意味着每 block 约用 32768 registers；在 sm_89 常见 65536 registers/SM 的预算下，registers 会把理论驻留限制到约 2 blocks/SM，也就是 512 threads/SM。真实 achieved occupancy 仍需 NCU 权限才能确认。

| Metric | cuda_reg @4096³ |
| ------ | --------------- |
| achieved occupancy | NCU blocked by `ERR_NVGPUCTRPERM`; theoretical register-limited occupancy about 512 threads/SM |
| registers/thread | 128 registers/thread from ptxas |
| local memory load/store | NCU blocked; ptxas reports 0 spill stores and 0 spill loads |
| DRAM throughput | NCU blocked by `ERR_NVGPUCTRPERM` |
| L2 throughput | NCU blocked by `ERR_NVGPUCTRPERM` |
| smem bank conflict | NCU blocked by `ERR_NVGPUCTRPERM` |

如果之后管理员临时开放 performance counters，应按 `RUN_ON_GPU.md` 的 NCU 命令补跑，并把 achieved occupancy、DRAM/L2 throughput、shared-memory bank conflict 写回这张表。
