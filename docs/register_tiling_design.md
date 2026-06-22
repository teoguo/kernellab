# Design Doc — `cuda_reg`: Register-Tiled FP32 GEMM

> Purpose: add a `naive → smem → reg → cublas` optimization step to kernellab.
> This backend focuses on a readable register-tiling implementation that improves arithmetic intensity without changing the benchmark methodology.
> 约束环境:RTX 6000 Ada (sm_89), CUDA 12.1, 现有 Backend/Session 框架。

## 1. Goal / Non-goals

**Goal**
- 新增后端 `cuda_reg`:2D block tiling + 每线程 register tiling(thread tile)。
- 现实目标:4096³ FP32 下达到 **cuBLAS 的 40–70%**(对比当前 `cuda_smem` 的 ~16%)。README 只记录真实测得值。

**Non-goals(硬约束)**
- **不删、不改** `cuda_naive` / `cuda_smem`(保留可比较的优化阶梯)。
- **不上 Tensor Core**,全程 **FP32**(float 输入 + float 寄存器累加),与现有 benchmark 同口径。
- 不引入新依赖;不改 GFLOPs 公式与计时方式。

## 2. 为什么 register tiling 是正确的下一级

- `cuda_smem` 卡在 **shared memory 带宽**:一个线程只算 1 个输出 → 每从 smem 读一个字节只做很少 FLOP(arithmetic intensity 低)。
- **register tiling**:一个线程算一个 **TM×TN 的 micro-tile**(如 8×8=64 个输出),64 个累加器**常驻寄存器**。从 smem 取一个值进寄存器后,在 micro-tile 里被**复用 TM 或 TN 次**(外积)→ arithmetic intensity 上去 → 瓶颈从 smem 带宽转到计算。这就是 ~8% 峰值 → ~40%+ 的那一跳。

## 3. 算法(经典 2D blocktiling)

- **Block tile**:每个 block 算 `BM×BN` 输出,K 方向每次推进 `BK`。建议 `BM=BN=128, BK=8`。
- **Thread tile**:每个线程算 `TM×TN`。建议 `TM=TN=8`。
- **threads/block** = (BM·BN)/(TM·TN) = 128·128/64 = **256**。
- **Shared mem**:`As`(存成转置 `[BK][BM]` → 内层按行读、coalesced 且无 bank conflict)、`Bs[BK][BN]`。用量 = (128·8 + 8·128)·4 = **8 KB**,够。
- **每个 K-step**:
  1. 256 个线程**协作加载** As(转置写入)和 Bs 到 smem(coalesced,每线程按 stride 搬多个元素);`__syncthreads()`;
  2. 内层 `for bk in 0..BK`:每线程从 As 取 TM 个、从 Bs 取 TN 个进寄存器,做 **TM×TN 外积**累加到寄存器累加器;
  3. `__syncthreads()` 进下一个 K-step。
- 最后把 TM×TN 个累加器写回 C。

## 4. 正确性与测量口径

- **FP32 全程**;用现有 `cpu_ref`(fp64 oracle)+ K-aware `DefaultTolerance` 校验。
- 复用现有 Backend/Session:`kernel_ms` 只用 cudaEvent 围 kernel,E2E 单列,alloc/H2D 进 `Prepare()`。**口径与 smem/cublas 完全一致**。
- 边界:对 M/N/K 非 tile 整数倍的情况做 guard,或明确文档化"已测尺寸"。

## 5. Deliverables

1. `src/backends/cuda/cuda_reg.cu` —— **重注释**(每个代码块写清"做什么 + 为什么")。
2. 在后端注册表 + CLI(`--backend cuda_reg`)+ CMake 里接好。
3. `docs/register_tiling_explained.md` —— 逐步讲解 tile size 取值理由、arithmetic intensity 论证、occupancy 影响、bank conflict 分析、预期 vs 实测 % cuBLAS。
4. 在 RTX 6000 Ada 上重跑 size sweep(1024/2048/4096),给 README results 加一行 `cuda_reg`(真实、FP32、已校验)。
5. 对 `cuda_reg` @4096³ 跑 **NCU**:achieved occupancy、registers/thread、local mem load/store(**警惕 spill**:64 累加器 + 索引可能把寄存器顶高 → 若 spill,如实记录或调小 TM/TN)、DRAM/L2 throughput、smem bank conflict。结果写进 explained.md。

## 6. Design Checklist

- 为什么 `TM=TN=8 / BM=BN=128 / BK=8`?(复用 vs occupancy 权衡;smem 预算)
- 为什么 As 在 smem 里存转置?(加载 coalesced + 内层读无 bank conflict)
- register tiling 为什么提高 arithmetic intensity?(每个 smem load 在寄存器里复用 TM/TN 次)
- 寄存器压力:64 累加器 + 操作数 → 约多少寄存器/线程 → occupancy?有没有 spill?(看 NCU)
- 为什么仍不及 cuBLAS?(无 double buffering / 无汇编级调度 / 无 autotuning;可能未做 float4 向量化)
- 预期数:~40–70% cuBLAS;**报实测**。

## 7. 风险 / 回退

- 若 NCU 显示 spill(local mem load/store > 0):调小 TM/TN 到 4,或设 `__launch_bounds__`/maxrregcount,并记录权衡。
- **可选增量(有时间再做)**:float4 向量化 load、double buffering(预取下一 tile 与计算重叠)。各自单列、单测,别一次堆。
