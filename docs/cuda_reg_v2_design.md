# Design Doc — `cuda_reg_v2`: Layered FP32 SGEMM Optimization

> Goal: extend `cuda_reg` with four isolated optimization layers: float4 movement, double buffering, cp.async, and warp tiling.
> The 90% cuBLAS target was an aspirational target for the layer sequence; README records only measured results.
> 环境:RTX 6000 Ada (sm_89), CUDA 12.1。

## 0. Development Rules

1. **分层增量,每层一个 commit、一行 benchmark、一段讲解、单独 verify。** 禁止一把梭重写。
2. **不删不改 `cuda_naive` / `cuda_smem` / `cuda_reg`。** `cuda_reg`(72%)是**永久保底**与阶梯一级。新代码进 `cuda_reg_v2`(或带层级开关的新后端)。
3. **任何一层做不对(oracle 不过)或做不出,就停在上一层。**
4. 全程 **FP32**,无 Tensor Core,口径不变(cudaEvent kernel time、`2MNK` 公式、fp64 oracle 校验)。

## 1. 起点与瓶颈(为什么是这四层)

`cuda_reg` 现状:72% cuBLAS、128 reg/thread、occupancy ~33%、0 spill。已知短板:
- A 的 global load 只"分段连续"(未向量化);
- load 与 compute 之间有 `__syncthreads` 硬等待,global-load 延迟没被掩盖;
- 没有 warp 级复用层。

四层正是逐个打这些点。

## 2. 分层实现(每层独立 verify + benchmark + 讲解)

### Layer 1 — float4 向量化 load(最低风险,预期 +5~8% → ~78-80%)
- global→smem、smem→register 改用 `float4`(128-bit)读写;要求 16B 对齐、tile 维度是 4 的倍数。
- 顺带修掉 A 的分段-coalescing 缺口(向量化提升内存吞吐)。
- 讲解点:128-bit load 省指令数 + 提带宽;对齐要求。

### Layer 2 — double buffering(smem 双缓冲,预期 +5~8% → ~83-86%)
- 开两套 smem(As0/As1, Bs0/Bs1):算 buffer i 的同时,把下一 K-tile 载入 buffer 1-i,**去掉 load↔compute 之间的硬同步**,用计算掩盖 global-load 延迟。
- smem 翻倍(~16KB),检查是否超 SM 上限、对 occupancy 的影响。
- 讲解点:为什么重叠 load(k+1)与 compute(k) 能藏延迟;smem/occupancy 代价。

### Layer 3 — cp.async(sm_80+,预期 +3~5% → ~86-89%)
- 把双缓冲里的 global→smem 拷贝换成 `cp.async`(异步拷贝,绕过寄存器),配 `commit_group` / `wait_group` 做真异步预取流水。
- Ada 是 sm_89,支持。注意 commit/wait 的位置与流水深度。
- 讲解点:cp.async 是什么(异步 global→smem、不过寄存器)、commit/wait group、比"普通 load 双缓冲"好在哪(省寄存器、真异步)、sm_80+ 要求。

### Layer 4 — warptiling(最难辩护,预期 +3~5% → ~目标 90%)
- 在 block tile 与 thread tile 之间加 **warp 级 tile**:让 8 个 warp 各算一块 sub-tile,提高寄存器复用与 ILP、减少 smem 流量。这是 CUTLASS 的结构。
- 讲解点:warp 级 tile 相比纯 thread tiling 为什么提升复用/ILP。

## 3. 每层硬交付

1. `cuda_reg_v2.cu`(或带层级宏的版本)——**重注释**,每层逻辑清晰可读。
2. CMake / 注册表 / CLI(`--backend cuda_reg_v2`)接好。
3. `docs/cuda_reg_v2_explained.md`——**每层一节**:做了什么、为什么、预期 vs 实测、辩护清单。
4. README results 表:**每个跑通的层加一行**(真实、FP32、verified)。
5. 每层 ptxas(reg/smem/spill);NCU 若权限允许补 occupancy/throughput,否则注明 `ERR_NVGPUCTRPERM` 并用 ptxas 推理论 occupancy。
6. 边界正确性(128³ / 非整除 / 512³)每层都过。

## 4. Design Checklist

- 每层"做了什么 + 为什么有用 + 实测提升多少"。
- double buffering 与 cp.async 的区别(后者真异步、省寄存器)。
- 为什么即使到 ~88% 仍不及 cuBLAS:autotuning across shapes、汇编级指令调度、可能的 split-K。
- occupancy 怎么随各层变(float4/双缓冲/warptiling 对寄存器与 smem 的影响)。

## 5. 风险 / 回退

- **保底**:`cuda_reg`(72%,已 verified)。
- `cuda_reg_v2` 是增量优化路径；如果某层正确性或性能退步，则保留上一层结果并记录原因。
- cp.async / 双缓冲极易写出"数值偶发错"的 bug → 每层必过 oracle,且在多尺寸上重复 verify。
