# Design Doc — Occupancy-First Autotuning for `cuda_reg_v2`

> Goal: improve over the 77% cuBLAS Layer 1 float4 result by tuning tile shape, register pressure, and occupancy.
> Core observation: the cp.async/warptiling layers regressed because register use reached about 130 registers/thread, limiting residency. The next lever is to recover occupancy while preserving arithmetic intensity.
> Reference point: hand-written FP32 SGEMM kernels can approach cuBLAS when tile shape, vectorized movement, warp mapping, and register budgeting are all tuned together. Results here are reported as measured.
> 环境:RTX 6000 Ada (sm_89), CUDA 12.1。

## 0. 纪律(不变)
1. 新分支 `feat/cuda-reg-autotune`(从 `feat/cuda-reg-v2` 切)。
2. 不删 naive/smem/reg;**L1 float4(77%)是保底**。每个配置/变体单独 verify(128³ / 130×129×17 / 512³)+ benchmark + ptxas,留最优、弃坏的。
3. 全程 FP32、不改口径、保留 float4。
4. 报完整 sweep 表;如果没有达到 85%,记录最优配置、occupancy 数据和剩余差距。

## 1. Step 1 — 参数化 kernel
- 把 `BM, BN, BK, TM, TN` 改成编译期模板参数(C++ template 或宏),保留 float4 向量化 load(已验证的真增益)。
- 用 `__launch_bounds__(threads, 2)` 给编译器明确的 residency 目标,把寄存器预算控制在能容纳 ≥2 block/SM 的范围内。

## 2. Step 2 — 扫一组聚焦配置(非穷举)
对每个配置:ptxas 看 reg/thread → 校验 occupancy(要求 `reg×threads×2 ≤ 65536`,即 ≥2 block/SM,目标 ~50% occupancy)→ oracle verify → 在 2048³ 和 4096³ benchmark。取**实测 GFLOPs 最高**者。

候选网格(全 FP32 + float4):
- `BM,BN ∈ {64,128}`,`BK ∈ {8,16}`
- `TM×TN ∈ {4×4, 8×4, 4×8, 8×8}`(**重点试 4×8 / 8×4 矩形 tile:累加器从 64 降到 32,寄存器大降,occupancy 升**)
- `threads = BM·BN/(TM·TN)`,保持 ∈ [128,512]
- smem =(BM·BK + BK·BN)·4·(双缓冲则 ×2)≤ ~100KB

经验上可能胜出的:`BM=BN=128, BK=8/16, TM=8,TN=4`(累加器减半→寄存器降→2-3 block/SM);或 `BM=128,BN=64`。关键是避免高 arithmetic intensity 被过高寄存器压力抵消。

## 3. Step 3 — 寄存器受控的 warptiling(基础 autotune 触顶后再上)
- 重做 warptiling:在 block tile 与 thread tile 间加 warp tile `WM×WN`,但**选 WM/WN/TM/TN 时保证每线程寄存器 ≤ ~128、仍 ≥2 block/SM**。
- 之前 L4 退步正因为寄存器到 130 → 1 block/SM;这次的硬约束就是"warptiling 不许把 occupancy 压回 17%"。

## 4. Step 4 — double buffering / cp.async(仅当 occupancy 已健康)
- occupancy 拉回 ~40%+ 后,double buffering 才有足够 warp 去 overlap → 这时再测,实测有提升才留。

## 5. Measurement Boundary / 回退
- 80–85% 合理可期;85–90% 是 stretch,要各项对齐。
- 到不了 85%:报最优配置 + occupancy/寄存器数据 + "剩余差距 = cuBLAS 的 SASS 级调度 + per-shape autotuning"。
- 保底永远是 L1 float4(77%)。

## 6. 交付 + 辩护清单
- sweep 表:config → reg/thread → 理论 occupancy → GFLOPs → % cuBLAS;标出冠军配置。
- `docs/cuda_reg_autotune_explained.md`:为什么 autotuning(性能是 register×occupancy×intensity 的非显式函数,经验搜索)、冠军配置为何赢、warptiling 这次为何生效(寄存器受控)vs 之前为何退步(寄存器膨胀→17% occupancy)。
- README 加冠军配置一行。
