# cuda_reg_v2 autotune explained

This pass changes direction from "add another CUDA feature" to "protect occupancy and search the tile shape." The previous cp.async/warptiling path was correct but regressed because it pushed register use to about 130 registers/thread. On RTX 6000 Ada that left only one 256-thread block resident per SM for those kernels, so the scheduler did not have enough warps to hide latency.

## What changed

`cuda_reg_v2` is now a compile-time autotuned kernel. The source keeps the same Backend/Session structure and the same cudaEvent timing contract, but the kernel shape is controlled by macros:

```text
KLAB_REG_V2_BM
KLAB_REG_V2_BN
KLAB_REG_V2_BK
KLAB_REG_V2_TM
KLAB_REG_V2_TN
KLAB_REG_V2_WARPTILE
```

The default champion is:

```text
BM=128, BN=128, BK=16, TM=8, TN=8, threads=256, warptile=on
```

The implementation deliberately returns to the winning Layer 1 memory layout:

```text
As[bk][row]  // transposed A in shared memory
Bs[bk][col]  // row-major B in shared memory
```

A is still loaded as `float4` along K, then scattered into transposed shared memory. That scatter is worth keeping because compute reads each thread's A fragment as contiguous `float4` values. B is loaded and stored as `float4` along N.

The kernel uses:

```cpp
__launch_bounds__(threads, 2)
```

This tells ptxas to compile with at least two blocks/SM in mind. It is not magic: if the accumulator tile is too large, the compiler can still use many registers. But it prevents free register growth and makes the register budget visible in ptxas.

## Sweep results

All listed configs passed:

```text
verify 128x128x128
verify 130x129x17
verify 512x512x512
```

NCU performance counters are blocked on this machine by `ERR_NVGPUCTRPERM`, so occupancy below is theoretical from ptxas:

```text
active blocks/SM = min(65536 / (reg * threads), 102400 / smem, 1536 / threads)
occupancy = active_blocks * threads / 32 / 48
```

The broad sweep started with low-probability `64x64` configs; after those proved far below the current 77% baseline, the run focused on the design doc's high-value candidates: rectangular `8x4`, square `8x8`, `BK=16`, and `64/128` block shapes.

| Config | Threads | Reg/thread | Smem | Theoretical occupancy | 2048^3 GFLOPs / %cuBLAS | 4096^3 GFLOPs / %cuBLAS | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BM64 BN64 BK8 TM4 TN4 | 256 | 61 | 4224 | 66.7% | 26841 / 52.7% | 29248 / 62.0% | discarded |
| BM64 BN64 BK8 TM8 TN4 | 128 | 79 | 4224 | 50.0% | 28165 / 54.8% | 32760 / 69.5% | discarded |
| BM64 BN64 BK8 TM4 TN8 | 128 | 80 | 4224 | 50.0% | 25026 / 48.7% | 29085 / 61.8% | discarded |
| BM128 BN128 BK8 TM8 TN4 | 512 | 64 | 8320 | 66.7% | 35095 / 68.4% | 34684 / 74.2% | discarded |
| BM128 BN128 BK8 TM4 TN8 | 512 | 64 | 8320 | 66.7% | 24584 / 48.7% | 27095 / 57.8% | discarded |
| BM128 BN128 BK8 TM8 TN8 | 256 | 127 | 8320 | 33.3% | 34903 / 69.0% | 36088 / 77.1% | baseline-equivalent |
| BM128 BN128 BK16 TM8 TN4 | 512 | 64 | 16640 | 66.7% | 37771 / 74.5% | 37706 / 80.3% | finalist |
| BM128 BN128 BK16 TM4 TN8 | 512 | 64 | 16640 | 66.7% | 26901 / 52.2% | 29597 / 63.1% | discarded |
| BM128 BN128 BK16 TM8 TN8 | 256 | 127 | 16640 | 33.3% | 37008 / 73.1% | 38174 / 81.3% | finalist |
| BM128 BN64 BK16 TM8 TN4 | 256 | 79 | 12544 | 50.0% | 29170 / 56.8% | 35793 / 76.6% | discarded |
| BM128 BN64 BK16 TM8 TN8 | 128 | 127 | 12544 | 33.3% | 33480 / 65.8% | 36602 / 78.2% | discarded |
| BM64 BN128 BK16 TM8 TN4 | 256 | 79 | 12544 | 50.0% | 32880 / 63.9% | 36422 / 77.7% | discarded |
| BM64 BN128 BK16 TM8 TN8 | 128 | 127 | 12544 | 33.3% | 33912 / 66.3% | 36828 / 78.3% | discarded |
| BM128 BN128 BK16 TM8 TN8 + warptile | 256 | 127 | 16640 | 33.3% | 37469 / 73.5% | 39108 / 83.1% | champion |

## Why the champion wins

The initial hypothesis was "reduce TMxTN from 64 accumulators to 32 accumulators, lower registers, and win through occupancy." That helped register count: `8x4` uses 64 registers/thread and reaches a theoretical 66.7% active-warp occupancy. But it also loses too much per-thread ILP and arithmetic intensity. The `8x4` `BK=8` candidate only reached 74.2% cuBLAS at 4096^3.

The real win was `BK=16`. It doubles the K depth per shared-memory tile, so each CTA does more math per tile load and pays fewer synchronization/load phases across K. With `BM=BN=128` and `TM=TN=8`, it keeps the strong register-tile math from Layer 1 while improving K-tile reuse.

The controlled warptile then adds the missing piece. Unlike the previous Layer 4, this warptile does not switch A to row-major shared memory and does not introduce cp.async. It only remaps the existing 8x8 thread tiles so eight warps cover a `2x4` grid of `64x32` warp tiles:

```text
block tile: 128x128
warp grid : 2 x 4
warp tile : 64x32
lane tile : 8x8
```

Because the memory layout stays transposed-A/float4, ptxas remains at 127 registers/thread and 0 spill. The warp mapping improves locality/scheduling enough to move 4096^3 from 38.17 TFLOPs to 39.11 TFLOPs.

## Why this warptiling helped but the previous one regressed

Previous Layer 4:

- changed A shared memory to row-major so cp.async could copy it directly;
- increased compute-side A shared-memory instruction pressure;
- used about 130 registers/thread and roughly one resident block/SM;
- landed at 76.1% cuBLAS.

This autotuned warptile:

- keeps the Layer 1 transposed A layout;
- keeps `float4` shared-memory fragments;
- stays at 127 registers/thread with 0 spill;
- reaches 83.1% cuBLAS at 4096^3.

The lesson is that warptiling is not automatically good. It only helps when the tile mapping improves reuse or scheduling without breaking the register/occupancy budget.

## ptxas / NCU

Champion ptxas:

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 127 registers, 16640 bytes smem, 388 bytes cmem[0]
```

NCU attempt:

```text
ERR_NVGPUCTRPERM
No kernels were profiled.
```

Theoretical occupancy from ptxas:

```text
blocks_by_reg = floor(65536 / (127 * 256)) = 2
blocks_by_smem = floor(102400 / 16640) = 6
blocks_by_threads = floor(1536 / 256) = 6
active_blocks = 2
active_warps = 2 * 256 / 32 = 16
theoretical active-warp occupancy = 16 / 48 = 33.3%
```

This is below the design doc's ideal 40%+ target, but the 8x4 configs show that higher occupancy alone is not enough. The best point in this sweep is the balance of high ILP, `BK=16`, and controlled warptiling.

## Honest boundary

The target 85% was a stretch. The best verified result is:

```text
4096^3: 39107.6 GFLOPs, 83.06% cuBLAS
```

The remaining gap is plausible cuBLAS territory: SASS-level instruction scheduling, deeper per-shape autotuning, more careful smem bank/load scheduling, and possibly split-K or other shape-specific strategies. This version is a readable FP32 educational kernel; it does not use Tensor Cores and does not change the timing or GFLOPs formula.
