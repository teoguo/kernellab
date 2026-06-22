# cuda_reg_v2 explained

`cuda_reg_v2` is the experimental path above the verified `cuda_reg` baseline. The baseline stays untouched so it remains the honest fallback. Each layer below must pass the fp64 `cpu_ref` oracle on `128^3`, `130x129x17`, and `512^3` before the next layer starts.

## Layer 1: float4 vectorized load path

### What changed

Layer 1 keeps the exact `cuda_reg` math shape: `BM=BN=128`, `BK=8`, `TM=TN=8`, and 256 threads/block. One thread still owns one `8x8` register tile.

Only the load path changes:

- A global memory loads are grouped as `float4` across the K dimension when the row pointer is 16-byte aligned and the full 4-float vector is inside bounds.
- B global memory loads are grouped as `float4` across the N dimension and written to `Bs` as `float4`.
- `As` is still stored transposed, but its stride changes from `BM+1=129` to `BM+4=132`. The `+4` padding keeps each `As[bk]` row 16-byte aligned for shared-memory `float4` reads while still avoiding a plain 128-float bank-aligned stride.
- The compute loop loads each thread's 8 A values and 8 B values from shared memory as two `float4` reads each, then unpacks them into the readable scalar `a_frag[]` and `b_frag[]` arrays.
- Edge tiles and non-16-byte-aligned rows fall back to scalar-safe behavior by zero-padding out-of-range vectors. This is why the non-divisible `130x129x17` oracle test remains mandatory.

### Why it helps

The original `cuda_reg` already has high arithmetic intensity because each smem value feeds an `8x8` register outer product. Layer 1 does not change that algorithm. It tries to reduce instruction overhead and improve memory transaction shape:

- A `float4` global load asks the compiler/hardware for one 128-bit load instead of four 32-bit scalar loads.
- A `float4` shared-memory fragment load cuts the smem instruction count for each thread fragment.
- Keeping `As` transposed preserves the compute-friendly access pattern from `cuda_reg`.

### Expected vs measured

Expected from the design doc: roughly +5-8% over `cuda_reg` if load instruction count is a meaningful bottleneck.

Measured on `c2smarter-MS-7D69` / RTX 6000 Ada / CUDA 12.1:

| Size | cuda_reg_v2 Layer 1 GFLOPs | % cuBLAS | Verification |
| ---: | -------------------------: | -------: | :----------: |
| 1024^3 | 12468.2 | 37.5% | passed |
| 2048^3 | 34248.2 | 67.8% | passed |
| 4096^3 | 36094.7 | 77.3% | passed |

Same-run uplift over `cuda_reg`:

| Size | cuda_reg GFLOPs | cuda_reg_v2 L1 GFLOPs | Uplift |
| ---: | --------------: | --------------------: | -----: |
| 1024^3 | 12061.0 | 12468.2 | +3.4% |
| 2048^3 | 32812.5 | 34248.2 | +4.4% |
| 4096^3 | 33244.7 | 36094.7 | +8.6% |

### ptxas / NCU

Nsight Compute still reports `ERR_NVGPUCTRPERM`, so achieved occupancy, DRAM/L2 throughput, and bank-conflict counters are unavailable without admin-side performance counter access.

`ptxas --verbose` output:

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 113 registers, 8320 bytes smem, 388 bytes cmem[0]
```

Compared with `cuda_reg` baseline ptxas (`128 registers`, `8224 bytes smem`, `0 spill`), Layer 1 lowers register allocation while adding a tiny amount of smem padding. A rough register-limited occupancy estimate on an SM with 65536 registers is still about two 256-thread blocks per SM: `2 * 256 * 113 = 57856` registers, while three blocks would exceed the register file.

| Metric | Layer 1 |
| ------ | ------- |
| registers/thread | 113 |
| smem | 8320 bytes |
| spill stores/loads | 0 / 0 |
| NCU counters | blocked by `ERR_NVGPUCTRPERM` |

### Defense checklist

- Why `float4`? It reduces load instruction count and expresses 16-byte aligned accesses on hot global/smem load paths.
- Why keep `As` transposed? Compute still reads fixed-`bk` rows from shared memory; transposed `As[bk][row]` keeps the inner-loop fragment contiguous.
- Why change padding to `+4`? `+1` breaks bank stride but misaligns later rows for `float4`; `+4` keeps 16-byte row alignment and still avoids a 128-float stride.
- Why scalar fallback? Real verification includes odd sizes; vectorization must never be allowed to read invalid memory or change the fp32 math.

## Layer 2: ordinary-load double buffering

### What changed

Layer 2 keeps Layer 1's float4 load path and opens two shared-memory buffers:

```text
As[2][BK][BM + 4]
Bs[2][BK][BN]
```

The kernel first loads K tile 0 into buffer 0. For each loop iteration:

1. choose `current_buffer` for compute and `next_buffer = 1 - current_buffer`;
2. if there is a next K tile, cooperatively load it into `next_buffer`;
3. compute the 8x8 register tile from `current_buffer`;
4. `__syncthreads()` so every warp sees a fully loaded next buffer before the next iteration;
5. flip `current_buffer`.

The important correctness point is that no warp overwrites the buffer being consumed. The previous iteration's barrier proves the next buffer is ready and the old current buffer is no longer being read.

### Why it helps

The intended win is to reduce the hard "load tile, barrier, compute tile, barrier" rhythm. Even though ordinary global loads are still synchronous within each thread, putting next-tile loads before current-tile compute can let the warp scheduler hide some load stalls behind arithmetic from other warps. The price is doubled shared memory.

This is not the same as `cp.async`. Layer 2 still routes global loads through registers and cannot create the explicit async copy pipeline that Layer 3 will try.

### Expected vs measured

Expected from the design doc: +5-8% over Layer 1 if global-load latency is exposed.

Measured results:

| Size | cuda_reg_v2 Layer 2 GFLOPs | % cuBLAS | Verification |
| ---: | -------------------------: | -------: | :----------: |
| 1024^3 | 12451.3 | 37.3% | passed |
| 2048^3 | 34979.4 | 68.6% | passed |
| 4096^3 | 36056.5 | 77.0% | passed |

Same-run uplift over `cuda_reg`:

| Size | cuda_reg GFLOPs | cuda_reg_v2 L2 GFLOPs | Uplift |
| ---: | --------------: | --------------------: | -----: |
| 1024^3 | 12135.4 | 12451.3 | +2.6% |
| 2048^3 | 32996.6 | 34979.4 | +6.0% |
| 4096^3 | 33187.9 | 36056.5 | +8.6% |

Layer 2 is essentially flat versus Layer 1 at 4096^3: 36056.5 GFLOPs vs Layer 1's 36094.7 GFLOPs. The honest interpretation is that ordinary-load double buffering did not add meaningful overlap for the largest square GEMM. It did not break correctness, and it keeps the door open for Layer 3 `cp.async`, where the copy can actually be asynchronous and can avoid staging through registers.

### ptxas / NCU

NCU remains blocked by `ERR_NVGPUCTRPERM`.

`ptxas --verbose` output:

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 109 registers, 16640 bytes smem, 388 bytes cmem[0]
```

Shared memory doubled from 8320 bytes to 16640 bytes, exactly as expected. Registers/thread dropped from 113 to 109, and there is still no spill. With 109 registers/thread and 256 threads/block, a 65536-register SM can still fit at most two such blocks by registers: `2 * 256 * 109 = 55808`, while three blocks would exceed the register file.

### Defense checklist

- What is double buffering? Two shared-memory buffers let one K tile be prepared while the other is consumed.
- Why is there a barrier after compute? It proves all next-buffer loads are visible before any warp reads that buffer next iteration.
- Why did it not improve much at 4096^3? Ordinary loads are still synchronous in each issuing warp, so this layer only gives scheduler-level latency hiding, not a true async copy pipeline.
- Why continue to cp.async? `cp.async` is the mechanism that can make global-to-smem copy genuinely asynchronous and avoid extra register staging.

## Layer 3: cp.async with row-major A staging

### What changed

Layer 3 replaces the ordinary global-to-shared loads with `cp.async` for the aligned hot path. The kernel still uses `BM=BN=128`, `BK=8`, `TM=TN=8`, 256 threads/block, and the same cudaEvent timing contract.

The hard layout change is A:

```text
Layer 2 As: As[bk][row]        // compute-friendly transpose
Layer 3 As: As[row][bk + pad]  // cp.async-friendly row-major staging
```

That change is required because `cp.async` can copy contiguous global bytes into contiguous shared-memory bytes, but it cannot transpose or scatter values while copying. A is row-major in global memory, so one row's K segment is contiguous. Each thread copies one 16-byte half-row of A and one 16-byte segment of B.

The A row stride is `BK + 4 = 12` floats, plus a 4-float skew for alternating 8-row groups. The padding keeps each A row's shared-memory destination 16-byte aligned for `cp.async`; the skew avoids making the two row groups inside the current warp mapping repeatedly hit the same shared-memory banks.

Boundary correctness needed one extra rule: only fully valid and 16-byte-aligned chunks use `cp.async`. Odd shapes such as `130x129x17` make many row starts unaligned, so those chunks fall back to scalar zero-fill stores. This keeps the fp64 oracle test correct without slowing the large aligned benchmark shapes.

The copy pipeline is:

1. issue async copies for tile 0 into buffer 0;
2. `cp.async.commit_group`;
3. `cp.async.wait_group 0`;
4. `__syncthreads()` because wait is per-thread but the tile is cooperative;
5. for each K tile, issue async copies for the next tile into the other buffer;
6. compute the current tile while the next copy group is in flight;
7. wait and synchronize before reading the next buffer.

### Why it helps, and why it hurt here

The intended win is real: `cp.async` moves global data directly into shared memory, avoiding register staging and enabling a true async copy group. That is better machinery than Layer 2's ordinary-load double buffering.

The problem is the A layout tradeoff. Layer 2's transposed `As[bk][row]` let each thread load its 8 A values with two contiguous `float4` shared-memory reads. Layer 3's row-major `As[row][bk]` makes the compute loop read one fixed `bk` across 8 rows, so each thread performs 8 scalar A shared-memory reads per `bk`. The copy path improved, but the compute-side shared-memory instruction count and register pressure increased.

### Expected vs measured

Expected from the design doc: +3-5% over Layer 2 if async copy removes exposed load latency without damaging the compute loop.

Measured results:

| Size | cuda_reg_v2 Layer 3 GFLOPs | % cuBLAS | Verification |
| ---: | -------------------------: | -------: | :----------: |
| 1024^3 | 15795.9 | 47.1% | passed |
| 2048^3 | 32827.9 | 64.5% | passed |
| 4096^3 | 34726.2 | 74.2% | passed |

Same-run comparison against `cuda_reg`:

| Size | cuda_reg GFLOPs | cuda_reg_v2 L3 GFLOPs | Uplift |
| ---: | --------------: | --------------------: | -----: |
| 1024^3 | 12037.7 | 15795.9 | +31.2% |
| 2048^3 | 32799.2 | 32827.9 | +0.1% |
| 4096^3 | 33264.1 | 34726.2 | +4.4% |

Against Layer 2, Layer 3 is a regression at the target 4096^3 size: 34726.2 GFLOPs vs Layer 2's 36056.5 GFLOPs. This layer is correct and useful as a learning checkpoint, but it is not the best-performing v2 checkpoint so far.

### ptxas / NCU

NCU is still blocked by `ERR_NVGPUCTRPERM`, so hardware counters such as achieved occupancy, local memory transactions, DRAM/L2 throughput, and shared bank conflicts are unavailable from this account.

`ptxas --verbose` output:

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 129 registers, 20512 bytes smem, 388 bytes cmem[0]
```

There is no spill, but register pressure jumped from Layer 2's 109 registers/thread to 129 registers/thread. With 256 threads/block and a 65536-register SM, two resident blocks would need `2 * 256 * 129 = 66048` registers, slightly above the register file. That implies a one-block-per-SM register limit before any other scheduling limits are considered, which helps explain the 4096^3 slowdown.

### Defense checklist

- What is `cp.async`? It is an sm_80+ instruction that copies global memory to shared memory asynchronously, without first placing the loaded values in normal registers.
- Why change `As` to row-major? `cp.async` can only copy contiguous bytes; it cannot perform the old A transpose scatter.
- Why keep scalar fallback? Boundary shapes can make source pointers unaligned or partially valid. Issuing 16-byte `cp.async` there would be undefined or would read outside the matrix.
- Why `commit_group` and `wait_group 0`? `commit_group` publishes the issued async copies as a group; `wait_group 0` waits until no committed group is pending before the tile is consumed.
- Why still use `__syncthreads()` after wait? `wait_group` is per-thread. The tile is cooperatively loaded, so the block needs a barrier before every thread can safely read every other thread's copied data.
- Why did 4096^3 regress? The row-major A layout increased compute-side shared-memory loads and pushed register use to 129/thread, reducing theoretical occupancy to one block/SM.

## Layer 4: explicit warp tiling

### What changed

Layer 4 keeps Layer 3's cp.async copy path and row-major A staging, but changes how the 256 threads are assigned to the 128x128 block tile.

Layer 3 effectively mapped each warp to a skinny `16x128` strip: two 8-row groups and all sixteen 8-column groups. Layer 4 makes the hierarchy explicit:

```text
block tile: 128x128
warp grid : 2 x 4 warps
warp tile : 64x32
lane grid : 8 x 4 lanes per warp
lane tile : 8x8
```

So the 8 warps in the CTA cover the block tile as:

```text
warp 0: rows   0..63, cols   0..31
warp 1: rows   0..63, cols  32..63
warp 2: rows   0..63, cols  64..95
warp 3: rows   0..63, cols  96..127
warp 4: rows  64..127, cols   0..31
warp 5: rows  64..127, cols  32..63
warp 6: rows  64..127, cols  64..95
warp 7: rows  64..127, cols  96..127
```

Each lane still computes one `8x8` register tile, so the fp32 math and store guards do not change. The A shared-memory skew also changes from a two-group pattern to a monotonic 16-group skew. That avoids overlap in the padded A buffer and spreads the eight row groups inside a warp tile across different bank starts.

### Why it helps

Warp tiling adds an explicit level between the block tile and the thread tile. The goal is to make each warp work on a more balanced 2D sub-tile instead of a long horizontal strip. Compared with Layer 3, a warp now has more M-side locality and less N-side span. That can improve scheduling and shared-memory access balance because B reuse and A reuse are less lopsided inside a warp.

This is still much simpler than CUTLASS/cuBLAS. It does not add warp-level MMA instructions, Tensor Cores, hand-scheduled SASS, or an autotuned tile search. It is a readable educational warp tile, not a production GEMM generator.

### Expected vs measured

Expected from the design doc: +3-5% over the cp.async layer if the warp-level reuse pattern improves ILP and shared-memory behavior.

Measured results:

| Size | cuda_reg_v2 Layer 4 GFLOPs | % cuBLAS | Verification |
| ---: | -------------------------: | -------: | :----------: |
| 1024^3 | 16251.5 | 48.9% | passed |
| 2048^3 | 34071.0 | 66.9% | passed |
| 4096^3 | 35646.9 | 76.1% | passed |

Same-run comparison against `cuda_reg`:

| Size | cuda_reg GFLOPs | cuda_reg_v2 L4 GFLOPs | Uplift |
| ---: | --------------: | --------------------: | -----: |
| 1024^3 | 11983.1 | 16251.5 | +35.6% |
| 2048^3 | 33113.8 | 34071.0 | +2.9% |
| 4096^3 | 33244.7 | 35646.9 | +7.2% |

Layer 4 improves over Layer 3 at every measured size, including 4096^3: 35646.9 GFLOPs vs 34726.2 GFLOPs. It still does not beat Layer 2 at the target size: Layer 2 measured 36056.5 GFLOPs. The honest conclusion is that the warp mapping helped, but not enough to repay the row-major A staging and high register pressure introduced for cp.async.

### ptxas / NCU

NCU remains blocked by `ERR_NVGPUCTRPERM`.

`ptxas --verbose` output:

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 130 registers, 20960 bytes smem, 388 bytes cmem[0]
```

There is still no spill, but the kernel remains register-limited to one 256-thread block per SM: `2 * 256 * 130 = 66560` registers would exceed a 65536-register file. That occupancy limit is the clearest ptxas-visible reason this educational path stalls in the mid-70% cuBLAS range instead of moving toward 90%.

### Defense checklist

- What is warp tiling? It splits the block tile into warp-owned sub-tiles, then splits each warp tile into lane-owned register tiles.
- Why `64x32` per warp? `2 x 4` warp tiles exactly cover `128x128`, and each `64x32` warp tile contains 32 lanes times one `8x8` lane tile.
- What changed versus Layer 3? Only work assignment and A-bank skew. The copy pipeline, fp32 arithmetic, and oracle/timing contract stay the same.
- Did it reach 90%? No. It recovered part of the cp.async regression, but 4096^3 reached 76.1% cuBLAS, below Layer 2's 77.0%.
- Why not keep pushing blindly? The next wins likely require deeper layout redesign, register-pressure reduction, autotuning, or assembly-level scheduling. Those are beyond this readable four-layer educational pass.
