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
