# LLM GEMM Atlas

LLM GEMM Atlas connects kernel-level GEMM efficiency with end-to-end LLM
inference traces. The question is simple:

> How much of inference time lands in GEMM, and how much of that GEMM work is
> wasted on memory-bound decode shapes?

This module is intentionally narrow. It does not implement a serving stack,
multi-GPU training, FlashAttention, or TensorRT-LLM replacement. It measures
cuBLAS across transformer-shaped GEMMs, profiles a real Hugging Face inference
run with Nsight Systems, and maps the two views together.

## Setup

Record the exact environment before interpreting the numbers:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("tf32", torch.backends.cuda.matmul.allow_tf32)
PY
nsys --version
```

Target run used by the project plan:

- Hardware: NVIDIA RTX 6000 Ada Generation
- Model: `Qwen/Qwen2.5-1.5B`
- Microbench dtype focus: fp16, with fp32 and bf16 included for comparison
- Inference mode: greedy decode, `use_cache=True`, 128 generated tokens

## Methodology

### Microbench

`atlas/bench_transformer_gemm.py` builds transformer-shaped GEMMs and times
`torch.matmul`, which dispatches to cuBLAS. It uses CUDA events around the timed
loop, with warmup iterations before measurement, so `time_ms` is device time.

The FLOP model is:

```text
flops = 2 * M * N * K
gflops = flops / seconds / 1e9
pct_peak = gflops / theoretical_peak_gflops * 100
```

Strict fp32 is the default: TF32 is disabled unless `--allow-tf32` is passed.

### End-to-End Profile

`atlas/infer_with_profile.py` loads a small causal LM and wraps the measured
region with `cudaProfilerStart` / `cudaProfilerStop`, matching the Nsight
Systems capture range:

```bash
nsys profile \
  -o traces/trace_qwen \
  --capture-range=cudaProfilerApi \
  --cuda-memory-usage=true \
  python atlas/infer_with_profile.py \
    --model Qwen/Qwen2.5-1.5B \
    --max-new-tokens 128
```

The default path performs an explicit prefill forward pass followed by a
token-by-token decode loop. NVTX ranges label `prefill`, `decode_loop`, and
`decode_token`, so Nsight can separate the first large GEMM burst from the
repeated decode kernels.

## Microbench Results

Generate the table:

```bash
python atlas/bench_transformer_gemm.py \
  --hardware RTX6000Ada \
  --out results/transformer_gemm_atlas.csv
```

Expected output artifact:

- `results/transformer_gemm_atlas.csv`

Expected plots:

```bash
python atlas/plot_atlas.py \
  --csv results/transformer_gemm_atlas.csv \
  --outdir docs/img
```

- `docs/img/atlas_pct_peak_by_phase.png`
- `docs/img/atlas_gflops_by_shape.png`

Paste the measured headline here after running on the GPU host:

```text
HEADLINE (fp16): prefill avg <X>% peak vs decode avg <Y>% peak (<Z>x gap)
```

## End-to-End Kernel Breakdown

Export top kernels from the Nsight report:

```bash
nsys stats traces/trace_qwen.nsys-rep \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output results/nsys_cuda_gpu_kern_sum

python atlas/parse_nsys_kernels.py \
  --csv results/nsys_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv \
  --out results/kernel_breakdown.csv
```

Expected output artifact:

- `results/kernel_breakdown.csv`

If `traces/trace_qwen.nsys-rep` is larger than 50 MB, leave it uncommitted and
commit the exported stats plus screenshots instead.

## Microbench to E2E Mapping

Use this table once `results/kernel_breakdown.csv` and
`results/transformer_gemm_atlas.csv` exist:

| E2E kernel pattern | Category | Matched atlas shape | Atlas fp16 % peak | E2E total ms | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| `*gemm*` / `*cublas*` / `*hmma*` | GEMM | prefill or decode projection | TBD | TBD | Map by timeline phase and matrix shape |
| `*attention*` / `*fmha*` / `*flash*` | Attention | N/A | N/A | TBD | Separate from GEMM-dominated projections |

## Findings To Fill From A Real Run

The intended engineering conclusions are:

1. Decode GEMMs should achieve far lower peak utilization than prefill GEMMs
   because `M=1` shapes have low arithmetic intensity and poor Tensor Core
   occupancy.
2. Prefill projection and FFN GEMMs should be the closest to the hardware roof
   because they expose large, dense matrix shapes to cuBLAS.
3. A nontrivial share of E2E decode time should land in GEMM kernels even when
   the GEMMs show poor `% peak`, which is the skinny-shape penalty this atlas is
   designed to make visible.
4. The practical serving implication is batching or continuous batching:
   increasing effective `M` turns repeated skinny decode GEMMs into fatter
   shapes that amortize weight reads better.

Do not fill this section with estimates. Commit only numbers produced by the
commands above on the target GPU host.
