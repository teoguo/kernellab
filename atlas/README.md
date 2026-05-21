# LLM GEMM Atlas

LLM GEMM Atlas connects two views of inference performance:

1. Microbench: transformer-shaped GEMMs measured directly with cuBLAS.
2. E2E: Nsight Systems traces from a real Hugging Face model inference run.

The goal is not to beat TensorRT-LLM. The goal is to identify which GEMM
shapes appear in real inference and quantify how much efficiency is lost in
memory-bound decode shapes.

## Setup

```bash
python -m pip install -r atlas/requirements.txt
nvidia-smi --query-gpu=name --format=csv
nsys --version
```

## 1. Transformer GEMM Microbench

RTX 6000 Ada:

```bash
python atlas/bench_transformer_gemm.py \
  --hardware RTX6000Ada \
  --out results/transformer_gemm_atlas.csv
```

Qwen2.5-1.5B-shaped GEMMs:

```bash
python atlas/bench_transformer_gemm.py \
  --hardware RTX6000Ada \
  --model-dim 1536 \
  --ffn-dim 8960 \
  --out results/atlas_qwen.csv
```

The script prints a headline such as:

```text
HEADLINE (fp16): prefill avg 84.3% peak vs decode avg 0.9% peak (89.9x gap)
```

## 2. End-to-End Inference Profile

Sanity check first:

```bash
python atlas/infer_with_profile.py \
  --model Qwen/Qwen2.5-1.5B \
  --max-new-tokens 64
```

Capture the profiled region with Nsight Systems:

```bash
nsys profile \
  -o traces/trace_qwen \
  --capture-range=cudaProfilerApi \
  --cuda-memory-usage=true \
  python atlas/infer_with_profile.py \
    --model Qwen/Qwen2.5-1.5B \
    --max-new-tokens 128
```

Export top kernels:

```bash
nsys stats traces/trace_qwen.nsys-rep \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output results/nsys_cuda_gpu_kern_sum

python atlas/parse_nsys_kernels.py \
  --csv results/nsys_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv \
  --out results/kernel_breakdown.csv
```

If `traces/trace_qwen.nsys-rep` is larger than 50 MB, do not commit it. Commit
the exported stats CSV and screenshots instead.

## 3. Plots

```bash
python atlas/plot_atlas.py \
  --csv results/transformer_gemm_atlas.csv \
  --outdir docs/img
```

Then update `docs/atlas.md` with the run-specific setup, microbench table,
kernel breakdown, screenshots, and findings.
