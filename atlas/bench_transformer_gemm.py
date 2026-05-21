#!/usr/bin/env python3
"""Transformer-shaped GEMM microbenchmark.

This is the "microbench" half of LLM GEMM Atlas. It measures cuBLAS
efficiency via torch.matmul across the GEMM shapes that appear in
transformer inference: prefill shapes with M=sequence length and decode
shapes with M=1.

No CUDA compilation is needed. Timing uses CUDA events, so the reported
time is device time rather than Python wall-clock loop overhead.
"""

import argparse
import csv
import os

# Theoretical dense peak FLOPS (TFLOPS). fp32 is CUDA-core SGEMM; fp16/bf16
# uses Tensor Cores. Keep fp32 TF32 disabled unless --allow-tf32 is passed.
PEAK_TFLOPS = {
    "RTX4090":    {"fp32": 82.6,  "fp16": 165.2, "bf16": 165.2},
    "RTX6000Ada": {"fp32": 91.1,  "fp16": 182.0, "bf16": 182.0},
}

HARDWARE_ALIASES = {
    "NVIDIA GeForce RTX 4090": "RTX4090",
    "NVIDIA RTX 6000 Ada Generation": "RTX6000Ada",
}

DTYPE_NAMES = ("fp32", "fp16", "bf16")


def build_shapes(D, ffn, seq_prefill):
    """Return list of (name, phase, M, N, K)."""
    qkv = 3 * D
    return [
        # ---- prefill: M = seq_len (compute-bound regime) ----
        ("prefill_qkv",     "prefill", seq_prefill, qkv,  D),
        ("prefill_outproj", "prefill", seq_prefill, D,    D),
        ("prefill_ffn_up",  "prefill", seq_prefill, ffn,  D),
        ("prefill_ffn_down","prefill", seq_prefill, D,    ffn),
        # ---- decode: M = 1 (memory-bound regime) ----
        ("decode_qkv",      "decode",  1,           qkv,  D),
        ("decode_outproj",  "decode",  1,           D,    D),
        ("decode_ffn_up",   "decode",  1,           ffn,  D),
        ("decode_ffn_down", "decode",  1,           D,    ffn),
    ]


def resolve_hardware(torch, name):
    if name != "auto":
        return name
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name in HARDWARE_ALIASES:
        return HARDWARE_ALIASES[gpu_name]
    raise SystemExit(
        f"Unknown GPU '{gpu_name}'. Add it to PEAK_TFLOPS or pass --hardware explicitly."
    )


def bench_one(torch, M, N, K, dtype, n_iter=100, warmup=20):
    """Benchmark one [M,K] @ [K,N] GEMM. Returns (time_ms, gflops)."""
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        _ = a @ b
    end.record()
    end.synchronize()

    total_ms = start.elapsed_time(end)
    time_ms = total_ms / n_iter
    flops = 2.0 * M * N * K
    gflops = flops / (time_ms / 1e3) / 1e9
    return time_ms, gflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default="auto", choices=["auto"] + list(PEAK_TFLOPS))
    ap.add_argument("--model-dim", type=int, default=4096, help="hidden dim D")
    ap.add_argument("--ffn-dim", type=int, default=11008, help="FFN intermediate")
    ap.add_argument("--seq-prefill", type=int, default=2048)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--dtypes", default="fp32,fp16,bf16",
                    help="comma-separated subset of fp32,fp16,bf16")
    ap.add_argument("--allow-tf32", action="store_true",
                    help="allow TF32 for fp32 matmul; default is strict fp32")
    ap.add_argument("--out", default="results/transformer_gemm_atlas.csv")
    args = ap.parse_args()

    import torch

    dtypes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    assert torch.cuda.is_available(), "No CUDA device found."
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    hardware = resolve_hardware(torch, args.hardware)
    dtype_names = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    for d in dtype_names:
        if d not in DTYPE_NAMES:
            raise SystemExit(f"Unknown dtype '{d}'. Choose from {sorted(DTYPE_NAMES)}")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Peak TFLOPS table entry: {hardware} -> {PEAK_TFLOPS[hardware]}")
    print(f"TF32 enabled for fp32 matmul: {torch.backends.cuda.matmul.allow_tf32}")

    shapes = build_shapes(args.model_dim, args.ffn_dim, args.seq_prefill)
    rows = []

    for name, phase, M, N, K in shapes:
        for dname in dtype_names:
            dtype = dtypes[dname]
            try:
                time_ms, gflops = bench_one(
                    torch, M, N, K, dtype, n_iter=args.n_iter, warmup=args.warmup
                )
            except RuntimeError as e:
                print(f"  SKIP {name}/{dname}: {e}")
                torch.cuda.empty_cache()
                continue
            peak = PEAK_TFLOPS[hardware][dname] * 1e3
            pct = 100.0 * gflops / peak
            rows.append({
                "shape_name": name, "phase": phase,
                "M": M, "N": N, "K": K, "dtype": dname,
                "time_ms": round(time_ms, 4),
                "gflops": round(gflops, 1),
                "pct_peak": round(pct, 2),
                "hardware": hardware,
                "tf32": bool(args.allow_tf32),
                "warmup": args.warmup,
                "n_iter": args.n_iter,
            })
            print(f"  {name:18s} {dname:4s}  {time_ms:8.3f} ms  "
                  f"{gflops:9.0f} GFLOPS  {pct:5.1f}% peak")

    if not rows:
        raise SystemExit("No benchmark rows produced.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")

    # quick headline finding
    decode = [r for r in rows if r["phase"] == "decode" and r["dtype"] == "fp16"]
    prefill = [r for r in rows if r["phase"] == "prefill" and r["dtype"] == "fp16"]
    if decode and prefill:
        avg_decode = sum(r["pct_peak"] for r in decode) / len(decode)
        avg_prefill = sum(r["pct_peak"] for r in prefill) / len(prefill)
        print(f"\nHEADLINE (fp16): prefill avg {avg_prefill:.1f}% peak vs "
              f"decode avg {avg_decode:.1f}% peak  "
              f"({avg_prefill/max(avg_decode,1e-9):.1f}x gap)")


if __name__ == "__main__":
    main()
