#!/usr/bin/env python3
"""Convert an Nsight Systems CUDA kernel summary CSV into kernel_breakdown.csv.

Recommended producer command:

    nsys stats traces/trace_qwen.nsys-rep \
      --report cuda_gpu_kern_sum \
      --format csv \
      --output results/nsys_cuda_gpu_kern_sum

Nsight writes a CSV whose exact column names vary by version. This parser
handles the common `Time (%)`, `Total Time (ns)`, `Instances`, and `Name`
schema and emits a compact top-N table for the atlas writeup.
"""

import argparse
import csv
import os


def pick(row, candidates):
    lower = {k.lower().strip(): k for k in row}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower:
            return row[lower[key]]
    raise KeyError(f"None of {candidates} found in columns {list(row)}")


def category_for(name):
    n = name.lower()
    if any(s in n for s in ("gemm", "cublas", "matmul", "hmma", "hgemm", "sgemm")):
        return "GEMM"
    if any(s in n for s in ("attention", "flash", "fmha", "scaled_dot")):
        return "Attention"
    if "softmax" in n:
        return "Softmax"
    if any(s in n for s in ("layernorm", "rmsnorm", "norm")):
        return "Norm"
    if any(s in n for s in ("silu", "gelu", "relu", "elementwise", "add", "mul", "cast")):
        return "Elementwise"
    return "Other"


def as_float(value):
    return float(str(value).replace(",", "").strip())


def as_int(value):
    return int(float(str(value).replace(",", "").strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Nsight cuda_gpu_kern_sum CSV")
    ap.add_argument("--out", default="results/kernel_breakdown.csv")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r or all(v in (None, "") for v in r.values()):
                continue
            name = pick(r, ["Name", "Kernel Name"])
            total_ns = as_float(pick(r, ["Total Time (ns)", "Total Time"]))
            calls = as_int(pick(r, ["Instances", "Calls", "Count"]))
            pct = as_float(pick(r, ["Time (%)", "Time%"]))
            rows.append({
                "kernel_name": name,
                "calls": calls,
                "total_ms": round(total_ns / 1e6, 4),
                "pct_total": round(pct, 2),
                "category": category_for(name),
            })

    rows.sort(key=lambda r: r["total_ms"], reverse=True)
    rows = rows[:args.top]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["kernel_name", "calls", "total_ms", "pct_total", "category"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
