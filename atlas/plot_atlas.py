#!/usr/bin/env python3
"""Plot the GEMM efficiency atlas from the microbench CSV.

Usage:
    python plot_atlas.py --csv results/transformer_gemm_atlas.csv --outdir docs/img

Produces:
    docs/img/atlas_pct_peak_by_phase.png   percent of peak, prefill vs decode, per dtype
    docs/img/atlas_gflops_by_shape.png     GFLOPS per shape (fp16)
"""

import argparse
import csv
import os
from collections import defaultdict


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["pct_peak"] = float(r["pct_peak"])
            r["gflops"] = float(r["gflops"])
            rows.append(r)
    return rows


def plot_pct_peak_by_phase(rows, outdir):
    """Grouped bar: avg % peak for prefill vs decode, per dtype."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = defaultdict(list)  # (phase, dtype) -> [pct...]
    for r in rows:
        agg[(r["phase"], r["dtype"])].append(r["pct_peak"])
    dtypes = sorted({r["dtype"] for r in rows})
    phases = ["prefill", "decode"]

    import numpy as np
    x = np.arange(len(dtypes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, phase in enumerate(phases):
        vals = [sum(agg[(phase, d)]) / len(agg[(phase, d)]) if agg[(phase, d)] else 0
                for d in dtypes]
        ax.bar(x + (i - 0.5) * w, vals, w, label=phase)
    ax.set_xticks(x)
    ax.set_xticklabels(dtypes)
    ax.set_ylabel("% of theoretical peak")
    ax.set_title("cuBLAS efficiency: prefill vs decode (avg over shapes)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(outdir, "atlas_pct_peak_by_phase.png")
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print("wrote", path)


def plot_gflops_by_shape(rows, outdir, dtype="fp16"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = [r for r in rows if r["dtype"] == dtype]
    sub.sort(key=lambda r: r["gflops"])
    names = [r["shape_name"] for r in sub]
    vals = [r["gflops"] for r in sub]
    colors = ["tab:red" if r["phase"] == "decode" else "tab:blue" for r in sub]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, vals, color=colors)
    ax.set_xlabel(f"GFLOPS ({dtype})")
    ax.set_title(f"Achieved GFLOPS per transformer GEMM shape ({dtype})\n"
                 "red = decode (memory-bound), blue = prefill (compute-bound)")
    ax.grid(axis="x", alpha=0.3)
    path = os.path.join(outdir, "atlas_gflops_by_shape.png")
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/transformer_gemm_atlas.csv")
    ap.add_argument("--outdir", default="docs/img")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = load(args.csv)
    plot_pct_peak_by_phase(rows, args.outdir)
    plot_gflops_by_shape(rows, args.outdir)


if __name__ == "__main__":
    main()
