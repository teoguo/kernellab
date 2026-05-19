# Methodology

`kernellab` reports two timing families for each backend iteration:

- `kernel_ms`
  Device/kernel time for a single prepared session run. For CPU backends this matches the measured compute section. For CUDA backends this is measured with CUDA events around the launched kernel or cuBLAS call on a dedicated session stream.
- `e2e_ms`
  End-to-end time for one prepared session iteration, including any output materialization needed for verification and reporting. It excludes `Prepare()` costs such as device allocation, host-to-device upload, and cuBLAS handle creation.

This distinction matters:

- `kernel_ms` is the number to use when comparing pure kernel performance.
- `e2e_ms` is the number to use when comparing the real cost of one reusable execution session.

`kernellab` intentionally does **not** hide this difference because a framework that reports only one number would encourage misleading cross-backend conclusions.

## Verification

Verification is always against `cpu_ref`, which is the higher-precision correctness oracle in v2 foundations.

- `cpu_ref`
  Accumulates in `double` and serves as the oracle.
- `cpu_naive`
  Scalar `float32` CPU baseline for performance comparison.

Verification reports:

- first mismatch location and values
- maximum absolute error
- maximum relative error
- count of elements above tolerance

Default tolerances scale with problem size through `DefaultTolerance(problem)` so larger `K` values are not forced through an unrealistically strict fixed threshold.

## Export Schema

Versioned reports currently use `schema_version = 3`.

Every JSON report includes:

- `environment`
- `problem`
- `run_options`
- `verify_options`
- per-result `iterations`
- per-result `gflops`
- per-result `pct_of_cublas`
- per-result `summary.kernel_ms`
- per-result `summary.e2e_ms`
- `verification_status`

CSV output includes the same core information in a flat, automation-friendly format.

## Throughput

For every result, `kernellab` also reports `gflops = 2 * M * N * K /
(kernel_mean_ms * 1e6)`. This uses the mean kernel time only — `e2e_ms`
deliberately is not the throughput basis because it would penalize
backends with a heavier D2H step (e.g. a kernel that returns the full
output matrix) versus those whose output is already in the right place.
For comparing pure kernel performance use the GFLOPs column; for
comparing real-world reusable-session cost, weight by `e2e_ms`.

When comparing through the `compare` subcommand, kernellab prints a
`pct_of_cublas` column when cuBLAS is in the candidate list. JSON and
CSV exports include the same derived value, using `null`/empty when a
successful cuBLAS row is absent. This is purely a presentation
convenience — the underlying `gflops` numbers are the source of truth.

The cuBLAS backend explicitly sets `CUBLAS_PEDANTIC_MATH` after handle
creation. That makes the cuBLAS row a strict fp32 SGEMM baseline rather
than relying on CUDA's default math mode, which may use lower-precision
Tensor Core paths on architectures that support them.

## Oracle handling in `compare`

The reference backend (`cpu_ref`) is run exactly once when `compare` is
invoked, and only to produce the matrix used for element-wise
verification. The reference row's timing in the report is populated by
running it through the normal benchmark loop when it appears in the
candidate list, but the oracle execution itself does **not** consume
benchmark iterations. This keeps `compare` at K=4096 to roughly one
oracle pass plus per-candidate cost, instead of `warmup + timed`
oracle passes.

## Benchmark Notes

For stable comparisons:

- use warmups
- prefer multiple timed iterations
- compare the same problem and seed across backends
- interpret `verification_status=reference` as the reference row, not as a passed comparison

On CUDA systems, session preparation amortizes one-time costs. That makes the reported timed iterations much closer to what users expect from a serious GEMM benchmarking tool.
