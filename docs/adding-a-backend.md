# Adding a Backend

This is the shortest path to adding a new backend in the v2 foundations model.

## 1. Implement `Backend`

Create a backend class that provides:

- `id()`
- `is_available()`
- `Prepare(const Matrix& a, const Matrix& b, BackendSessionPtr& session)`

`Prepare()` should validate shapes and create a session object. It should not perform per-iteration work.

## 2. Implement `BackendSession`

Your session must implement:

- `Status Run(Matrix& output, IterationMeasurement& measurement)`

`Run()` is one timed iteration on a prepared session.

- For CPU backends:
  run the compute body and fill both `measurement.kernel_ms` and `measurement.e2e_ms`
- For CUDA backends:
  allocate/upload/create handles in session initialization
  use a dedicated stream
  measure kernel time with CUDA events
  materialize output before returning

## 3. Register the Backend

Add a factory in [`src/runtime/registry.cpp`](../src/runtime/registry.cpp) and register it in `CreateDefaultRegistry()`.

Keep ids stable and machine-friendly, for example:

- `cpu_naive`
- `cuda_naive`
- `cublas`

## 4. Add Tests

At minimum add:

- one known-small correctness test if the backend is always available
- one executor/verification path test if behavior changes
- one CLI smoke test if user-facing behavior changes

If the backend is conditional, make the test resilient to unavailable environments.

## 5. Update Docs

Update:

- [`README.md`](../README.md)
- [`docs/methodology.md`](methodology.md)

If the backend introduces new timing or correctness semantics, document them before relying on them in exported artifacts.
