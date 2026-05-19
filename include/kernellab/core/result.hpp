#pragma once

#include "kernellab/backend/session.hpp"
#include "kernellab/benchmark/stats.hpp"
#include "kernellab/core/problem.hpp"
#include "kernellab/core/status.hpp"
#include "kernellab/verify/verification.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kernellab {

struct RunOptions {
  std::int32_t warmup_iterations = 0;
  std::int32_t timed_iterations = 1;
};

struct EnvironmentInfo {
  std::string timestamp_utc;
  std::string cpu_model;
  std::string gpu_name;
  int openmp_threads = 0;
};

struct BackendRunResult {
  std::string backend_id;
  bool available = false;
  bool success = false;
  bool is_reference = false;
  std::string error_message;
  std::vector<IterationMeasurement> iterations;
  std::vector<double> timings_ms;
  SummaryStatistics kernel_statistics;
  SummaryStatistics statistics;
  // 2 * M * N * K / (kernel_mean_ms * 1e6). 0 when timing is unavailable
  // (success == false or kernel_statistics.mean <= 0).
  double gflops = 0.0;
  std::optional<VerificationResult> verification;
};

struct ComparisonReport {
  int schema_version = 3;
  EnvironmentInfo environment;
  ProblemSpec problem;
  RunOptions run_options;
  VerifyOptions verify_options;
  std::vector<BackendRunResult> results;
};

EnvironmentInfo CollectEnvironmentInfo();

// GEMM throughput in GFLOPs/s from a kernel time in milliseconds.
// GEMM cost is 2 * M * N * K flops (one multiply + one add per inner step).
// Returns 0.0 if any input is non-positive (e.g. failed run, empty problem).
double ComputeGflops(const ProblemSpec& problem, double kernel_ms) noexcept;

} // namespace kernellab
