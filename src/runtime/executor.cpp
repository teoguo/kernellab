#include "kernellab/runtime/executor.hpp"

#include "kernellab/benchmark/stats.hpp"
#include "kernellab/runtime/input_generator.hpp"
#include "kernellab/verify/verification.hpp"

#include <chrono>
#include <utility>

namespace kernellab {

namespace {

struct BackendExecution {
  BackendRunResult result;
  Matrix output;
};

// Run the oracle exactly once to produce a reference output. No
// per-iteration timing, no warmup -- we only need the output matrix.
// Without this, calling CompareBackends at 1024^3+ wastes minutes
// running cpu_ref through `warmup + timed` iterations of a fp64 triple
// loop just to throw away every iteration's timing except for stats.
BackendExecution ExecuteOracleOnce(const Backend& backend, const GeneratedInputs& inputs) {
  BackendExecution execution;
  execution.result.backend_id = std::string(backend.id());
  execution.result.available = backend.is_available();
  execution.output = Matrix(inputs.a.rows(), inputs.b.cols());

  if (!execution.result.available) {
    execution.result.error_message = "oracle backend is unavailable";
    return execution;
  }
  BackendSessionPtr session;
  const auto prepare_status = backend.Prepare(inputs.a, inputs.b, session);
  if (!prepare_status.ok) {
    execution.result.error_message = prepare_status.message;
    return execution;
  }
  if (session == nullptr) {
    execution.result.error_message = "oracle backend prepare returned no session";
    return execution;
  }
  IterationMeasurement measurement;
  const auto status = session->Run(execution.output, measurement);
  if (!status.ok) {
    execution.result.error_message = status.message;
    return execution;
  }
  execution.result.success = true;
  return execution;
}

void AttachVerificationFailureMessage(BackendRunResult& result) {
  if (!result.verification.has_value() || result.verification->passed) {
    return;
  }
  if (!result.error_message.empty()) {
    return;
  }
  result.error_message = DescribeVerificationFailure(*result.verification);
}

BackendExecution ExecuteWithInputs(const Backend& backend,
                                   const GeneratedInputs& inputs,
                                   const ProblemSpec& problem,
                                   const RunOptions& options) {
  BackendExecution execution;
  execution.result.backend_id = std::string(backend.id());
  execution.result.available = backend.is_available();
  execution.output = Matrix(inputs.a.rows(), inputs.b.cols());

  if (!execution.result.available) {
    execution.result.error_message = "backend is unavailable";
    return execution;
  }

  if (options.timed_iterations <= 0) {
    execution.result.error_message = "timed_iterations must be positive";
    return execution;
  }

  BackendSessionPtr session;
  const auto prepare_status = backend.Prepare(inputs.a, inputs.b, session);
  if (!prepare_status.ok) {
    execution.result.error_message = prepare_status.message;
    return execution;
  }
  if (session == nullptr) {
    execution.result.error_message = "backend prepare returned no session";
    return execution;
  }

  for (std::int32_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    Matrix warmup_output(inputs.a.rows(), inputs.b.cols());
    IterationMeasurement measurement;
    const auto status = session->Run(warmup_output, measurement);
    if (!status.ok) {
      execution.result.error_message = status.message;
      return execution;
    }
  }

  std::vector<double> kernel_timings_ms;
  kernel_timings_ms.reserve(static_cast<std::size_t>(options.timed_iterations));
  execution.result.iterations.reserve(static_cast<std::size_t>(options.timed_iterations));
  execution.result.timings_ms.reserve(static_cast<std::size_t>(options.timed_iterations));
  for (std::int32_t iteration = 0; iteration < options.timed_iterations; ++iteration) {
    IterationMeasurement measurement;
    const auto status = session->Run(execution.output, measurement);
    if (!status.ok) {
      execution.result.error_message = status.message;
      return execution;
    }

    execution.result.iterations.push_back(measurement);
    execution.result.timings_ms.push_back(measurement.e2e_ms);
    kernel_timings_ms.push_back(measurement.kernel_ms);
  }

  execution.result.kernel_statistics = ComputeSummaryStatistics(kernel_timings_ms);
  execution.result.statistics = ComputeSummaryStatistics(execution.result.timings_ms);
  execution.result.success = true;
  execution.result.gflops =
      ComputeGflops(problem, execution.result.kernel_statistics.mean);
  return execution;
}

}  // namespace

BackendRunResult RunBackend(const Backend& backend,
                            const ProblemSpec& problem,
                            const RunOptions& options) {
  const auto inputs = GenerateInputs(problem);
  return ExecuteWithInputs(backend, inputs, problem, options).result;
}

BackendRunResult VerifyBackend(const Backend& reference_backend,
                               const Backend& candidate_backend,
                               const ProblemSpec& problem,
                               const RunOptions& options,
                               const VerifyOptions& verify_options) {
  const auto inputs = GenerateInputs(problem);
  const auto reference_execution = ExecuteOracleOnce(reference_backend, inputs);
  if (!reference_execution.result.success) {
    return reference_execution.result;
  }

  auto candidate_execution = ExecuteWithInputs(candidate_backend, inputs, problem, options);
  if (candidate_execution.result.success) {
    candidate_execution.result.verification = VerifyMatrices(
        reference_execution.output, candidate_execution.output, verify_options);
    AttachVerificationFailureMessage(candidate_execution.result);
  }

  return candidate_execution.result;
}

ComparisonReport CompareBackends(const Backend& reference_backend,
                                 const std::vector<const Backend*>& backends,
                                 const ProblemSpec& problem,
                                 const RunOptions& options,
                                 const VerifyOptions& verify_options) {
  ComparisonReport report;
  report.environment = CollectEnvironmentInfo();
  report.problem = problem;
  report.run_options = options;
  report.verify_options = verify_options;

  const auto inputs = GenerateInputs(problem);
  // Run the oracle ONCE for verification (no warmup, no multi-iter
  // timing). If the user also asked for cpu_ref in the candidate list,
  // we still run it through full benchmark timing below in the loop;
  // this oracle pass is solely about producing the reference output.
  const auto reference_execution = ExecuteOracleOnce(reference_backend, inputs);

  for (const Backend* backend : backends) {
    if (backend == nullptr) {
      continue;
    }

    auto candidate_execution = ExecuteWithInputs(*backend, inputs, problem, options);
    if (backend == &reference_backend && candidate_execution.result.success) {
      // User asked for the oracle in the comparison list. Mark it as
      // the reference row so consumers don't treat the (empty)
      // verification as "not run".
      candidate_execution.result.is_reference = true;
      report.results.push_back(std::move(candidate_execution.result));
      continue;
    }
    if (reference_execution.result.success && candidate_execution.result.success) {
      candidate_execution.result.verification = VerifyMatrices(
          reference_execution.output, candidate_execution.output, verify_options);
      AttachVerificationFailureMessage(candidate_execution.result);
    }
    report.results.push_back(std::move(candidate_execution.result));
  }

  return report;
}

}  // namespace kernellab
