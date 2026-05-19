#pragma once

#include "kernellab/backend/backend.hpp"
#include "kernellab/core/result.hpp"

#include <vector>

namespace kernellab {

BackendRunResult RunBackend(const Backend& backend,
                            const ProblemSpec& problem,
                            const RunOptions& options);

BackendRunResult VerifyBackend(const Backend& reference_backend,
                               const Backend& candidate_backend,
                               const ProblemSpec& problem,
                               const RunOptions& options,
                               const VerifyOptions& verify_options);

ComparisonReport CompareBackends(const Backend& reference_backend,
                                 const std::vector<const Backend*>& backends,
                                 const ProblemSpec& problem,
                                 const RunOptions& options,
                                 const VerifyOptions& verify_options);

}  // namespace kernellab
