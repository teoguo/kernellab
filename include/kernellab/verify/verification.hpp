#pragma once

#include "kernellab/core/problem.hpp"
#include "kernellab/core/matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace kernellab {

struct VerifyOptions {
  double atol = 1e-5;
  double rtol = 1e-5;
};

struct MismatchInfo {
  std::int64_t row = 0;
  std::int64_t col = 0;
  float reference_value = 0.0F;
  float candidate_value = 0.0F;
  double absolute_error = 0.0;
  double relative_error = 0.0;
};

struct VerificationResult {
  bool passed = true;
  std::optional<MismatchInfo> first_mismatch;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  std::size_t count_over_tolerance = 0;
};

VerifyOptions DefaultTolerance(const ProblemSpec& problem);
VerificationResult VerifyMatrices(const Matrix& reference,
                                  const Matrix& candidate,
                                  const VerifyOptions& options);
std::string DescribeVerificationFailure(const VerificationResult& result);

}  // namespace kernellab
