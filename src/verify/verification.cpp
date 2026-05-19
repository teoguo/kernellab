#include "kernellab/verify/verification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace kernellab {

VerifyOptions DefaultTolerance(const ProblemSpec& problem) {
  // Element-wise error for fp32 GEMM with random inputs in [-1, 1] is
  // bounded by O(eps * sqrt(K)) for the typical case and O(eps * K) in
  // the worst case. Different backends use different summation orders;
  // we want the verifier to admit those differences but still catch
  // real kernel bugs. The constant `32` corresponds to ~5 sigma of the
  // sqrt(K) noise model and keeps fp32 GEMM verification stable up to
  // K ~ 32k.
  const double scale =
      std::max(1.0, std::sqrt(static_cast<double>(std::max<std::int64_t>(1, problem.k))));
  const double base = std::numeric_limits<float>::epsilon() * scale * 32.0;
  return VerifyOptions{base, base};
}

VerificationResult VerifyMatrices(const Matrix& reference,
                                  const Matrix& candidate,
                                  const VerifyOptions& options) {
  VerificationResult result;

  if (reference.rows() != candidate.rows() || reference.cols() != candidate.cols()) {
    result.passed = false;
    result.first_mismatch = MismatchInfo{
        0,
        0,
        0.0F,
        0.0F,
        0.0,
        0.0,
    };
    return result;
  }

  for (std::int64_t row = 0; row < reference.rows(); ++row) {
    for (std::int64_t col = 0; col < reference.cols(); ++col) {
      const float reference_value = reference(row, col);
      const float candidate_value = candidate(row, col);
      const double absolute_error =
          std::abs(static_cast<double>(candidate_value) - static_cast<double>(reference_value));
      const double relative_error =
          absolute_error / std::max(std::abs(static_cast<double>(reference_value)), 1e-12);
      result.max_absolute_error = std::max(result.max_absolute_error, absolute_error);
      result.max_relative_error = std::max(result.max_relative_error, relative_error);
      const double tolerance =
          options.atol + (options.rtol * std::abs(static_cast<double>(reference_value)));
      if (absolute_error > tolerance) {
        ++result.count_over_tolerance;
        result.passed = false;
        if (!result.first_mismatch.has_value()) {
          result.first_mismatch = MismatchInfo{
              row,
              col,
              reference_value,
              candidate_value,
              absolute_error,
              relative_error,
          };
        }
      }
    }
  }

  return result;
}

std::string DescribeVerificationFailure(const VerificationResult& result) {
  if (result.passed) {
    return "verification passed";
  }
  if (!result.first_mismatch.has_value()) {
    return "verification failed";
  }

  const auto& mismatch = *result.first_mismatch;
  std::ostringstream output;
  output << "verification failed"
         << " row=" << mismatch.row
         << " col=" << mismatch.col
         << " reference=" << mismatch.reference_value
         << " candidate=" << mismatch.candidate_value
         << " abs_error=" << mismatch.absolute_error
         << " rel_error=" << mismatch.relative_error
         << " max_abs_error=" << result.max_absolute_error
         << " max_rel_error=" << result.max_relative_error
         << " count_over_tolerance=" << result.count_over_tolerance;
  return output.str();
}

}  // namespace kernellab
