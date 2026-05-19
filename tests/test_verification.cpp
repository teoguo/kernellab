#include "kernellab/core/matrix.hpp"
#include "kernellab/verify/verification.hpp"

#include "test_support/check.hpp"

namespace {

void TestAcceptsWithinTolerance() {
  kernellab::Matrix reference(1, 2);
  kernellab::Matrix candidate(1, 2);

  reference(0, 0) = 1.0F;
  reference(0, 1) = -2.0F;
  candidate(0, 0) = 1.00001F;
  candidate(0, 1) = -2.00001F;

  const auto result = kernellab::VerifyMatrices(
      reference, candidate, kernellab::VerifyOptions{1e-4, 1e-4});

  KERNELLAB_CHECK(result.passed);
}

void TestRejectsOutOfToleranceValue() {
  kernellab::Matrix reference(1, 2);
  kernellab::Matrix candidate(1, 2);

  reference(0, 0) = 1.0F;
  reference(0, 1) = 2.0F;
  candidate(0, 0) = 1.2F;
  candidate(0, 1) = 2.0F;

  const auto result = kernellab::VerifyMatrices(
      reference, candidate, kernellab::VerifyOptions{1e-5, 1e-5});

  KERNELLAB_CHECK(!result.passed);
  KERNELLAB_CHECK(result.first_mismatch.has_value());
  KERNELLAB_CHECK(result.first_mismatch->row == 0);
  KERNELLAB_CHECK(result.first_mismatch->col == 0);
}

void TestDescribeVerificationFailureIncludesMismatchContext() {
  kernellab::Matrix reference(1, 1);
  kernellab::Matrix candidate(1, 1);

  reference(0, 0) = 1.0F;
  candidate(0, 0) = 1.5F;

  const auto result = kernellab::VerifyMatrices(
      reference, candidate, kernellab::VerifyOptions{1e-5, 1e-5});
  const auto description = kernellab::DescribeVerificationFailure(result);

  KERNELLAB_CHECK(!result.passed);
  KERNELLAB_CHECK(description.find("verification failed") != std::string::npos);
  KERNELLAB_CHECK(description.find("row=0") != std::string::npos);
  KERNELLAB_CHECK(description.find("col=0") != std::string::npos);
  KERNELLAB_CHECK(description.find("reference=1") != std::string::npos);
  KERNELLAB_CHECK(description.find("candidate=1.5") != std::string::npos);
}

void TestVerificationReportsAggregateErrorStats() {
  kernellab::Matrix reference(1, 3);
  kernellab::Matrix candidate(1, 3);

  reference(0, 0) = 1.0F;
  reference(0, 1) = 2.0F;
  reference(0, 2) = 4.0F;
  candidate(0, 0) = 1.2F;
  candidate(0, 1) = 2.0F;
  candidate(0, 2) = 3.0F;

  const auto result = kernellab::VerifyMatrices(
      reference, candidate, kernellab::VerifyOptions{1e-5, 1e-5});

  KERNELLAB_CHECK(!result.passed);
  KERNELLAB_CHECK(result.first_mismatch.has_value());
  KERNELLAB_CHECK(result.count_over_tolerance == 2);
  KERNELLAB_CHECK(result.max_absolute_error == 1.0);
  KERNELLAB_CHECK(result.max_relative_error > 0.24);
}

void TestDefaultToleranceScalesWithProblemSize() {
  const auto small = kernellab::DefaultTolerance(kernellab::ProblemSpec{64, 64, 16, 1});
  const auto large = kernellab::DefaultTolerance(kernellab::ProblemSpec{64, 64, 4096, 1});

  KERNELLAB_CHECK(small.atol >= 0.0);
  KERNELLAB_CHECK(small.rtol >= 0.0);
  KERNELLAB_CHECK(large.atol > small.atol);
  KERNELLAB_CHECK(large.rtol >= small.rtol);
}

}  // namespace

int main() {
  TestAcceptsWithinTolerance();
  TestRejectsOutOfToleranceValue();
  TestDescribeVerificationFailureIncludesMismatchContext();
  TestVerificationReportsAggregateErrorStats();
  TestDefaultToleranceScalesWithProblemSize();
  return 0;
}
