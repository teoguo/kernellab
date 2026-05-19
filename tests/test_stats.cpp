#include "kernellab/benchmark/stats.hpp"

#include "test_support/check.hpp"

#include <cmath>
#include <vector>

namespace {

void TestComputesMeanMedianMinMaxAndStddev() {
  const std::vector<double> samples{1.0, 2.0, 3.0, 4.0};
  const auto stats = kernellab::ComputeSummaryStatistics(samples);

  KERNELLAB_CHECK(stats.count == 4);
  KERNELLAB_CHECK(std::abs(stats.mean - 2.5) < 1e-12);
  KERNELLAB_CHECK(std::abs(stats.median - 2.5) < 1e-12);
  KERNELLAB_CHECK(std::abs(stats.min - 1.0) < 1e-12);
  KERNELLAB_CHECK(std::abs(stats.max - 4.0) < 1e-12);
  KERNELLAB_CHECK(std::abs(stats.stddev - std::sqrt(1.25)) < 1e-12);
}

}  // namespace

int main() {
  TestComputesMeanMedianMinMaxAndStddev();
  return 0;
}
