#pragma once

#include <cstddef>
#include <vector>

namespace kernellab {

struct SummaryStatistics {
  std::size_t count = 0;
  double mean = 0.0;
  double median = 0.0;
  double min = 0.0;
  double max = 0.0;
  double stddev = 0.0;
};

SummaryStatistics ComputeSummaryStatistics(const std::vector<double>& samples);

}  // namespace kernellab
