#include "kernellab/benchmark/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace kernellab {

SummaryStatistics ComputeSummaryStatistics(const std::vector<double>& samples) {
  if (samples.empty()) {
    throw std::invalid_argument("cannot compute statistics from empty samples");
  }

  SummaryStatistics stats;
  stats.count = samples.size();

  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  stats.mean = sum / static_cast<double>(stats.count);

  auto ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  stats.min = ordered.front();
  stats.max = ordered.back();

  const std::size_t midpoint = ordered.size() / 2;
  if ((ordered.size() % 2U) == 0U) {
    stats.median = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0;
  } else {
    stats.median = ordered[midpoint];
  }

  double variance_sum = 0.0;
  for (const double sample : samples) {
    const double delta = sample - stats.mean;
    variance_sum += delta * delta;
  }
  stats.stddev = std::sqrt(variance_sum / static_cast<double>(stats.count));

  return stats;
}

}  // namespace kernellab
