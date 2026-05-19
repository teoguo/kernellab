#pragma once

#include <cstdint>

namespace kernellab {

struct ProblemSpec {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::uint64_t seed = 0;
};

bool IsValidProblemSpec(const ProblemSpec& spec);

}  // namespace kernellab
