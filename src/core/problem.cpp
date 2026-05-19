#include "kernellab/core/problem.hpp"

namespace kernellab {

bool IsValidProblemSpec(const ProblemSpec& spec) {
  return spec.m > 0 && spec.n > 0 && spec.k > 0;
}

}  // namespace kernellab
