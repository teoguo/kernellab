#pragma once

#include "kernellab/core/matrix.hpp"
#include "kernellab/core/problem.hpp"

namespace kernellab {

struct GeneratedInputs {
  Matrix a;
  Matrix b;
};

GeneratedInputs GenerateInputs(const ProblemSpec& spec);

}  // namespace kernellab
