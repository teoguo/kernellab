#include "kernellab/runtime/input_generator.hpp"

#include <random>
#include <stdexcept>

namespace kernellab {

namespace {

float NextDeterministicValue(std::mt19937& generator) {
  const std::uint32_t bucket = generator() & 0x00FFFFFFU;
  const double unit = static_cast<double>(bucket) / static_cast<double>(0x01000000U);
  return static_cast<float>((unit * 2.0) - 1.0);
}

void FillMatrix(Matrix& matrix, std::mt19937& generator) {
  for (std::int64_t row = 0; row < matrix.rows(); ++row) {
    for (std::int64_t col = 0; col < matrix.cols(); ++col) {
      matrix(row, col) = NextDeterministicValue(generator);
    }
  }
}

}  // namespace

GeneratedInputs GenerateInputs(const ProblemSpec& spec) {
  if (!IsValidProblemSpec(spec)) {
    throw std::invalid_argument("problem dimensions must be positive");
  }

  std::mt19937 generator(static_cast<std::mt19937::result_type>(spec.seed));

  GeneratedInputs inputs{
      Matrix(spec.m, spec.k),
      Matrix(spec.k, spec.n),
  };

  FillMatrix(inputs.a, generator);
  FillMatrix(inputs.b, generator);
  return inputs;
}

}  // namespace kernellab
