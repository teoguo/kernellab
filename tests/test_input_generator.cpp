#include "kernellab/core/problem.hpp"
#include "kernellab/runtime/input_generator.hpp"

#include "test_support/check.hpp"

namespace {

void TestSameSeedProducesSameMatrices() {
  const kernellab::ProblemSpec spec{4, 3, 2, 42};

  const auto first = kernellab::GenerateInputs(spec);
  const auto second = kernellab::GenerateInputs(spec);

  KERNELLAB_CHECK(first.a.rows() == second.a.rows());
  KERNELLAB_CHECK(first.a.cols() == second.a.cols());
  KERNELLAB_CHECK(first.b.rows() == second.b.rows());
  KERNELLAB_CHECK(first.b.cols() == second.b.cols());

  for (std::int64_t row = 0; row < first.a.rows(); ++row) {
    for (std::int64_t col = 0; col < first.a.cols(); ++col) {
      KERNELLAB_CHECK(first.a(row, col) == second.a(row, col));
    }
  }

  for (std::int64_t row = 0; row < first.b.rows(); ++row) {
    for (std::int64_t col = 0; col < first.b.cols(); ++col) {
      KERNELLAB_CHECK(first.b(row, col) == second.b(row, col));
    }
  }
}

void TestDifferentSeedsProduceDifferentMatrices() {
  const kernellab::ProblemSpec first_spec{4, 3, 2, 7};
  const kernellab::ProblemSpec second_spec{4, 3, 2, 8};

  const auto first = kernellab::GenerateInputs(first_spec);
  const auto second = kernellab::GenerateInputs(second_spec);

  bool saw_difference = false;
  for (std::int64_t row = 0; row < first.a.rows() && !saw_difference; ++row) {
    for (std::int64_t col = 0; col < first.a.cols(); ++col) {
      if (first.a(row, col) != second.a(row, col)) {
        saw_difference = true;
        break;
      }
    }
  }

  if (!saw_difference) {
    for (std::int64_t row = 0; row < first.b.rows() && !saw_difference; ++row) {
      for (std::int64_t col = 0; col < first.b.cols(); ++col) {
        if (first.b(row, col) != second.b(row, col)) {
          saw_difference = true;
          break;
        }
      }
    }
  }

  KERNELLAB_CHECK(saw_difference);
}

}  // namespace

int main() {
  TestSameSeedProducesSameMatrices();
  TestDifferentSeedsProduceDifferentMatrices();
  return 0;
}
