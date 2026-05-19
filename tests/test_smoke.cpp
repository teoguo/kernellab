#include "kernellab/core/matrix.hpp"

#include "test_support/check.hpp"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>

namespace {

void TestMatrixRejectsNegativeDimensionsBeforeAllocation() {
  bool threw = false;
  try {
    static_cast<void>(kernellab::Matrix(-1, 4));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  KERNELLAB_CHECK(threw);
}

void TestMatrixRejectsOverflowingElementCount() {
  bool threw = false;
  try {
    static_cast<void>(kernellab::Matrix(
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max()));
  } catch (const std::overflow_error&) {
    threw = true;
  } catch (const std::length_error&) {
    threw = true;
  } catch (const std::bad_alloc&) {
    threw = true;
  }
  KERNELLAB_CHECK(threw);
}

}  // namespace

int main() {
  TestMatrixRejectsNegativeDimensionsBeforeAllocation();
  TestMatrixRejectsOverflowingElementCount();
  return EXIT_SUCCESS;
}
