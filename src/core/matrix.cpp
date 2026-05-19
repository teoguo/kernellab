#include "kernellab/core/matrix.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace kernellab {

namespace {

std::int64_t FlatIndex(const std::int64_t cols,
                       const std::int64_t row,
                       const std::int64_t col) {
  return row * cols + col;
}

std::size_t CheckedElementCount(const std::int64_t rows, const std::int64_t cols) {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("matrix dimensions must be non-negative");
  }

  std::size_t count = 0;
  if (__builtin_mul_overflow(static_cast<std::size_t>(rows),
                             static_cast<std::size_t>(cols),
                             &count)) {
    throw std::overflow_error("matrix element count overflow");
  }

  if (count > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
    throw std::overflow_error("matrix storage size overflow");
  }

  return count;
}

}  // namespace

Matrix::Matrix(const std::int64_t rows, const std::int64_t cols)
    : rows_(rows), cols_(cols) {
  values_.assign(CheckedElementCount(rows, cols), 0.0F);
}

std::int64_t Matrix::rows() const noexcept { return rows_; }

std::int64_t Matrix::cols() const noexcept { return cols_; }

std::int64_t Matrix::size() const noexcept {
  return static_cast<std::int64_t>(values_.size());
}

float* Matrix::data() noexcept { return values_.data(); }

const float* Matrix::data() const noexcept { return values_.data(); }

float& Matrix::operator()(const std::int64_t row, const std::int64_t col) noexcept {
  return values_[static_cast<std::size_t>(FlatIndex(cols_, row, col))];
}

const float& Matrix::operator()(const std::int64_t row, const std::int64_t col) const noexcept {
  return values_[static_cast<std::size_t>(FlatIndex(cols_, row, col))];
}

}  // namespace kernellab
