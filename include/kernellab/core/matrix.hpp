#pragma once

#include <cstdint>
#include <vector>

namespace kernellab {

class Matrix {
 public:
  Matrix() = default;
  Matrix(std::int64_t rows, std::int64_t cols);

  std::int64_t rows() const noexcept;
  std::int64_t cols() const noexcept;
  std::int64_t size() const noexcept;

  float* data() noexcept;
  const float* data() const noexcept;

  float& operator()(std::int64_t row, std::int64_t col) noexcept;
  const float& operator()(std::int64_t row, std::int64_t col) const noexcept;

 private:
  std::int64_t rows_ = 0;
  std::int64_t cols_ = 0;
  std::vector<float> values_;
};

}  // namespace kernellab
