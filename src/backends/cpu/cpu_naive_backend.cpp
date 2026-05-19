#include "kernellab/backend/backend.hpp"

#include <chrono>
#include <memory>

namespace kernellab {

namespace {

class CpuNaiveSession final : public BackendSession {
 public:
  CpuNaiveSession(const Matrix& a, const Matrix& b) : a_(a), b_(b) {}

  Status Run(Matrix& c, IterationMeasurement& measurement) override {
    if (a_.cols() != b_.rows()) {
      return Status::Error("cpu_naive requires a.cols() == b.rows()");
    }
    if (c.rows() != a_.rows() || c.cols() != b_.cols()) {
      return Status::Error("cpu_naive output matrix has wrong shape");
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::int64_t row = 0; row < c.rows(); ++row) {
      for (std::int64_t col = 0; col < c.cols(); ++col) {
        float sum = 0.0F;
        for (std::int64_t inner = 0; inner < a_.cols(); ++inner) {
          sum += a_(row, inner) * b_(inner, col);
        }
        c(row, col) = sum;
      }
    }
    const auto finish = std::chrono::steady_clock::now();
    measurement.e2e_ms = std::chrono::duration<double, std::milli>(finish - start).count();
    measurement.kernel_ms = measurement.e2e_ms;
    return Status::Ok();
  }

 private:
  const Matrix& a_;
  const Matrix& b_;
};

class CpuNaiveBackend final : public Backend {
 public:
  std::string_view id() const noexcept override { return "cpu_naive"; }

  bool is_available() const noexcept override { return true; }

  Status Prepare(const Matrix& a,
                 const Matrix& b,
                 BackendSessionPtr& session) const override {
    if (a.cols() != b.rows()) {
      return Status::Error("cpu_naive requires a.cols() == b.rows()");
    }
    session = std::make_unique<CpuNaiveSession>(a, b);
    return Status::Ok();
  }
};

}  // namespace

BackendPtr MakeCpuNaiveBackend() {
  return std::make_unique<CpuNaiveBackend>();
}

}  // namespace kernellab
