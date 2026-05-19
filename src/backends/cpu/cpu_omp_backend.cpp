#include "kernellab/backend/backend.hpp"

#include <chrono>
#include <memory>

#if defined(KERNELLAB_HAS_OPENMP) && KERNELLAB_HAS_OPENMP
#include <omp.h>
#endif

namespace kernellab {

namespace {

class CpuOmpSession final : public BackendSession {
 public:
  CpuOmpSession(const Matrix& a, const Matrix& b) : a_(a), b_(b) {}

  Status Run(Matrix& c, IterationMeasurement& measurement) override {
    if (c.rows() != a_.rows() || c.cols() != b_.cols()) {
      return Status::Error("cpu_omp output matrix has wrong shape");
    }

    const auto start = std::chrono::steady_clock::now();
#if defined(KERNELLAB_HAS_OPENMP) && KERNELLAB_HAS_OPENMP
#pragma omp parallel for collapse(2)
    for (std::int64_t row = 0; row < c.rows(); ++row) {
      for (std::int64_t col = 0; col < c.cols(); ++col) {
        float sum = 0.0F;
        for (std::int64_t inner = 0; inner < a_.cols(); ++inner) {
          sum += a_(row, inner) * b_(inner, col);
        }
        c(row, col) = sum;
      }
    }
#endif
    const auto finish = std::chrono::steady_clock::now();
    measurement.e2e_ms = std::chrono::duration<double, std::milli>(finish - start).count();
    measurement.kernel_ms = measurement.e2e_ms;
    return Status::Ok();
  }

 private:
  const Matrix& a_;
  const Matrix& b_;
};

class CpuOmpBackend final : public Backend {
 public:
  std::string_view id() const noexcept override { return "cpu_omp"; }

  bool is_available() const noexcept override {
#if defined(KERNELLAB_HAS_OPENMP) && KERNELLAB_HAS_OPENMP
    return true;
#else
    return false;
#endif
  }

  std::string unavailable_reason() const override {
    return "OpenMP disabled at build time";
  }

  Status Prepare(const Matrix& a,
                 const Matrix& b,
                 BackendSessionPtr& session) const override {
    if (!is_available()) {
      return Status::Error("cpu_omp backend is unavailable because OpenMP is disabled");
    }
    if (a.cols() != b.rows()) {
      return Status::Error("cpu_omp requires a.cols() == b.rows()");
    }
    session = std::make_unique<CpuOmpSession>(a, b);
    return Status::Ok();
  }
};

}  // namespace

BackendPtr MakeCpuOmpBackend() {
  return std::make_unique<CpuOmpBackend>();
}

}  // namespace kernellab
