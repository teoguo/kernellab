#include "kernellab/backend/backend.hpp"

#include <chrono>
#include <memory>

namespace kernellab {

namespace {

class CpuRefSession final : public BackendSession {
 public:
  CpuRefSession(const Matrix& a, const Matrix& b) : a_(a), b_(b) {}

  Status Run(Matrix& c, IterationMeasurement& measurement) override {
    if (a_.cols() != b_.rows()) {
      return Status::Error("cpu_ref requires a.cols() == b.rows()");
    }
    if (c.rows() != a_.rows() || c.cols() != b_.cols()) {
      return Status::Error("cpu_ref output matrix has wrong shape");
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::int64_t row = 0; row < c.rows(); ++row) {
      for (std::int64_t col = 0; col < c.cols(); ++col) {
        double sum = 0.0;
        for (std::int64_t inner = 0; inner < a_.cols(); ++inner) {
          sum += static_cast<double>(a_(row, inner)) * static_cast<double>(b_(inner, col));
        }
        c(row, col) = static_cast<float>(sum);
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

class CpuRefBackend final : public Backend {
 public:
  std::string_view id() const noexcept override { return "cpu_ref"; }

  bool is_available() const noexcept override { return true; }

  Status Prepare(const Matrix& a,
                 const Matrix& b,
                 BackendSessionPtr& session) const override {
    if (a.cols() != b.rows()) {
      return Status::Error("cpu_ref requires a.cols() == b.rows()");
    }
    session = std::make_unique<CpuRefSession>(a, b);
    return Status::Ok();
  }
};

}  // namespace

BackendPtr MakeCpuRefBackend() {
  return std::make_unique<CpuRefBackend>();
}

}  // namespace kernellab
