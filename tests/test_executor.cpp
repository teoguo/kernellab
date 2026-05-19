#include "kernellab/backend/backend.hpp"
#include "kernellab/backend/registry.hpp"
#include "kernellab/backend/session.hpp"
#include "kernellab/runtime/executor.hpp"

#include "test_support/check.hpp"

namespace {

class RecordingSession final : public kernellab::BackendSession {
public:
  explicit RecordingSession(const class RecordingBackend& backend) : backend_(backend) {}

  kernellab::Status Run(kernellab::Matrix& c,
                        kernellab::IterationMeasurement& measurement) override;

private:
  const class RecordingBackend& backend_;
};

class RecordingBackend final : public kernellab::Backend {
public:
  explicit RecordingBackend(const char* backend_id) : backend_id_(backend_id) {}

  std::string_view id() const noexcept override {
    return backend_id_;
  }

  bool is_available() const noexcept override {
    return true;
  }

  kernellab::Status Prepare(const kernellab::Matrix& a, const kernellab::Matrix& b,
                            std::unique_ptr<kernellab::BackendSession>& session) const override {
    observed_a_sum = 0.0;
    observed_b_sum = 0.0;
    ++prepare_count;

    for (std::int64_t row = 0; row < a.rows(); ++row) {
      for (std::int64_t col = 0; col < a.cols(); ++col) {
        observed_a_sum += a(row, col);
      }
    }

    for (std::int64_t row = 0; row < b.rows(); ++row) {
      for (std::int64_t col = 0; col < b.cols(); ++col) {
        observed_b_sum += b(row, col);
      }
    }

    session = std::make_unique<RecordingSession>(*this);
    return kernellab::Status::Ok();
  }

  mutable double observed_a_sum = 0.0;
  mutable double observed_b_sum = 0.0;
  mutable int prepare_count = 0;
  mutable int run_count = 0;

private:
  friend class RecordingSession;
  const char* backend_id_;
};

kernellab::Status RecordingSession::Run(kernellab::Matrix& c,
                                        kernellab::IterationMeasurement& measurement) {
  ++backend_.run_count;

  measurement.kernel_ms = 1.25;
  measurement.e2e_ms = 2.5;

  for (std::int64_t row = 0; row < c.rows(); ++row) {
    for (std::int64_t col = 0; col < c.cols(); ++col) {
      c(row, col) = static_cast<float>((row + 1) * (col + 1));
    }
  }

  return kernellab::Status::Ok();
}

class OffsetSession final : public kernellab::BackendSession {
public:
  OffsetSession(const kernellab::Matrix& a, const kernellab::Matrix& b) : a_(a), b_(b) {}

  kernellab::Status Run(kernellab::Matrix& c,
                        kernellab::IterationMeasurement& measurement) override {
    measurement.kernel_ms = 0.5;
    measurement.e2e_ms = 0.75;

    for (std::int64_t row = 0; row < c.rows(); ++row) {
      for (std::int64_t col = 0; col < c.cols(); ++col) {
        float sum = 0.0F;
        for (std::int64_t inner = 0; inner < a_.cols(); ++inner) {
          sum += a_(row, inner) * b_(inner, col);
        }
        c(row, col) = sum + 0.25F;
      }
    }

    return kernellab::Status::Ok();
  }

private:
  const kernellab::Matrix& a_;
  const kernellab::Matrix& b_;
};

class OffsetBackend final : public kernellab::Backend {
public:
  std::string_view id() const noexcept override {
    return "offset";
  }

  bool is_available() const noexcept override {
    return true;
  }

  kernellab::Status Prepare(const kernellab::Matrix& a, const kernellab::Matrix& b,
                            std::unique_ptr<kernellab::BackendSession>& session) const override {
    if (a.cols() != b.rows()) {
      return kernellab::Status::Error("shape mismatch");
    }

    session = std::make_unique<OffsetSession>(a, b);
    return kernellab::Status::Ok();
  }
};

void TestCompareReusesSharedInputsAcrossBackends() {
  RecordingBackend reference("cpu_ref");
  RecordingBackend candidate("recording");

  const auto report = kernellab::CompareBackends(
      reference, {&reference, &candidate}, kernellab::ProblemSpec{4, 4, 4, 11},
      kernellab::RunOptions{1, 2}, kernellab::VerifyOptions{1e-5, 1e-5});

  KERNELLAB_CHECK(report.results.size() == 2);
  KERNELLAB_CHECK(reference.observed_a_sum == candidate.observed_a_sum);
  KERNELLAB_CHECK(reference.observed_b_sum == candidate.observed_b_sum);
}

void TestRunProducesIterationStatistics() {
  RecordingBackend backend("recording");

  const auto result = kernellab::RunBackend(backend, kernellab::ProblemSpec{8, 8, 8, 3},
                                            kernellab::RunOptions{2, 3});

  KERNELLAB_CHECK(result.success);
  KERNELLAB_CHECK(result.iterations.size() == 3);
  KERNELLAB_CHECK(result.timings_ms.size() == 3);
  KERNELLAB_CHECK(result.statistics.count == 3);
  KERNELLAB_CHECK(result.kernel_statistics.count == 3);
  KERNELLAB_CHECK(result.statistics.mean == 2.5);
  KERNELLAB_CHECK(result.kernel_statistics.mean == 1.25);
}

void TestExecutorCallsPrepareOnceAndSessionRunNTimes() {
  RecordingBackend backend("recording");

  const auto result = kernellab::RunBackend(backend, kernellab::ProblemSpec{8, 8, 8, 3},
                                            kernellab::RunOptions{2, 3});

  KERNELLAB_CHECK(result.success);
  KERNELLAB_CHECK(backend.prepare_count == 1);
  KERNELLAB_CHECK(backend.run_count == 5);
}

void TestCompareRunsOracleOnceOutsideCandidateTiming() {
  RecordingBackend reference("cpu_ref");
  RecordingBackend candidate("recording");

  const auto report = kernellab::CompareBackends(
      reference, {&reference, &candidate}, kernellab::ProblemSpec{8, 8, 8, 3},
      kernellab::RunOptions{2, 3}, kernellab::VerifyOptions{1e-5, 1e-5});

  KERNELLAB_CHECK(report.results.size() == 2);
  KERNELLAB_CHECK(reference.prepare_count == 2);
  KERNELLAB_CHECK(reference.run_count == 6);
  KERNELLAB_CHECK(candidate.prepare_count == 1);
  KERNELLAB_CHECK(candidate.run_count == 5);
}

void TestReferenceRowIsMarkedAsReferenceNotVerifiedTrue() {
  RecordingBackend reference("cpu_ref");
  RecordingBackend candidate("recording");

  const auto report = kernellab::CompareBackends(
      reference, {&reference, &candidate}, kernellab::ProblemSpec{4, 4, 4, 11},
      kernellab::RunOptions{0, 1}, kernellab::VerifyOptions{1e-5, 1e-5});

  KERNELLAB_CHECK(report.results.size() == 2);
  KERNELLAB_CHECK(report.results[0].is_reference);
  KERNELLAB_CHECK(!report.results[0].verification.has_value());
  KERNELLAB_CHECK(!report.results[1].is_reference);
}

void TestVerifyBackendCarriesMismatchDetails() {
  auto registry = kernellab::CreateDefaultRegistry();
  const kernellab::Backend* cpu_ref = registry.Find("cpu_ref");

  KERNELLAB_CHECK(cpu_ref != nullptr);

  OffsetBackend candidate;
  const auto result =
      kernellab::VerifyBackend(*cpu_ref, candidate, kernellab::ProblemSpec{4, 4, 4, 5},
                               kernellab::RunOptions{0, 1}, kernellab::VerifyOptions{1e-6, 1e-6});

  KERNELLAB_CHECK(result.success);
  KERNELLAB_CHECK(result.verification.has_value());
  KERNELLAB_CHECK(!result.verification->passed);
  KERNELLAB_CHECK(result.verification->first_mismatch.has_value());
  KERNELLAB_CHECK(result.verification->first_mismatch->absolute_error > 0.0);
  KERNELLAB_CHECK(result.error_message.find("verification failed") != std::string::npos);
  KERNELLAB_CHECK(result.error_message.find("row=") != std::string::npos);
}

void TestCompareBackendsCarriesMismatchErrorMessage() {
  auto registry = kernellab::CreateDefaultRegistry();
  const kernellab::Backend* cpu_ref = registry.Find("cpu_ref");

  KERNELLAB_CHECK(cpu_ref != nullptr);

  OffsetBackend candidate;
  const auto report = kernellab::CompareBackends(
      *cpu_ref, {cpu_ref, &candidate}, kernellab::ProblemSpec{4, 4, 4, 5},
      kernellab::RunOptions{0, 1}, kernellab::VerifyOptions{1e-6, 1e-6});

  KERNELLAB_CHECK(report.results.size() == 2);
  KERNELLAB_CHECK(report.results[1].verification.has_value());
  KERNELLAB_CHECK(!report.results[1].verification->passed);
  KERNELLAB_CHECK(report.results[1].error_message.find("verification failed") != std::string::npos);
  KERNELLAB_CHECK(report.results[1].error_message.find("row=") != std::string::npos);
}

} // namespace

int main() {
  TestCompareReusesSharedInputsAcrossBackends();
  TestRunProducesIterationStatistics();
  TestExecutorCallsPrepareOnceAndSessionRunNTimes();
  TestCompareRunsOracleOnceOutsideCandidateTiming();
  TestReferenceRowIsMarkedAsReferenceNotVerifiedTrue();
  TestVerifyBackendCarriesMismatchDetails();
  TestCompareBackendsCarriesMismatchErrorMessage();
  return 0;
}
