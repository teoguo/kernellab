#include "kernellab/backend/registry.hpp"
#include "kernellab/backend/session.hpp"
#include "kernellab/core/matrix.hpp"
#include "kernellab/runtime/input_generator.hpp"
#include "kernellab/verify/verification.hpp"

#include "test_support/check.hpp"

namespace {

kernellab::Status RunPreparedOnce(const kernellab::Backend& backend,
                                  const kernellab::Matrix& a,
                                  const kernellab::Matrix& b,
                                  kernellab::Matrix& c) {
  std::unique_ptr<kernellab::BackendSession> session;
  const auto prepare_status = backend.Prepare(a, b, session);
  if (!prepare_status.ok) {
    return prepare_status;
  }

  KERNELLAB_CHECK(session != nullptr);
  kernellab::IterationMeasurement measurement;
  return session->Run(c, measurement);
}

void TestCpuRefComputesSmallKnownMatmul() {
  auto registry = kernellab::CreateDefaultRegistry();
  const kernellab::Backend* backend = registry.Find("cpu_ref");

  KERNELLAB_CHECK(backend != nullptr);
  KERNELLAB_CHECK(backend->is_available());

  kernellab::Matrix a(2, 3);
  kernellab::Matrix b(3, 2);
  kernellab::Matrix c(2, 2);

  a(0, 0) = 1.0F;
  a(0, 1) = 2.0F;
  a(0, 2) = 3.0F;
  a(1, 0) = 4.0F;
  a(1, 1) = 5.0F;
  a(1, 2) = 6.0F;

  b(0, 0) = 7.0F;
  b(0, 1) = 8.0F;
  b(1, 0) = 9.0F;
  b(1, 1) = 10.0F;
  b(2, 0) = 11.0F;
  b(2, 1) = 12.0F;

  const auto status = RunPreparedOnce(*backend, a, b, c);

  KERNELLAB_CHECK(status.ok);
  KERNELLAB_CHECK(c(0, 0) == 58.0F);
  KERNELLAB_CHECK(c(0, 1) == 64.0F);
  KERNELLAB_CHECK(c(1, 0) == 139.0F);
  KERNELLAB_CHECK(c(1, 1) == 154.0F);
}

void TestCpuOmpMatchesCpuRefOnSmallProblem() {
  auto registry = kernellab::CreateDefaultRegistry();
  const kernellab::Backend* cpu_ref = registry.Find("cpu_ref");
  const kernellab::Backend* cpu_naive = registry.Find("cpu_naive");
  const kernellab::Backend* cpu_omp = registry.Find("cpu_omp");

  KERNELLAB_CHECK(cpu_ref != nullptr);
  KERNELLAB_CHECK(cpu_naive != nullptr);
  KERNELLAB_CHECK(cpu_omp != nullptr);
  KERNELLAB_CHECK(cpu_ref->is_available());
  KERNELLAB_CHECK(cpu_naive->is_available());

  if (!cpu_omp->is_available()) {
    return;
  }

  const kernellab::ProblemSpec problem{4, 4, 4, 9};
  const auto inputs = kernellab::GenerateInputs(problem);
  kernellab::Matrix ref_out(4, 4);
  kernellab::Matrix omp_out(4, 4);

  KERNELLAB_CHECK(RunPreparedOnce(*cpu_ref, inputs.a, inputs.b, ref_out).ok);
  KERNELLAB_CHECK(RunPreparedOnce(*cpu_omp, inputs.a, inputs.b, omp_out).ok);

  const auto verification = kernellab::VerifyMatrices(
      ref_out, omp_out, kernellab::DefaultTolerance(problem));
  KERNELLAB_CHECK(verification.passed);
}

void TestCpuRefUsesHigherPrecisionOraclePath() {
  auto registry = kernellab::CreateDefaultRegistry();
  const kernellab::Backend* cpu_ref = registry.Find("cpu_ref");
  const kernellab::Backend* cpu_naive = registry.Find("cpu_naive");

  KERNELLAB_CHECK(cpu_ref != nullptr);
  KERNELLAB_CHECK(cpu_naive != nullptr);

  kernellab::Matrix a(1, 3);
  kernellab::Matrix b(3, 1);
  kernellab::Matrix ref_out(1, 1);
  kernellab::Matrix naive_out(1, 1);

  a(0, 0) = 100000000.0F;
  a(0, 1) = 1.0F;
  a(0, 2) = -100000000.0F;
  b(0, 0) = 1.0F;
  b(1, 0) = 1.0F;
  b(2, 0) = 1.0F;

  KERNELLAB_CHECK(RunPreparedOnce(*cpu_ref, a, b, ref_out).ok);
  KERNELLAB_CHECK(RunPreparedOnce(*cpu_naive, a, b, naive_out).ok);

  KERNELLAB_CHECK(ref_out(0, 0) == 1.0F);
  KERNELLAB_CHECK(naive_out(0, 0) == 0.0F);
}

}  // namespace

int main() {
  TestCpuRefComputesSmallKnownMatmul();
  TestCpuOmpMatchesCpuRefOnSmallProblem();
  TestCpuRefUsesHigherPrecisionOraclePath();
  return 0;
}
