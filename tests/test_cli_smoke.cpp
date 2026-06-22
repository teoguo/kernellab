#include "kernellab/app/cli.hpp"
#include "kernellab/backend/registry.hpp"

#include "test_support/check.hpp"

#include <string>
#include <vector>

namespace {

int RunCommand(const std::vector<std::string>& args, std::string& stdout_text,
               std::string& stderr_text) {
  return kernellab::RunCli(args, stdout_text, stderr_text);
}

std::string FindUnavailableBackendId() {
  const auto registry = kernellab::CreateDefaultRegistry();
  for (const kernellab::Backend* backend : registry.All()) {
    if (backend != nullptr && !backend->is_available()) {
      return std::string(backend->id());
    }
  }
  return {};
}

void TestRunCommandSucceedsForCpuRef() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"run", "--backend", "cpu_ref", "--m", "8", "--n", "8", "--k",
                                    "8", "--iterations", "2", "--warmup", "1"},
                                   stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("backend=cpu_ref") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("available=true") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("success=true") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("kernel_mean_ms=") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestRunCommandAcceptsWhitespaceWrappedBackendId() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand(
      {"run", "--backend", " cpu_ref ", "--m", "8", "--n", "8", "--k", "8", "--iterations", "1"},
      stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("backend=cpu_ref") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestVerifyCommandFailsForUnknownBackend() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code =
      RunCommand({"verify", "--backend", "does_not_exist", "--m", "4", "--n", "4", "--k", "4"},
                 stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("unknown backend") != std::string::npos);
}

void TestVerifyCommandPrintsReadableBooleanSummary() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code =
      RunCommand({"verify", "--backend", "cpu_ref", "--m", "4", "--n", "4", "--k", "4"},
                 stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("available=true") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("success=true") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("kernel_mean_ms=") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("verification_passed=true") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestCompareCommandPrintsBackendSummary() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand(
      {"compare", "--backends", "cpu_ref", "--m", "4", "--n", "4", "--k", "4", "--iterations", "2"},
      stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("backend_id,available,success,verification_status,mean_ms,"
                                   "kernel_mean_ms,gflops,pct_of_cublas,error_message") !=
                  std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("cpu_ref") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("cpu_ref,true,true,reference,") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestHelpCommandPrintsUsage() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"--help"}, stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("Usage: kernellab") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("run") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("verify") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("compare") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestVersionCommandPrintsVersion() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"--version"}, stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("kernellab ") == 0);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestListBackendsPrintsStableAvailabilityLines() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"--list-backends"}, stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("cpu_ref\ttrue\tavailable") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("cpu_naive\ttrue\tavailable") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("cuda_reg\t") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestRunHelpCommandPrintsSubcommandUsage() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"run", "--help"}, stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("Usage: kernellab run") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("--backend <id>") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestVerifyHelpCommandPrintsSubcommandUsage() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"verify", "--help"}, stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("Usage: kernellab verify") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("--atol <value> --rtol <value>") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("--iterations") == std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("--warmup") == std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestCompareHelpCommandPrintsSubcommandUsage() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"compare", "--help"}, stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("Usage: kernellab compare") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("--backends <id1,id2,...>") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestRunCommandReportsUnavailableBackendClearly() {
  const std::string backend_id = FindUnavailableBackendId();
  if (backend_id.empty()) {
    return;
  }

  std::string stdout_text;
  std::string stderr_text;

  const int exit_code =
      RunCommand({"run", "--backend", backend_id, "--m", "8", "--n", "8", "--k", "8"}, stdout_text,
                 stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("unavailable") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

void TestRunCommandRejectsInvalidDimensionsCleanly() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code =
      RunCommand({"run", "--backend", "cpu_ref", "--m", "0", "--n", "8", "--k", "8"}, stdout_text,
                 stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("positive") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

void TestRunCommandRejectsMalformedIntegerFlagCleanly() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code =
      RunCommand({"run", "--backend", "cpu_ref", "--m", "abc", "--n", "8", "--k", "8"}, stdout_text,
                 stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("invalid numeric value") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

void TestRunCommandRejectsIterationCountOutsideInt32Range() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"run", "--backend", "cpu_ref", "--m", "1", "--n", "1", "--k",
                                    "1", "--iterations", "2147483648"},
                                   stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("iterations value is out of range") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

void TestRunCommandRejectsUnknownFlagCleanly() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand(
      {"run", "--backend", "cpu_ref", "--m", "8", "--n", "8", "--k", "8", "--iteratons", "2"},
      stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("unknown flag --iteratons") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

void TestUnknownFlagErrorsAreDeterministic() {
  std::string stdout_a;
  std::string stderr_a;
  std::string stdout_b;
  std::string stderr_b;

  const int exit_code_a = RunCommand({"run", "--backend", "cpu_ref", "--m", "8", "--n", "8", "--k",
                                      "8", "--zzz", "1", "--aaa", "2"},
                                     stdout_a, stderr_a);
  const int exit_code_b = RunCommand({"run", "--backend", "cpu_ref", "--m", "8", "--n", "8", "--k",
                                      "8", "--aaa", "2", "--zzz", "1"},
                                     stdout_b, stderr_b);

  KERNELLAB_CHECK(exit_code_a != 0);
  KERNELLAB_CHECK(exit_code_b != 0);
  KERNELLAB_CHECK(stderr_a == "unknown flag --aaa");
  KERNELLAB_CHECK(stderr_b == "unknown flag --aaa");
}

void TestRunCommandRejectsDuplicateFlagCleanly() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code =
      RunCommand({"run", "--backend", "cpu_ref", "--m", "8", "--m", "16", "--n", "8", "--k", "8"},
                 stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("duplicate flag --m") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

void TestCompareCommandReturnsFailureWhenBackendIsUnavailable() {
  const std::string backend_id = FindUnavailableBackendId();
  if (backend_id.empty()) {
    return;
  }

  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"compare", "--backends", "cpu_ref," + backend_id, "--m", "4",
                                    "--n", "4", "--k", "4", "--iterations", "2"},
                                   stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stdout_text.find("cpu_ref") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find(backend_id) != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("error_message") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestCompareCommandAcceptsWhitespaceSeparatedBackendList() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"compare", "--backends", "cpu_ref, cpu_ref", "--m", "4", "--n",
                                    "4", "--k", "4", "--iterations", "1"},
                                   stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("cpu_ref") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestCompareCommandRejectsEmptyBackendEntryCleanly() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"compare", "--backends", "cpu_ref,,cpu_ref", "--m", "4", "--n",
                                    "4", "--k", "4", "--iterations", "1"},
                                   stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("empty backend id") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

void TestCompareCommandRejectsEmptyBackendListCleanly() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand(
      {"compare", "--backends", "", "--m", "4", "--n", "4", "--k", "4", "--iterations", "1"},
      stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("at least one backend") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

void TestCompareCommandRejectsUnknownFlagCleanly() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand(
      {"compare", "--backends", "cpu_ref", "--m", "4", "--n", "4", "--k", "4", "--bogus", "value"},
      stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("unknown flag --bogus") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

void TestCompareCommandRejectsDuplicateFlagCleanly() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand({"compare", "--backends", "cpu_ref", "--m", "4", "--n", "4",
                                    "--k", "4", "--seed", "1", "--seed", "2"},
                                   stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("duplicate flag --seed") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

void TestRunDefaultsUseMultipleIterationsAndWarmups() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code =
      RunCommand({"run", "--backend", "cpu_ref", "--m", "16", "--n", "16", "--k", "16"},
                 stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code == 0);
  KERNELLAB_CHECK(stdout_text.find("mean_ms=") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.find("kernel_mean_ms=") != std::string::npos);
  KERNELLAB_CHECK(stderr_text.empty());
}

void TestVerifyRejectsBenchmarkOnlyFlags() {
  std::string stdout_text;
  std::string stderr_text;

  const int exit_code = RunCommand(
      {"verify", "--backend", "cpu_ref", "--m", "4", "--n", "4", "--k", "4", "--iterations", "2"},
      stdout_text, stderr_text);

  KERNELLAB_CHECK(exit_code != 0);
  KERNELLAB_CHECK(stderr_text.find("unknown flag --iterations") != std::string::npos);
  KERNELLAB_CHECK(stdout_text.empty());
}

} // namespace

int main() {
  TestRunCommandSucceedsForCpuRef();
  TestRunCommandAcceptsWhitespaceWrappedBackendId();
  TestVerifyCommandFailsForUnknownBackend();
  TestVerifyCommandPrintsReadableBooleanSummary();
  TestCompareCommandPrintsBackendSummary();
  TestHelpCommandPrintsUsage();
  TestVersionCommandPrintsVersion();
  TestListBackendsPrintsStableAvailabilityLines();
  TestRunHelpCommandPrintsSubcommandUsage();
  TestVerifyHelpCommandPrintsSubcommandUsage();
  TestCompareHelpCommandPrintsSubcommandUsage();
  TestRunCommandReportsUnavailableBackendClearly();
  TestRunCommandRejectsInvalidDimensionsCleanly();
  TestRunCommandRejectsMalformedIntegerFlagCleanly();
  TestRunCommandRejectsIterationCountOutsideInt32Range();
  TestRunCommandRejectsUnknownFlagCleanly();
  TestUnknownFlagErrorsAreDeterministic();
  TestRunCommandRejectsDuplicateFlagCleanly();
  TestCompareCommandReturnsFailureWhenBackendIsUnavailable();
  TestCompareCommandAcceptsWhitespaceSeparatedBackendList();
  TestCompareCommandRejectsEmptyBackendEntryCleanly();
  TestCompareCommandRejectsEmptyBackendListCleanly();
  TestCompareCommandRejectsUnknownFlagCleanly();
  TestCompareCommandRejectsDuplicateFlagCleanly();
  TestRunDefaultsUseMultipleIterationsAndWarmups();
  TestVerifyRejectsBenchmarkOnlyFlags();
  return 0;
}
