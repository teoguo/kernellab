#include "kernellab/io/export.hpp"
#include "kernellab/runtime/executor.hpp"

#include "test_support/check.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace {

double ExtractJsonNumberAfterKey(const std::string& json, const std::string& key) {
  const auto key_pos = json.find(key);
  KERNELLAB_CHECK(key_pos != std::string::npos);
  const auto colon_pos = json.find(':', key_pos + key.size());
  KERNELLAB_CHECK(colon_pos != std::string::npos);
  const auto value_start = colon_pos + 1;
  const auto value_end = json.find_first_of(",}]", value_start);
  KERNELLAB_CHECK(value_end != std::string::npos);
  return std::stod(json.substr(value_start, value_end - value_start));
}

kernellab::BackendRunResult MakeResult(std::string backend_id, bool available, bool success,
                                       std::string error_message, std::vector<double> timings_ms,
                                       kernellab::SummaryStatistics statistics,
                                       std::optional<kernellab::VerificationResult> verification,
                                       bool is_reference = false) {
  kernellab::BackendRunResult result;
  result.backend_id = std::move(backend_id);
  result.available = available;
  result.success = success;
  result.is_reference = is_reference;
  result.error_message = std::move(error_message);
  result.timings_ms = timings_ms;
  result.statistics = statistics;
  result.kernel_statistics = statistics;
  result.verification = std::move(verification);
  result.iterations.reserve(result.timings_ms.size());
  for (const double timing_ms : result.timings_ms) {
    result.iterations.push_back(kernellab::IterationMeasurement{timing_ms, timing_ms});
  }
  return result;
}

kernellab::BackendRunResult
MakeResultWithGflops(std::string backend_id, double gflops,
                     std::optional<kernellab::VerificationResult> verification = std::nullopt) {
  auto result =
      MakeResult(std::move(backend_id), true, true, "", {1.0},
                 kernellab::SummaryStatistics{1, 1.0, 1.0, 1.0, 1.0, 0.0}, std::move(verification));
  result.gflops = gflops;
  return result;
}

void FillTestEnvironment(kernellab::ComparisonReport& report) {
  report.environment.timestamp_utc = "2026-04-21T12:34:56Z";
  report.environment.cpu_model = "Test CPU";
  report.environment.gpu_name = "Test GPU";
  report.environment.openmp_threads = 8;
}

void CheckJsonParsesWithPython(const std::string& json) {
  const auto temp_dir = std::filesystem::temp_directory_path() / "kernellab_export_tests";
  std::filesystem::create_directories(temp_dir);
  const auto output_path = temp_dir / "report.json";
  {
    std::ofstream output(output_path);
    output << json;
  }

  const std::string command = "python3 -m json.tool \"" + output_path.string() + "\" > /dev/null";
  const int exit_code = std::system(command.c_str());
  std::filesystem::remove(output_path);
  KERNELLAB_CHECK(exit_code == 0);
}

void TestJsonContainsStructuredResultFields() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{16, 32, 8, 5};
  report.run_options = kernellab::RunOptions{1, 3};
  report.verify_options = kernellab::VerifyOptions{1e-5, 1e-5};
  report.results.push_back(MakeResult(
      "cpu_ref", true, true, "", {1.0, 2.0, 3.0},
      kernellab::SummaryStatistics{3, 2.0, 2.0, 1.0, 3.0, 0.816496580927726}, std::nullopt));

  const auto json = kernellab::SerializeReportJson(report);

  KERNELLAB_CHECK(json.find("\"schema_version\":3") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"environment\":{") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"timestamp_utc\":\"2026-04-21T12:34:56Z\"") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"backend_id\":\"cpu_ref\"") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"m\":16") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"verify_options\":{") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"atol\":") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"rtol\":") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"verification_status\":\"not_run\"") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"iterations\":[{\"kernel_ms\":1,\"e2e_ms\":1}") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"summary\":{\"kernel_ms\":{") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"e2e_ms\":{") != std::string::npos);
}

void TestJsonReportIsParseableWithMultipleResults() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{4, 4, 4, 9};
  report.run_options = kernellab::RunOptions{1, 2};
  report.verify_options = kernellab::VerifyOptions{1e-5, 1e-5};
  report.results.push_back(MakeResult("cpu_ref", true, true, "", {0.5, 0.6},
                                      kernellab::SummaryStatistics{2, 0.55, 0.55, 0.5, 0.6, 0.05},
                                      std::nullopt, true));
  report.results.push_back(MakeResult("cpu_naive", true, true, "", {0.3, 0.4},
                                      kernellab::SummaryStatistics{2, 0.35, 0.35, 0.3, 0.4, 0.05},
                                      kernellab::VerificationResult{}));

  CheckJsonParsesWithPython(kernellab::SerializeReportJson(report));
}

void TestJsonIncludesPctOfCublasWhenCublasIsPresent() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{4, 4, 4, 9};
  report.run_options = kernellab::RunOptions{0, 1};
  report.verify_options = kernellab::VerifyOptions{1e-5, 1e-5};
  report.results.push_back(
      MakeResultWithGflops("cuda_naive", 25.0, kernellab::VerificationResult{}));
  report.results.push_back(MakeResultWithGflops("cublas", 100.0, kernellab::VerificationResult{}));

  const auto json = kernellab::SerializeReportJson(report);

  KERNELLAB_CHECK(json.find("\"backend_id\":\"cuda_naive\"") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"pct_of_cublas\":25") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"backend_id\":\"cublas\"") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"pct_of_cublas\":100") != std::string::npos);
  CheckJsonParsesWithPython(json);
}

void TestJsonEmitsNullPctOfCublasWhenCublasIsAbsent() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{4, 4, 4, 9};
  report.run_options = kernellab::RunOptions{0, 1};
  report.results.push_back(MakeResultWithGflops("cuda_naive", 25.0));

  const auto json = kernellab::SerializeReportJson(report);

  KERNELLAB_CHECK(json.find("\"pct_of_cublas\":null") != std::string::npos);
  CheckJsonParsesWithPython(json);
}

void TestJsonSerializesDoubleWithRoundTripPrecision() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{1, 1, 1, 7};
  report.run_options = kernellab::RunOptions{0, 1};
  report.verify_options = kernellab::VerifyOptions{0.12345678912345678, 0.98765432198765432};
  report.results.push_back(MakeResult("cpu_ref", true, true, "", {0.12345678912345678},
                                      kernellab::SummaryStatistics{
                                          1,
                                          0.12345678912345678,
                                          0.12345678912345678,
                                          0.12345678912345678,
                                          0.12345678912345678,
                                          0.0,
                                      },
                                      std::nullopt));

  const auto json = kernellab::SerializeReportJson(report);

  KERNELLAB_CHECK(ExtractJsonNumberAfterKey(json, "\"atol\"") == report.verify_options.atol);
  KERNELLAB_CHECK(ExtractJsonNumberAfterKey(json, "\"rtol\"") == report.verify_options.rtol);
}

void TestCsvContainsOneRowPerBackend() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{4, 4, 4, 9};
  report.run_options = kernellab::RunOptions{0, 2};
  report.verify_options = kernellab::VerifyOptions{1e-5, 1e-5};
  report.results.push_back(MakeResult("cpu_ref", true, true, "", {0.5, 0.6},
                                      kernellab::SummaryStatistics{2, 0.55, 0.55, 0.5, 0.6, 0.05},
                                      std::nullopt));
  report.results.push_back(MakeResult("cpu_omp", true, true, "", {0.3, 0.4},
                                      kernellab::SummaryStatistics{2, 0.35, 0.35, 0.3, 0.4, 0.05},
                                      kernellab::VerificationResult{}));

  const auto csv = kernellab::SerializeReportCsv(report);

  KERNELLAB_CHECK(
      csv.find("schema_version,timestamp_utc,cpu_model,gpu_name,openmp_threads,backend_id") == 0);
  KERNELLAB_CHECK(csv.find("3,2026-04-21T12:34:56Z,Test CPU,Test GPU,8,cpu_ref,4,4,4,9") !=
                  std::string::npos);
  KERNELLAB_CHECK(csv.find("3,2026-04-21T12:34:56Z,Test CPU,Test GPU,8,cpu_omp,4,4,4,9") !=
                  std::string::npos);
}

void TestCsvIncludesPctOfCublasWhenCublasIsPresent() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{4, 4, 4, 9};
  report.run_options = kernellab::RunOptions{0, 1};
  report.results.push_back(MakeResultWithGflops("cuda_naive", 25.0));
  report.results.push_back(MakeResultWithGflops("cublas", 100.0));

  const auto csv = kernellab::SerializeReportCsv(report);

  KERNELLAB_CHECK(csv.find("verification_status,gflops,pct_of_cublas,e2e_mean_ms") !=
                  std::string::npos);
  KERNELLAB_CHECK(csv.find(",cuda_naive,4,4,4,9,0,1,true,true,not_run,25,25,") !=
                  std::string::npos);
  KERNELLAB_CHECK(csv.find(",cublas,4,4,4,9,0,1,true,true,not_run,100,100,") != std::string::npos);
}

void TestCsvEscapesErrorMessageField() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{4, 4, 4, 9};
  report.run_options = kernellab::RunOptions{0, 1};
  report.results.push_back(MakeResult("cuda_naive", false, false,
                                      "failed, reason: \"device missing\"", {},
                                      kernellab::SummaryStatistics{}, std::nullopt));

  const auto csv = kernellab::SerializeReportCsv(report);

  KERNELLAB_CHECK(csv.find("error_message") != std::string::npos);
  KERNELLAB_CHECK(csv.find("\"failed, reason: \"\"device missing\"\"\"") != std::string::npos);
}

void TestCsvIncludesVerificationMismatchDetails() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{2, 2, 2, 1};
  report.run_options = kernellab::RunOptions{0, 1};
  kernellab::VerificationResult verification;
  verification.passed = false;
  verification.first_mismatch = kernellab::MismatchInfo{
      1, 0, 3.5F, 7.0F, 3.5, 1.0,
  };
  report.results.push_back(MakeResult("cpu_omp", true, true, "verification failed", {0.1},
                                      kernellab::SummaryStatistics{1, 0.1, 0.1, 0.1, 0.1, 0.0},
                                      verification));

  const auto csv = kernellab::SerializeReportCsv(report);

  KERNELLAB_CHECK(csv.find("mismatch_row,mismatch_col,reference_value,candidate_value,absolute_"
                           "error,relative_error") != std::string::npos);
  KERNELLAB_CHECK(csv.find(",cpu_omp,2,2,2,1,0,1,true,true,failed,") != std::string::npos);
  KERNELLAB_CHECK(csv.find(",verification failed,1,0,3.5,7,3.5,1") != std::string::npos);
}

void TestJsonEscapesSpecialCharactersInStrings() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{2, 2, 2, 1};
  report.run_options = kernellab::RunOptions{0, 1};
  report.results.push_back(MakeResult("cpu_ref", true, false,
                                      "failed to open \"results.json\"\npermission denied", {},
                                      kernellab::SummaryStatistics{}, std::nullopt));

  const auto json = kernellab::SerializeReportJson(report);

  KERNELLAB_CHECK(json.find("\\\"results.json\\\"") != std::string::npos);
  KERNELLAB_CHECK(json.find("\\npermission denied") != std::string::npos);
}

void TestJsonEscapesControlCharactersBelow0x20() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{1, 1, 1, 1};
  report.run_options = kernellab::RunOptions{0, 1};
  report.results.push_back(MakeResult("cpu_ref", true, false, std::string("bad\x01path"), {},
                                      kernellab::SummaryStatistics{}, std::nullopt));

  const auto json = kernellab::SerializeReportJson(report);

  KERNELLAB_CHECK(json.find("\\u0001") != std::string::npos);
}

void TestJsonEmitsNullForNonFiniteValues() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{1, 1, 1, 1};
  report.run_options = kernellab::RunOptions{0, 1};
  report.results.push_back(MakeResult("cpu_ref", true, true, "",
                                      {std::numeric_limits<double>::infinity()},
                                      kernellab::SummaryStatistics{
                                          1,
                                          std::numeric_limits<double>::quiet_NaN(),
                                          std::numeric_limits<double>::infinity(),
                                          1.0,
                                          1.0,
                                          0.0,
                                      },
                                      std::nullopt));

  const auto json = kernellab::SerializeReportJson(report);

  KERNELLAB_CHECK(json.find("\"iterations\":[{\"kernel_ms\":null,\"e2e_ms\":null}]") !=
                  std::string::npos);
  KERNELLAB_CHECK(json.find("\"mean\":null") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"median\":null") != std::string::npos);
}

void TestJsonIncludesVerificationMismatchDetails() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{2, 2, 2, 1};
  report.run_options = kernellab::RunOptions{0, 1};
  kernellab::VerificationResult verification;
  verification.passed = false;
  verification.first_mismatch = kernellab::MismatchInfo{
      1, 0, 3.5F, 7.0F, 3.5, 1.0,
  };
  report.results.push_back(MakeResult("cpu_omp", true, true, "", {0.1},
                                      kernellab::SummaryStatistics{1, 0.1, 0.1, 0.1, 0.1, 0.0},
                                      verification));

  const auto json = kernellab::SerializeReportJson(report);

  KERNELLAB_CHECK(json.find("\"verification\":{\"passed\":false") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"first_mismatch\":{\"row\":1,\"col\":0") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"reference_value\":3.5") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"candidate_value\":7") != std::string::npos);
  KERNELLAB_CHECK(json.find("\"count_over_tolerance\":0") != std::string::npos ||
                  json.find("\"count_over_tolerance\":1") != std::string::npos);
}

void TestJsonVerificationIsResultLevelNotInsideSummary() {
  // Regression: when gflops/pct_of_cublas were added before "summary"
  // the closing braces shifted and "verification" ended up nested
  // *inside* summary instead of as a sibling of it. Consumers expect
  // result.verification.passed, not result.summary.verification.passed.
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{2, 2, 2, 1};
  report.run_options = kernellab::RunOptions{0, 1};
  kernellab::VerificationResult verification;
  verification.passed = true;
  report.results.push_back(MakeResult("cpu_omp", true, true, "", {0.1},
                                      kernellab::SummaryStatistics{1, 0.1, 0.1, 0.1, 0.1, 0.0},
                                      verification));

  const auto json = kernellab::SerializeReportJson(report);

  // The summary block must close before the verification key starts.
  // After the fix the byte sequence is `}},"verification":` (two
  // closes: e2e_ms, then summary). When the bug regresses it becomes
  // just `,"verification":` while we're still inside summary.
  KERNELLAB_CHECK(json.find("}},\"verification\":") != std::string::npos);
}

void TestReferenceRowUsesReferenceVerificationStatusSentinel() {
  kernellab::ComparisonReport report;
  FillTestEnvironment(report);
  report.problem = kernellab::ProblemSpec{2, 2, 2, 1};
  report.run_options = kernellab::RunOptions{0, 1};
  report.results.push_back(MakeResult("cpu_ref", true, true, "", {0.1},
                                      kernellab::SummaryStatistics{1, 0.1, 0.1, 0.1, 0.1, 0.0},
                                      std::nullopt, true));

  const auto json = kernellab::SerializeReportJson(report);
  const auto csv = kernellab::SerializeReportCsv(report);

  KERNELLAB_CHECK(json.find("\"verification_status\":\"reference\"") != std::string::npos);
  KERNELLAB_CHECK(csv.find(",reference,") != std::string::npos);
}

void TestWriteTextFileCreatesArtifact() {
  const auto temp_dir = std::filesystem::temp_directory_path() / "kernellab_export_tests";
  std::filesystem::create_directories(temp_dir);
  const auto output_path = temp_dir / "write_ok.txt";

  const auto status = kernellab::WriteTextFile(output_path.string(), "alpha,beta,gamma\n");

  KERNELLAB_CHECK(status.ok);

  std::ifstream input(output_path);
  std::string contents;
  std::getline(input, contents);
  KERNELLAB_CHECK(contents == "alpha,beta,gamma");

  std::filesystem::remove(output_path);
}

void TestWriteTextFileCreatesParentDirectories() {
  const auto temp_dir = std::filesystem::temp_directory_path() / "kernellab_export_tests_nested";
  const auto output_path = temp_dir / "nested" / "report.json";
  std::filesystem::remove_all(temp_dir);

  const auto status = kernellab::WriteTextFile(output_path.string(), "{\"ok\":true}\n");

  KERNELLAB_CHECK(status.ok);
  KERNELLAB_CHECK(std::filesystem::exists(output_path));
  std::filesystem::remove_all(temp_dir);
}

void TestWriteTextFileReportsDirectoryWriteFailure() {
  const auto temp_dir = std::filesystem::temp_directory_path() / "kernellab_export_tests";
  std::filesystem::create_directories(temp_dir);

  const auto status = kernellab::WriteTextFile(temp_dir.string(), "this should fail");

  KERNELLAB_CHECK(!status.ok);
  KERNELLAB_CHECK(status.message.find("failed to open output file") != std::string::npos);
  KERNELLAB_CHECK(status.message.find(temp_dir.string()) != std::string::npos);
}

} // namespace

int main() {
  TestJsonContainsStructuredResultFields();
  TestJsonReportIsParseableWithMultipleResults();
  TestJsonIncludesPctOfCublasWhenCublasIsPresent();
  TestJsonEmitsNullPctOfCublasWhenCublasIsAbsent();
  TestJsonSerializesDoubleWithRoundTripPrecision();
  TestCsvContainsOneRowPerBackend();
  TestCsvIncludesPctOfCublasWhenCublasIsPresent();
  TestCsvEscapesErrorMessageField();
  TestCsvIncludesVerificationMismatchDetails();
  TestJsonEscapesSpecialCharactersInStrings();
  TestJsonEscapesControlCharactersBelow0x20();
  TestJsonEmitsNullForNonFiniteValues();
  TestJsonIncludesVerificationMismatchDetails();
  TestJsonVerificationIsResultLevelNotInsideSummary();
  TestReferenceRowUsesReferenceVerificationStatusSentinel();
  TestWriteTextFileCreatesArtifact();
  TestWriteTextFileCreatesParentDirectories();
  TestWriteTextFileReportsDirectoryWriteFailure();
  return 0;
}
