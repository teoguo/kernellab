#include "kernellab/app/cli.hpp"

#include "kernellab/backend/cuda_support.hpp"
#include "kernellab/backend/registry.hpp"
#include "kernellab/io/export.hpp"
#include "kernellab/runtime/executor.hpp"
#include "kernellab/version.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string_view>
#include <vector>

namespace kernellab {

namespace {

std::string UsageText() {
  return "Usage: kernellab <subcommand> [options]\n"
         "\n"
         "Subcommands:\n"
         "  run      Run one backend on a deterministic GEMM problem\n"
         "  verify   Verify one backend against cpu_ref\n"
         "  compare  Compare multiple backends on the same inputs\n"
         "\n"
         "Meta commands:\n"
         "  --version\n"
         "  --list-backends\n"
         "  --print-device-info\n"
         "\n"
         "Common options:\n"
         "  --m <rows> --n <cols> --k <inner>\n"
         "  --iterations <count> --warmup <count> --seed <value>\n"
         "  --json-out <path> --csv-out <path>\n";
}

std::string RunUsageText() {
  return "Usage: kernellab run --backend <id> --m <rows> --n <cols> --k <inner> [options]\n"
         "\n"
         "Options:\n"
         "  --iterations <count> --warmup <count> --seed <value>\n"
         "  --json-out <path> --csv-out <path>\n";
}

std::string VerifyUsageText() {
  return "Usage: kernellab verify --backend <id> --m <rows> --n <cols> --k <inner> [options]\n"
         "\n"
         "Options:\n"
         "  --seed <value>\n"
         "  --atol <value> --rtol <value>\n"
         "  --json-out <path> --csv-out <path>\n";
}

std::string CompareUsageText() {
  return "Usage: kernellab compare --backends <id1,id2,...> --m <rows> --n <cols> --k <inner> "
         "[options]\n"
         "\n"
         "Options:\n"
         "  --iterations <count> --warmup <count> --seed <value>\n"
         "  --atol <value> --rtol <value>\n"
         "  --json-out <path> --csv-out <path>\n";
}

std::string UsageTextForSubcommand(std::string_view subcommand) {
  if (subcommand == "run") {
    return RunUsageText();
  }
  if (subcommand == "verify") {
    return VerifyUsageText();
  }
  if (subcommand == "compare") {
    return CompareUsageText();
  }
  return UsageText();
}

std::string BoolString(bool value) {
  return value ? "true" : "false";
}

std::string VerificationStatusString(const BackendRunResult& result) {
  if (result.is_reference) {
    return "reference";
  }
  if (!result.verification.has_value()) {
    return "not_run";
  }
  return result.verification->passed ? "passed" : "failed";
}

struct ParsedArguments {
  std::map<std::string, std::string> values;
};

class CliError final : public std::runtime_error {
public:
  explicit CliError(std::string message) : std::runtime_error(std::move(message)) {}
};

std::string TrimWhitespace(std::string_view input);

bool ParseArguments(const std::vector<std::string>& args, ParsedArguments& parsed,
                    std::string& error) {
  std::size_t index = 1;
  while (index < args.size()) {
    const auto& token = args[index];
    if (token.rfind("--", 0) != 0) {
      error = "expected flag starting with --";
      return false;
    }
    if ((index + 1) >= args.size()) {
      error = "missing value for " + token;
      return false;
    }
    const std::string key = token.substr(2);
    if (parsed.values.find(key) != parsed.values.end()) {
      error = "duplicate flag --" + key;
      return false;
    }
    parsed.values[key] = args[index + 1];
    index += 2;
  }
  return true;
}

std::string GetOrDefault(const ParsedArguments& parsed, std::string_view key,
                         const std::string& fallback) {
  const auto iterator = parsed.values.find(std::string(key));
  if (iterator == parsed.values.end()) {
    return fallback;
  }
  return iterator->second;
}

bool RequireValue(const ParsedArguments& parsed, std::string_view key, std::string& value,
                  std::string& error) {
  const auto iterator = parsed.values.find(std::string(key));
  if (iterator == parsed.values.end()) {
    error = "missing required flag --" + std::string(key);
    return false;
  }
  value = iterator->second;
  return true;
}

bool RequireTrimmedValue(const ParsedArguments& parsed, std::string_view key, std::string& value,
                         std::string& error) {
  if (!RequireValue(parsed, key, value, error)) {
    return false;
  }
  value = TrimWhitespace(value);
  if (value.empty()) {
    error = "missing value for --" + std::string(key);
    return false;
  }
  return true;
}

bool ValidateKnownFlags(const ParsedArguments& parsed,
                        std::initializer_list<std::string_view> allowed_flags, std::string& error) {
  std::vector<std::string> allowed;
  allowed.reserve(allowed_flags.size());
  for (const std::string_view flag : allowed_flags) {
    allowed.emplace_back(flag);
  }
  std::sort(allowed.begin(), allowed.end());

  for (const auto& entry : parsed.values) {
    if (!std::binary_search(allowed.begin(), allowed.end(), entry.first)) {
      error = "unknown flag --" + entry.first;
      return false;
    }
  }
  return true;
}

std::int64_t ParseInt64(std::string_view text, std::string_view context) {
  std::int64_t value = 0;
  const auto* begin = text.data();
  const auto* end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec == std::errc::invalid_argument || ptr != end) {
    throw CliError("invalid numeric value in " + std::string(context));
  }
  if (ec == std::errc::result_out_of_range) {
    throw CliError(std::string(context) + " value is out of range");
  }
  return value;
}

std::uint64_t ParseUInt64(std::string_view text, std::string_view context) {
  std::uint64_t value = 0;
  const auto* begin = text.data();
  const auto* end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec == std::errc::invalid_argument || ptr != end) {
    throw CliError("invalid numeric value in " + std::string(context));
  }
  if (ec == std::errc::result_out_of_range) {
    throw CliError(std::string(context) + " value is out of range");
  }
  return value;
}

std::int32_t ParseInt32Option(const ParsedArguments& parsed, std::string_view key,
                              const std::string& fallback) {
  const std::int64_t value = ParseInt64(GetOrDefault(parsed, key, fallback), key);
  if (value < std::numeric_limits<std::int32_t>::min() ||
      value > std::numeric_limits<std::int32_t>::max()) {
    throw CliError(std::string(key) + " value is out of range");
  }
  return static_cast<std::int32_t>(value);
}

double ParseDouble(std::string_view text, std::string_view context) {
  std::string owned(text);
  char* parse_end = nullptr;
  errno = 0;
  const double value = std::strtod(owned.c_str(), &parse_end);
  if (parse_end == owned.c_str() || *parse_end != '\0') {
    throw CliError("invalid numeric value in " + std::string(context));
  }
  if (errno == ERANGE) {
    throw CliError(std::string(context) + " value is out of range");
  }
  return value;
}

ProblemSpec BuildProblem(const ParsedArguments& parsed) {
  ProblemSpec problem{
      ParseInt64(GetOrDefault(parsed, "m", "0"), "problem specification"),
      ParseInt64(GetOrDefault(parsed, "n", "0"), "problem specification"),
      ParseInt64(GetOrDefault(parsed, "k", "0"), "problem specification"),
      ParseUInt64(GetOrDefault(parsed, "seed", "0"), "problem specification"),
  };
  if (!IsValidProblemSpec(problem)) {
    throw CliError("problem dimensions must be positive");
  }
  return problem;
}

RunOptions BuildRunOptions(const ParsedArguments& parsed) {
  RunOptions options{
      ParseInt32Option(parsed, "warmup", "2"),
      ParseInt32Option(parsed, "iterations", "10"),
  };
  if (options.warmup_iterations < 0 || options.timed_iterations <= 0) {
    throw CliError("warmup must be non-negative and iterations must be positive");
  }
  return options;
}

RunOptions VerifyRunOptions() {
  return RunOptions{0, 1};
}

VerifyOptions BuildVerifyOptions(const ParsedArguments& parsed, const ProblemSpec& problem) {
  const auto defaults = DefaultTolerance(problem);
  VerifyOptions options{
      ParseDouble(GetOrDefault(parsed, "atol", std::to_string(defaults.atol)), "verify options"),
      ParseDouble(GetOrDefault(parsed, "rtol", std::to_string(defaults.rtol)), "verify options"),
  };
  if (options.atol < 0.0 || options.rtol < 0.0) {
    throw CliError("atol and rtol must be non-negative");
  }
  return options;
}

const Backend* FindBackendOrNull(const BackendRegistry& registry, const std::string& id) {
  return registry.Find(id);
}

void MaybeWriteExports(const ComparisonReport& report, const ParsedArguments& parsed,
                       std::string& error) {
  const auto json_path = GetOrDefault(parsed, "json-out", "");
  if (!json_path.empty()) {
    const auto status = WriteTextFile(json_path, SerializeReportJson(report));
    if (!status.ok) {
      error = status.message;
      return;
    }
  }

  const auto csv_path = GetOrDefault(parsed, "csv-out", "");
  if (!csv_path.empty()) {
    const auto status = WriteTextFile(csv_path, SerializeReportCsv(report));
    if (!status.ok) {
      error = status.message;
    }
  }
}

std::vector<std::string> SplitBackendList(const std::string& raw) {
  std::vector<std::string> result;
  std::size_t start = 0;
  while (start < raw.size()) {
    const auto comma = raw.find(',', start);
    if (comma == std::string::npos) {
      result.push_back(raw.substr(start));
      break;
    }
    result.push_back(raw.substr(start, comma - start));
    start = comma + 1;
  }
  return result;
}

std::string TrimWhitespace(std::string_view input) {
  std::size_t begin = 0;
  while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
    ++begin;
  }

  std::size_t end = input.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
    --end;
  }

  return std::string(input.substr(begin, end - begin));
}

std::string AvailabilityReason(const Backend& backend) {
  return backend.is_available() ? "available" : backend.unavailable_reason();
}

std::string PrintBackendList(const BackendRegistry& registry) {
  std::ostringstream output;
  const auto backends = registry.All();
  for (std::size_t index = 0; index < backends.size(); ++index) {
    const Backend* backend = backends[index];
    if (backend == nullptr) {
      continue;
    }
    if (index > 0) {
      output << "\n";
    }
    output << backend->id() << "\t" << BoolString(backend->is_available()) << "\t"
           << AvailabilityReason(*backend);
  }
  return output.str();
}

std::string PrintDeviceInfo() {
  const auto environment = CollectEnvironmentInfo();
  std::ostringstream output;
  output << "cpu_model=" << environment.cpu_model << "\n"
         << "gpu_name=" << environment.gpu_name << "\n"
         << "openmp_threads=" << environment.openmp_threads;
  const auto& cuda_info = GetCudaRuntimeInfo();
  output << "\n"
         << "cuda_compiled=" << BoolString(cuda_info.compiled) << "\n"
         << "cuda_available=" << BoolString(cuda_info.available) << "\n"
         << "cuda_device_count=" << cuda_info.device_count;
  if (!cuda_info.error_message.empty()) {
    output << "\n"
           << "cuda_error=" << cuda_info.error_message;
  }
  return output.str();
}

} // namespace

int RunCli(const std::vector<std::string>& args, std::string& stdout_text,
           std::string& stderr_text) {
  stdout_text.clear();
  stderr_text.clear();

  try {
    if (args.empty()) {
      stderr_text = UsageText();
      return 1;
    }

    const std::string& subcommand = args.front();
    if (subcommand == "--version") {
      stdout_text = std::string("kernellab ") + kVersion;
      return 0;
    }
    if (subcommand == "--list-backends") {
      stdout_text = PrintBackendList(CreateDefaultRegistry());
      return 0;
    }
    if (subcommand == "--print-device-info") {
      stdout_text = PrintDeviceInfo();
      return 0;
    }
    if (subcommand == "--help" || subcommand == "-h" || subcommand == "help") {
      stdout_text = UsageText();
      return 0;
    }
    if (args.size() >= 2) {
      const std::string& maybe_help = args[1];
      if (maybe_help == "--help" || maybe_help == "-h" || maybe_help == "help") {
        stdout_text = UsageTextForSubcommand(subcommand);
        return 0;
      }
    }

    ParsedArguments parsed;
    if (!ParseArguments(args, parsed, stderr_text)) {
      return 1;
    }

    BackendRegistry registry = CreateDefaultRegistry();
    const Backend* cpu_ref = registry.Find("cpu_ref");
    if (cpu_ref == nullptr) {
      stderr_text = "cpu_ref backend is not registered";
      return 1;
    }

    if (subcommand == "run") {
      if (!ValidateKnownFlags(
              parsed,
              {"backend", "m", "n", "k", "iterations", "warmup", "seed", "json-out", "csv-out"},
              stderr_text)) {
        return 1;
      }
      std::string backend_id;
      if (!RequireTrimmedValue(parsed, "backend", backend_id, stderr_text)) {
        return 1;
      }
      const Backend* backend = FindBackendOrNull(registry, backend_id);
      if (backend == nullptr) {
        stderr_text = "unknown backend: " + backend_id;
        return 1;
      }

      const auto problem = BuildProblem(parsed);
      const auto run_options = BuildRunOptions(parsed);
      const auto result = RunBackend(*backend, problem, run_options);
      ComparisonReport report;
      report.environment = CollectEnvironmentInfo();
      report.problem = problem;
      report.run_options = run_options;
      report.results.push_back(result);
      MaybeWriteExports(report, parsed, stderr_text);
      if (!stderr_text.empty()) {
        return 1;
      }
      if (!result.success) {
        stderr_text = result.error_message.empty() ? "backend run failed" : result.error_message;
        return 1;
      }

      std::ostringstream output;
      output << "backend=" << result.backend_id << " available=" << BoolString(result.available)
             << " success=" << BoolString(result.success) << " mean_ms=" << result.statistics.mean
             << " kernel_mean_ms=" << result.kernel_statistics.mean
             << " median_ms=" << result.statistics.median
             << " gflops=" << result.gflops;
      stdout_text = output.str();
      return 0;
    }

    if (subcommand == "verify") {
      if (!ValidateKnownFlags(
              parsed, {"backend", "m", "n", "k", "seed", "atol", "rtol", "json-out", "csv-out"},
              stderr_text)) {
        return 1;
      }
      std::string backend_id;
      if (!RequireTrimmedValue(parsed, "backend", backend_id, stderr_text)) {
        return 1;
      }
      const Backend* backend = FindBackendOrNull(registry, backend_id);
      if (backend == nullptr) {
        stderr_text = "unknown backend: " + backend_id;
        return 1;
      }

      const auto problem = BuildProblem(parsed);
      const auto run_options = VerifyRunOptions();
      const auto verify_options = BuildVerifyOptions(parsed, problem);
      const auto result = VerifyBackend(*cpu_ref, *backend, problem, run_options, verify_options);
      ComparisonReport report;
      report.environment = CollectEnvironmentInfo();
      report.problem = problem;
      report.run_options = run_options;
      report.verify_options = verify_options;
      report.results.push_back(result);
      MaybeWriteExports(report, parsed, stderr_text);
      if (!stderr_text.empty()) {
        return 1;
      }

      const bool passed = result.verification.has_value() ? result.verification->passed : false;
      if (!result.success) {
        stderr_text =
            result.error_message.empty() ? "backend verification failed" : result.error_message;
        return 1;
      }
      if (!passed) {
        stderr_text = DescribeVerificationFailure(*result.verification);
        return 1;
      }

      std::ostringstream output;
      output << "backend=" << result.backend_id << " available=" << BoolString(result.available)
             << " success=" << BoolString(result.success) << " mean_ms=" << result.statistics.mean
             << " kernel_mean_ms=" << result.kernel_statistics.mean
             << " gflops=" << result.gflops
             << " verification_passed=" << BoolString(passed);
      stdout_text = output.str();
      return 0;
    }

    if (subcommand == "compare") {
      if (!ValidateKnownFlags(parsed,
                              {"backends", "m", "n", "k", "iterations", "warmup", "seed", "atol",
                               "rtol", "json-out", "csv-out"},
                              stderr_text)) {
        return 1;
      }
      std::string backend_list;
      if (!RequireValue(parsed, "backends", backend_list, stderr_text)) {
        return 1;
      }
      backend_list = TrimWhitespace(backend_list);
      if (backend_list.empty()) {
        stderr_text = "at least one backend must be provided";
        return 1;
      }

      std::vector<const Backend*> backends;
      for (const auto& backend_token : SplitBackendList(backend_list)) {
        const std::string backend_id = TrimWhitespace(backend_token);
        if (backend_id.empty()) {
          stderr_text = "empty backend id in --backends list";
          return 1;
        }
        const Backend* backend = FindBackendOrNull(registry, backend_id);
        if (backend == nullptr) {
          stderr_text = "unknown backend: " + backend_id;
          return 1;
        }
        backends.push_back(backend);
      }

      const auto problem = BuildProblem(parsed);
      const auto run_options = BuildRunOptions(parsed);
      const auto verify_options = BuildVerifyOptions(parsed, problem);
      const auto report = CompareBackends(*cpu_ref, backends, problem, run_options, verify_options);
      MaybeWriteExports(report, parsed, stderr_text);
      if (!stderr_text.empty()) {
        return 1;
      }

      // Find cublas (if present + successful) to compute %-of-cublas for
      // each row.  Pure presentation; doesn't affect machine-readable
      // gflops column.
      double cublas_gflops = 0.0;
      for (const auto& result : report.results) {
        if (result.backend_id == "cublas" && result.success && result.gflops > 0.0) {
          cublas_gflops = result.gflops;
          break;
        }
      }

      std::ostringstream output;
      output << "backend_id,available,success,verification_status,mean_ms,kernel_mean_ms,"
                "gflops,pct_of_cublas,error_message\n";
      bool all_ok = true;
      for (const auto& result : report.results) {
        bool verification_ok = true;
        if (result.verification.has_value()) {
          verification_ok = result.verification->passed;
        }
        if (!result.success || !verification_ok) {
          all_ok = false;
        }

        output << result.backend_id << "," << BoolString(result.available) << ","
               << BoolString(result.success) << ",";
        output << VerificationStatusString(result) << "," << result.statistics.mean << ","
               << result.kernel_statistics.mean << ","
               << result.gflops << ",";
        if (cublas_gflops > 0.0 && result.gflops > 0.0) {
          output << (100.0 * result.gflops / cublas_gflops);
        }
        output << "," << EscapeCsvField(result.error_message)
               << "\n";
      }
      stdout_text = output.str();
      return all_ok ? 0 : 1;
    }

    stderr_text = "unknown subcommand: " + subcommand + "\n\n" + UsageText();
    return 1;
  } catch (const CliError& error) {
    stderr_text = error.what();
    stdout_text.clear();
    return 1;
  } catch (const std::exception& error) {
    stderr_text = error.what();
    stdout_text.clear();
    return 1;
  }
}

} // namespace kernellab
