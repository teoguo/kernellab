#include "kernellab/io/export.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace kernellab {

namespace {

std::string BoolString(const bool value) {
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

std::string EscapeJsonString(std::string_view input) {
  std::string escaped;
  escaped.reserve(input.size());

  for (const unsigned char ch : input) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      if (ch < 0x20U) {
        std::ostringstream codepoint;
        codepoint << "\\u00" << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(ch);
        escaped += codepoint.str();
      } else {
        escaped += static_cast<char>(ch);
      }
      break;
    }
  }

  return escaped;
}

std::string FormatPathError(std::string_view message, const std::filesystem::path& path) {
  std::ostringstream output;
  output << message << ": " << path.string();
  return output.str();
}

void ConfigureNumericPrecision(std::ostringstream& output) {
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
}

void AppendJsonNumber(std::ostringstream& output, const double value) {
  if (std::isfinite(value)) {
    output << value;
  } else {
    output << "null";
  }
}

double FindCublasGflops(const ComparisonReport& report) {
  for (const auto& result : report.results) {
    if (result.backend_id == "cublas" && result.success && result.gflops > 0.0) {
      return result.gflops;
    }
  }
  return 0.0;
}

double ComputePctOfCublas(const BackendRunResult& result, const double cublas_gflops) {
  if (cublas_gflops <= 0.0 || result.gflops <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return 100.0 * result.gflops / cublas_gflops;
}

} // namespace

std::string EscapeCsvField(std::string_view input) {
  bool needs_quotes = false;
  std::string escaped;
  escaped.reserve(input.size());

  for (const char ch : input) {
    if (ch == '"' || ch == ',' || ch == '\n' || ch == '\r') {
      needs_quotes = true;
    }
    if (ch == '"') {
      escaped += "\"\"";
    } else if (ch == '\n') {
      escaped += "\\n";
    } else if (ch == '\r') {
      escaped += "\\r";
    } else {
      escaped += ch;
    }
  }

  if (!needs_quotes) {
    return escaped;
  }
  return "\"" + escaped + "\"";
}

std::string SerializeReportJson(const ComparisonReport& report) {
  std::ostringstream output;
  ConfigureNumericPrecision(output);
  const double cublas_gflops = FindCublasGflops(report);
  output << "{";
  output << "\"schema_version\":" << report.schema_version << ",";
  output << "\"environment\":{"
         << "\"timestamp_utc\":\"" << EscapeJsonString(report.environment.timestamp_utc) << "\","
         << "\"cpu_model\":\"" << EscapeJsonString(report.environment.cpu_model) << "\","
         << "\"gpu_name\":\"" << EscapeJsonString(report.environment.gpu_name) << "\","
         << "\"openmp_threads\":" << report.environment.openmp_threads << "},";
  output << "\"problem\":{"
         << "\"m\":" << report.problem.m << ","
         << "\"n\":" << report.problem.n << ","
         << "\"k\":" << report.problem.k << ","
         << "\"seed\":" << report.problem.seed << "},";
  output << "\"run_options\":{"
         << "\"warmup_iterations\":" << report.run_options.warmup_iterations << ","
         << "\"timed_iterations\":" << report.run_options.timed_iterations << "},";
  output << "\"verify_options\":{"
         << "\"atol\":" << report.verify_options.atol << ","
         << "\"rtol\":" << report.verify_options.rtol << "},";
  output << "\"results\":[";
  for (std::size_t index = 0; index < report.results.size(); ++index) {
    const auto& result = report.results[index];
    if (index > 0) {
      output << ",";
    }
    output << "{";
    output << "\"backend_id\":\"" << EscapeJsonString(result.backend_id) << "\",";
    output << "\"available\":" << BoolString(result.available) << ",";
    output << "\"success\":" << BoolString(result.success) << ",";
    output << "\"verification_status\":\"" << VerificationStatusString(result) << "\",";
    output << "\"error_message\":\"" << EscapeJsonString(result.error_message) << "\",";
    output << "\"iterations\":[";
    for (std::size_t timing_index = 0; timing_index < result.iterations.size(); ++timing_index) {
      if (timing_index > 0) {
        output << ",";
      }
      output << "{";
      output << "\"kernel_ms\":";
      AppendJsonNumber(output, result.iterations[timing_index].kernel_ms);
      output << ",\"e2e_ms\":";
      AppendJsonNumber(output, result.iterations[timing_index].e2e_ms);
      output << "}";
    }
    output << "],";
    output << "\"gflops\":";
    AppendJsonNumber(output, result.gflops);
    output << ",\"pct_of_cublas\":";
    AppendJsonNumber(output, ComputePctOfCublas(result, cublas_gflops));
    output << ",";
    output << "\"summary\":{";
    output << "\"kernel_ms\":{"
           << "\"count\":" << result.kernel_statistics.count << ","
           << "\"mean\":";
    AppendJsonNumber(output, result.kernel_statistics.mean);
    output << ",\"median\":";
    AppendJsonNumber(output, result.kernel_statistics.median);
    output << ",\"min\":";
    AppendJsonNumber(output, result.kernel_statistics.min);
    output << ",\"max\":";
    AppendJsonNumber(output, result.kernel_statistics.max);
    output << ",\"stddev\":";
    AppendJsonNumber(output, result.kernel_statistics.stddev);
    output << "},";
    output << "\"e2e_ms\":{"
           << "\"count\":" << result.statistics.count << ","
           << "\"mean\":";
    AppendJsonNumber(output, result.statistics.mean);
    output << ",\"median\":";
    AppendJsonNumber(output, result.statistics.median);
    output << ",\"min\":";
    AppendJsonNumber(output, result.statistics.min);
    output << ",\"max\":";
    AppendJsonNumber(output, result.statistics.max);
    output << ",\"stddev\":";
    AppendJsonNumber(output, result.statistics.stddev);
    output << "}";
    // Close the summary object before emitting verification as a
    // sibling of summary (not nested inside it).
    output << "},";
    if (result.verification.has_value()) {
      output << "\"verification\":{\"passed\":" << BoolString(result.verification->passed);
      output << ",\"max_absolute_error\":";
      AppendJsonNumber(output, result.verification->max_absolute_error);
      output << ",\"max_relative_error\":";
      AppendJsonNumber(output, result.verification->max_relative_error);
      output << ",\"count_over_tolerance\":" << result.verification->count_over_tolerance;
      if (result.verification->first_mismatch.has_value()) {
        const auto& mismatch = *result.verification->first_mismatch;
        output << ",\"first_mismatch\":{"
               << "\"row\":" << mismatch.row << ","
               << "\"col\":" << mismatch.col << ","
               << "\"reference_value\":" << mismatch.reference_value << ","
               << "\"candidate_value\":" << mismatch.candidate_value << ","
               << "\"absolute_error\":" << mismatch.absolute_error << ","
               << "\"relative_error\":" << mismatch.relative_error << "}";
      }
      output << "}";
    } else {
      output << "\"verification\":null";
    }
    output << "}";
  }
  output << "]";
  output << "}";
  return output.str();
}

std::string SerializeReportCsv(const ComparisonReport& report) {
  std::ostringstream output;
  ConfigureNumericPrecision(output);
  output << "schema_version,timestamp_utc,cpu_model,gpu_name,openmp_threads,backend_id,m,n,k,seed,"
            "warmup_iterations,timed_iterations,available,success,verification_status,gflops,pct_"
            "of_cublas,"
            "e2e_mean_ms,e2e_median_ms,e2e_min_ms,e2e_max_ms,e2e_stddev_ms,kernel_mean_ms,"
            "kernel_median_ms,kernel_min_ms,kernel_max_ms,kernel_stddev_ms,error_message,"
            "mismatch_row,mismatch_col,reference_value,candidate_value,absolute_error,"
            "relative_error,max_absolute_error,max_relative_error,count_over_tolerance\n";
  const double cublas_gflops = FindCublasGflops(report);
  for (const auto& result : report.results) {
    output << report.schema_version << "," << EscapeCsvField(report.environment.timestamp_utc)
           << "," << EscapeCsvField(report.environment.cpu_model) << ","
           << EscapeCsvField(report.environment.gpu_name) << ","
           << report.environment.openmp_threads << "," << EscapeCsvField(result.backend_id) << ","
           << report.problem.m << "," << report.problem.n << "," << report.problem.k << ","
           << report.problem.seed << "," << report.run_options.warmup_iterations << ","
           << report.run_options.timed_iterations << "," << BoolString(result.available) << ","
           << BoolString(result.success) << "," << VerificationStatusString(result) << ","
           << result.gflops << ",";
    const double pct_of_cublas = ComputePctOfCublas(result, cublas_gflops);
    if (std::isfinite(pct_of_cublas)) {
      output << pct_of_cublas;
    }
    output << "," << result.statistics.mean << "," << result.statistics.median << ","
           << result.statistics.min << "," << result.statistics.max << ","
           << result.statistics.stddev << "," << result.kernel_statistics.mean << ","
           << result.kernel_statistics.median << "," << result.kernel_statistics.min << ","
           << result.kernel_statistics.max << "," << result.kernel_statistics.stddev << ","
           << EscapeCsvField(result.error_message);
    if (result.verification.has_value() && result.verification->first_mismatch.has_value()) {
      const auto& mismatch = *result.verification->first_mismatch;
      output << "," << mismatch.row << "," << mismatch.col << "," << mismatch.reference_value << ","
             << mismatch.candidate_value << "," << mismatch.absolute_error << ","
             << mismatch.relative_error << "," << result.verification->max_absolute_error << ","
             << result.verification->max_relative_error << ","
             << result.verification->count_over_tolerance;
    } else if (result.verification.has_value()) {
      output << ",,,,,," << result.verification->max_absolute_error << ","
             << result.verification->max_relative_error << ","
             << result.verification->count_over_tolerance;
    } else {
      output << ",,,,,,,,,";
    }
    output << "\n";
  }
  return output.str();
}

Status WriteTextFile(const std::string& path, std::string_view contents) {
  const auto output_path = std::filesystem::path(path);
  if (output_path.has_parent_path()) {
    std::error_code error;
    std::filesystem::create_directories(output_path.parent_path(), error);
    if (error) {
      return Status::Error(
          FormatPathError("failed to create output directory", output_path.parent_path()));
    }
  }

  std::ofstream output(path);
  if (!output.is_open()) {
    return Status::Error(FormatPathError("failed to open output file", output_path));
  }
  output << contents;
  if (!output.good()) {
    return Status::Error(FormatPathError("failed to write output file", output_path));
  }
  return Status::Ok();
}

} // namespace kernellab
