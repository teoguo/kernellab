#pragma once

#include "kernellab/core/result.hpp"

#include <string>
#include <string_view>

namespace kernellab {

std::string SerializeReportJson(const ComparisonReport& report);
std::string SerializeReportCsv(const ComparisonReport& report);
Status WriteTextFile(const std::string& path, std::string_view contents);

// RFC 4180-style CSV field escaping. Quotes the field iff it contains
// `"`, `,`, `\n`, or `\r`; doubles embedded `"`. Used by both the report
// serializer and the CLI's compare-output formatter; live here so the
// two callers can't drift.
std::string EscapeCsvField(std::string_view input);

}  // namespace kernellab
