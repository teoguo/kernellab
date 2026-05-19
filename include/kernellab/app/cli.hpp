#pragma once

#include <string>
#include <vector>

namespace kernellab {

int RunCli(const std::vector<std::string>& args, std::string& stdout_text, std::string& stderr_text);

}  // namespace kernellab
