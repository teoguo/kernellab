#include "kernellab/app/cli.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int index = 1; index < argc; ++index) {
    args.emplace_back(argv[index]);
  }

  std::string stdout_text;
  std::string stderr_text;
  const int exit_code = kernellab::RunCli(args, stdout_text, stderr_text);
  if (!stdout_text.empty()) {
    std::cout << stdout_text << '\n';
  }
  if (!stderr_text.empty()) {
    std::cerr << stderr_text << '\n';
  }
  return exit_code;
}
