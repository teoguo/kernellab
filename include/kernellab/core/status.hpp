#pragma once

#include <string>

namespace kernellab {

struct Status {
  bool ok = true;
  std::string message;

  static Status Ok();
  static Status Error(std::string message);
};

}  // namespace kernellab
