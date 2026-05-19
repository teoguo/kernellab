#pragma once

#include "kernellab/core/matrix.hpp"
#include "kernellab/core/status.hpp"

#include <memory>

namespace kernellab {

struct IterationMeasurement {
  double kernel_ms = 0.0;
  double e2e_ms = 0.0;
};

class BackendSession {
 public:
  virtual ~BackendSession() = default;

  virtual Status Run(Matrix& output, IterationMeasurement& measurement) = 0;
};

using BackendSessionPtr = std::unique_ptr<BackendSession>;

}  // namespace kernellab
