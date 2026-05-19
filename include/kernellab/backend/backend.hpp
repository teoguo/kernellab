#pragma once

#include "kernellab/backend/session.hpp"
#include "kernellab/core/matrix.hpp"
#include "kernellab/core/status.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace kernellab {

class Backend {
 public:
  virtual ~Backend() = default;

  virtual std::string_view id() const noexcept = 0;
  virtual bool is_available() const noexcept = 0;

  // Human-readable reason when is_available() returns false. The base
  // default is generic; backends that have a specific failure mode
  // (OpenMP disabled, CUDA driver too old, no GPU detected, ...)
  // should override.  Only consulted when is_available() == false.
  virtual std::string unavailable_reason() const { return "unavailable"; }

  virtual Status Prepare(const Matrix& a,
                         const Matrix& b,
                         BackendSessionPtr& session) const = 0;
};

using BackendPtr = std::unique_ptr<Backend>;

}  // namespace kernellab
