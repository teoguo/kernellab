#pragma once

#include "kernellab/backend/backend.hpp"

#include <string_view>
#include <vector>

namespace kernellab {

class BackendRegistry {
 public:
  void Register(BackendPtr backend);

  const Backend* Find(std::string_view id) const noexcept;
  std::vector<const Backend*> All() const;

 private:
  std::vector<BackendPtr> backends_;
};

BackendRegistry CreateDefaultRegistry();

}  // namespace kernellab
