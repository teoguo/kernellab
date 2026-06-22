#include "kernellab/backend/registry.hpp"
#include "kernellab/backend/cuda_support.hpp"

#include <utility>

namespace kernellab {

BackendPtr MakeCpuRefBackend();
BackendPtr MakeCpuNaiveBackend();
BackendPtr MakeCpuOmpBackend();

void BackendRegistry::Register(BackendPtr backend) {
  backends_.push_back(std::move(backend));
}

const Backend* BackendRegistry::Find(const std::string_view id) const noexcept {
  for (const auto& backend : backends_) {
    if (backend->id() == id) {
      return backend.get();
    }
  }
  return nullptr;
}

std::vector<const Backend*> BackendRegistry::All() const {
  std::vector<const Backend*> result;
  result.reserve(backends_.size());
  for (const auto& backend : backends_) {
    result.push_back(backend.get());
  }
  return result;
}

BackendRegistry CreateDefaultRegistry() {
  BackendRegistry registry;
  registry.Register(MakeCpuRefBackend());
  registry.Register(MakeCpuNaiveBackend());
  registry.Register(MakeCpuOmpBackend());
  registry.Register(MakeCudaNaiveBackend());
  registry.Register(MakeCudaSmemBackend());
  registry.Register(MakeCudaRegBackend());
  registry.Register(MakeCudaRegV2Backend());
  registry.Register(MakeCublasBackend());
  return registry;
}

}  // namespace kernellab
