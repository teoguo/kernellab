#include "kernellab/core/result.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <sstream>
#include <utility>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(KERNELLAB_HAS_OPENMP) && KERNELLAB_HAS_OPENMP
#include <omp.h>
#endif

#if defined(KERNELLAB_HAS_CUDA) && KERNELLAB_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace kernellab {

namespace {

std::string CurrentTimestampUtc() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm utc_tm{};
#if defined(_WIN32)
  gmtime_s(&utc_tm, &now_time);
#else
  gmtime_r(&now_time, &utc_tm);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_tm);
  return buffer;
}

std::string ReadFirstMatchingCpuInfoLine() {
#if defined(__linux__)
  std::ifstream input("/proc/cpuinfo");
  std::string line;
  while (std::getline(input, line)) {
    constexpr std::string_view prefix = "model name\t: ";
    if (line.rfind(prefix.data(), 0) == 0) {
      return line.substr(prefix.size());
    }
  }
#elif defined(__APPLE__)
  char value[256];
  std::size_t size = sizeof(value);
  if (sysctlbyname("machdep.cpu.brand_string", value, &size, nullptr, 0) == 0 && size > 0) {
    return std::string(value, size - 1);
  }
#endif
  return "unknown";
}

std::string DetectGpuName() {
#if defined(KERNELLAB_HAS_CUDA) && KERNELLAB_HAS_CUDA
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return "none";
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
    return "unknown";
  }
  return properties.name;
#else
  return "none";
#endif
}

int DetectOpenMpThreads() {
#if defined(KERNELLAB_HAS_OPENMP) && KERNELLAB_HAS_OPENMP
  return omp_get_max_threads();
#else
  return 0;
#endif
}

}  // namespace

Status Status::Ok() { return {}; }

Status Status::Error(std::string message) {
  return Status{false, std::move(message)};
}

double ComputeGflops(const ProblemSpec& problem, const double kernel_ms) noexcept {
  if (kernel_ms <= 0.0 || problem.m <= 0 || problem.n <= 0 || problem.k <= 0) {
    return 0.0;
  }
  const double flops = 2.0
      * static_cast<double>(problem.m)
      * static_cast<double>(problem.n)
      * static_cast<double>(problem.k);
  const double seconds = kernel_ms * 1e-3;
  return (flops / seconds) / 1e9;
}

EnvironmentInfo CollectEnvironmentInfo() {
  EnvironmentInfo info;
  info.timestamp_utc = CurrentTimestampUtc();
  info.cpu_model = ReadFirstMatchingCpuInfoLine();
  info.gpu_name = DetectGpuName();
  info.openmp_threads = DetectOpenMpThreads();
  return info;
}

}  // namespace kernellab
