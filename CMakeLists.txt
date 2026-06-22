cmake_minimum_required(VERSION 3.28)

project(
  kernellab
  VERSION 0.1.0
  LANGUAGES CXX)

include(cmake/KernellabOptions.cmake)
include(CheckLanguage)

set(KERNELLAB_GENERATED_INCLUDE_DIR "${PROJECT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY "${KERNELLAB_GENERATED_INCLUDE_DIR}/kernellab")
configure_file(
  include/kernellab/build_info.hpp.in
  "${KERNELLAB_GENERATED_INCLUDE_DIR}/kernellab/build_info.hpp"
  @ONLY)

set(KERNELLAB_OPENMP_ENABLED OFF)
if(KERNELLAB_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    set(KERNELLAB_OPENMP_ENABLED ON)
  endif()
endif()

set(KERNELLAB_CUDA_ENABLED OFF)
if(KERNELLAB_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(KERNELLAB_CUDA_ENABLED ON)
    endif()
  endif()
endif()

add_library(kernellab_core STATIC
  src/benchmark/stats.cpp
  src/backends/cpu/cpu_naive_backend.cpp
  src/backends/cpu/cpu_omp_backend.cpp
  src/backends/cpu/cpu_ref_backend.cpp
  src/backends/cuda/cuda_support.cpp
  src/core/matrix.cpp
  src/core/problem.cpp
  src/core/result.cpp
  src/io/export.cpp
  src/app/cli.cpp
  src/runtime/input_generator.cpp
  src/runtime/executor.cpp
  src/runtime/registry.cpp
  src/verify/verification.cpp)

target_compile_features(kernellab_core PUBLIC cxx_std_17)
target_include_directories(kernellab_core
  PUBLIC
    ${KERNELLAB_GENERATED_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(kernellab_core
  PUBLIC
    KERNELLAB_HAS_OPENMP=$<BOOL:${KERNELLAB_OPENMP_ENABLED}>
    KERNELLAB_HAS_CUDA=$<BOOL:${KERNELLAB_CUDA_ENABLED}>)

if(KERNELLAB_ENABLE_WARNINGS)
  target_compile_options(kernellab_core
    PUBLIC
      $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>)
endif()

if(KERNELLAB_ENABLE_SANITIZERS)
  target_compile_options(kernellab_core
    PUBLIC
      $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-fsanitize=address,undefined;-fno-omit-frame-pointer>)
  target_link_options(kernellab_core
    PUBLIC
      $<$<LINK_LANGUAGE:CXX>:-fsanitize=address,undefined>)
endif()

if(KERNELLAB_OPENMP_ENABLED)
  target_link_libraries(kernellab_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(KERNELLAB_CUDA_ENABLED)
  target_sources(kernellab_core
    PRIVATE
      src/backends/cuda/cuda_naive.cu
      src/backends/cuda/cuda_smem.cu
      src/backends/cuda/cuda_reg.cu
      src/backends/cuda/cuda_reg_v2.cu
      src/backends/cuda/cublas_backend.cpp)
  set_target_properties(kernellab_core PROPERTIES
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "${KERNELLAB_CUDA_ARCHITECTURES}")
  target_link_libraries(kernellab_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

add_executable(kernellab
  src/app/main.cpp)

target_compile_features(kernellab PRIVATE cxx_std_17)
target_link_libraries(kernellab PRIVATE kernellab_core)

if(KERNELLAB_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests)
endif()
