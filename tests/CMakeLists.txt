add_executable(kernellab_smoke
  test_smoke.cpp)

target_compile_features(kernellab_smoke PRIVATE cxx_std_17)
target_include_directories(kernellab_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernellab_smoke PRIVATE kernellab_core)

add_test(
  NAME kernellab_smoke
  COMMAND kernellab_smoke)

add_executable(kernellab_test_input_generator
  test_input_generator.cpp)

target_compile_features(kernellab_test_input_generator PRIVATE cxx_std_17)
target_include_directories(kernellab_test_input_generator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernellab_test_input_generator PRIVATE kernellab_core)

add_test(
  NAME input_generator
  COMMAND kernellab_test_input_generator)

add_executable(kernellab_test_stats
  test_stats.cpp)

target_compile_features(kernellab_test_stats PRIVATE cxx_std_17)
target_include_directories(kernellab_test_stats PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernellab_test_stats PRIVATE kernellab_core)

add_test(
  NAME stats
  COMMAND kernellab_test_stats)

add_executable(kernellab_test_verification
  test_verification.cpp)

target_compile_features(kernellab_test_verification PRIVATE cxx_std_17)
target_include_directories(kernellab_test_verification PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernellab_test_verification PRIVATE kernellab_core)

add_test(
  NAME verification
  COMMAND kernellab_test_verification)

add_executable(kernellab_test_cpu_backends
  test_cpu_backends.cpp)

target_compile_features(kernellab_test_cpu_backends PRIVATE cxx_std_17)
target_include_directories(kernellab_test_cpu_backends PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernellab_test_cpu_backends PRIVATE kernellab_core)

add_test(
  NAME cpu_backends
  COMMAND kernellab_test_cpu_backends)

add_executable(kernellab_test_executor
  test_executor.cpp)

target_compile_features(kernellab_test_executor PRIVATE cxx_std_17)
target_include_directories(kernellab_test_executor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernellab_test_executor PRIVATE kernellab_core)

add_test(
  NAME executor
  COMMAND kernellab_test_executor)

add_executable(kernellab_test_export
  test_export.cpp)

target_compile_features(kernellab_test_export PRIVATE cxx_std_17)
target_include_directories(kernellab_test_export PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernellab_test_export PRIVATE kernellab_core)

add_test(
  NAME export
  COMMAND kernellab_test_export)

add_executable(kernellab_test_cli_smoke
  test_cli_smoke.cpp)

target_compile_features(kernellab_test_cli_smoke PRIVATE cxx_std_17)
target_include_directories(kernellab_test_cli_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernellab_test_cli_smoke PRIVATE kernellab_core)

add_test(
  NAME cli_smoke
  COMMAND kernellab_test_cli_smoke)
