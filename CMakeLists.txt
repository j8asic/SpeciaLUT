cmake_minimum_required(VERSION 3.21)

project(SpeciaLUT LANGUAGES CXX)

# The library is the single header; either copy it, or use this target:
#   add_subdirectory(SpeciaLUT)
#   target_link_libraries(app PRIVATE SpeciaLUT::SpeciaLUT)
add_library(SpeciaLUT INTERFACE)
add_library(SpeciaLUT::SpeciaLUT ALIAS SpeciaLUT)
target_include_directories(SpeciaLUT INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(SpeciaLUT INTERFACE cxx_std_20)

# CUDA parts are built by default if a CUDA compiler is around (nvcc, or Clang
# through -DCMAKE_CUDA_COMPILER=clang++)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    set(SPECIALUT_CUDA_DEFAULT ON)
else()
    set(SPECIALUT_CUDA_DEFAULT OFF)
endif()

option(SPECIALUT_BUILD_EXAMPLES "Build the example and the benchmark" ${PROJECT_IS_TOP_LEVEL})
option(SPECIALUT_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(SPECIALUT_CUDA "Build the CUDA example and the CUDA tests" ${SPECIALUT_CUDA_DEFAULT})
option(SPECIALUT_NATIVE "Tune the benchmark for the CPU it is built on" ON)
option(SPECIALUT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(SPECIALUT_SANITIZE "Build the tests with address and undefined-behaviour sanitizers" OFF)

if (NOT SPECIALUT_BUILD_EXAMPLES AND NOT SPECIALUT_BUILD_TESTS)
    return()
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (SPECIALUT_CUDA)
    enable_language(CUDA)
    if (NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 20)
    endif()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Warnings for the host code of this repository
function(specialut_warnings target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /W4 $<$<BOOL:${SPECIALUT_WARNINGS_AS_ERRORS}>:/WX>)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
            $<$<BOOL:${SPECIALUT_WARNINGS_AS_ERRORS}>:-Werror>)
    endif()
endfunction()

if (SPECIALUT_BUILD_EXAMPLES)
    # Usage examples; with CUDA, the same file also launches kernels
    add_executable(SpeciaLUT_example main.cpp)
    target_link_libraries(SpeciaLUT_example PRIVATE SpeciaLUT::SpeciaLUT)
    if (SPECIALUT_CUDA)
        set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
        target_compile_definitions(SpeciaLUT_example PRIVATE USE_CUDA)
    else()
        specialut_warnings(SpeciaLUT_example)
    endif()

    # LUT dispatch vs run-time branching
    add_executable(SpeciaLUT_benchmark benchmark.cpp)
    target_link_libraries(SpeciaLUT_benchmark PRIVATE SpeciaLUT::SpeciaLUT)
    specialut_warnings(SpeciaLUT_benchmark)
    if (SPECIALUT_NATIVE AND NOT MSVC)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-march=native SPECIALUT_HAS_MARCH_NATIVE)
        if (SPECIALUT_HAS_MARCH_NATIVE)
            target_compile_options(SpeciaLUT_benchmark PRIVATE -march=native)
        endif()
    endif()

    # The same comparison for CUDA kernels
    if (SPECIALUT_CUDA)
        add_executable(SpeciaLUT_benchmark_cuda benchmark_cuda.cu)
        target_link_libraries(SpeciaLUT_benchmark_cuda PRIVATE SpeciaLUT::SpeciaLUT)
    endif()
endif()

if (SPECIALUT_BUILD_TESTS)
    enable_testing()

    add_executable(SpeciaLUT_tests tests/test.cpp)
    target_link_libraries(SpeciaLUT_tests PRIVATE SpeciaLUT::SpeciaLUT)
    specialut_warnings(SpeciaLUT_tests)
    add_test(NAME host COMMAND SpeciaLUT_tests)

    if (SPECIALUT_SANITIZE AND NOT MSVC)
        target_compile_options(SpeciaLUT_tests PRIVATE
            -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
        target_link_options(SpeciaLUT_tests PRIVATE -fsanitize=address,undefined)
        set_tests_properties(host PROPERTIES
            ENVIRONMENT "ASAN_OPTIONS=detect_stack_use_after_return=1")
    endif()

    # With exceptions disabled, a state outside of the table must abort
    if (NOT MSVC)
        add_executable(SpeciaLUT_tests_no_exceptions tests/test_no_exceptions.cpp)
        target_link_libraries(SpeciaLUT_tests_no_exceptions PRIVATE SpeciaLUT::SpeciaLUT)
        target_compile_options(SpeciaLUT_tests_no_exceptions PRIVATE -fno-exceptions)
        specialut_warnings(SpeciaLUT_tests_no_exceptions)
        add_test(NAME no_exceptions COMMAND SpeciaLUT_tests_no_exceptions)
    endif()

    if (SPECIALUT_BUILD_EXAMPLES)
        add_test(NAME example COMMAND SpeciaLUT_example)
    endif()

    if (SPECIALUT_CUDA)
        add_executable(SpeciaLUT_tests_cuda tests/test_cuda.cu)
        target_link_libraries(SpeciaLUT_tests_cuda PRIVATE SpeciaLUT::SpeciaLUT)
        add_test(NAME cuda COMMAND SpeciaLUT_tests_cuda)
        # the test returns 77 on a machine without a usable GPU
        set_tests_properties(cuda PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
