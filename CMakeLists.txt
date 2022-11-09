cmake_minimum_required(VERSION 3.19)

project(SpeciaLUT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit)

if (DEFINED CUDAToolkit_VERSION)
    message(STATUS "Found CUDA ${CUDAToolkit_VERSION} at ${CUDA_TOOLKIT_ROOT_DIR}")
    if (CMAKE_CXX_COMPILER_ID STREQUAL Clang)
        add_compile_definitions(USE_CUDA __STRICT_ANSI__)
        add_compile_options (-x cuda --cuda-gpu-arch=sm_35 -Wno-unknown-cuda-version)
    elseif (WIN32) #if (CMAKE_CXX_COMPILER_ID STREQUAL NVIDIA)
        add_compile_options(-Xcompiler "/std:c++20")
    endif()
endif()

add_executable(SpeciaLUT_test specialut.hpp main.cpp)

if (DEFINED CUDAToolkit_VERSION)
    target_include_directories(SpeciaLUT_test SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(SpeciaLUT_test PRIVATE CUDA::cudart_static)
endif()
