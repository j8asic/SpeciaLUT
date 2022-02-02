cmake_minimum_required(VERSION 3.19)

project(SpeciaLUT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA)

if (CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    enable_language(CUDA)
    if (CMAKE_CXX_COMPILER_ID STREQUAL Clang)
        add_compile_definitions(USE_CUDA __STRICT_ANSI__)
        add_compile_options (-x cuda --cuda-gpu-arch=sm_35 -Wno-unknown-cuda-version)
    elseif (WIN32) #if (CMAKE_CXX_COMPILER_ID STREQUAL NVIDIA)
        add_compile_options(-Xcompiler "/std:c++20")
    endif()
endif()

add_executable(SpeciaLUT_test specialut.hpp main.cpp)

if (CUDA_FOUND)
    target_include_directories(SpeciaLUT_test SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(SpeciaLUT_test ${CUDA_LIBRARIES})
endif()
