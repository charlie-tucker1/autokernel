cmake_minimum_required(VERSION 3.24)

# nvcc 13 refuses Fedora's gcc 16 as a host compiler. The `laptop` preset in
# CMakePresets.json sets the compiler explicitly; this fallback covers a bare
# `cmake -S . -B build` and CLion's Default profile. Must run before project().
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    foreach(candidate
            "$ENV{HOME}/miniforge3/envs/cuda-gcc15/bin/x86_64-conda-linux-gnu-g++"
            /usr/bin/g++-15)
        if(EXISTS "${candidate}")
            set(CMAKE_CUDA_HOST_COMPILER "${candidate}")
            break()
        endif()
    endforeach()
endif()

project(autokernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_subdirectory(harness)
