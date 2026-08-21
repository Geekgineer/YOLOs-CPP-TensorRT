# ============================================================================
# YOLOs-TRT - High-Performance YOLO Inference with TensorRT
# ============================================================================
cmake_minimum_required(VERSION 3.18)
# ============================================================================
# CUDA Architectures — MUST be chosen BEFORE project()
# ============================================================================
# project(... LANGUAGES CUDA) initializes CMAKE_CUDA_ARCHITECTURES from the
# compiler default (sm_52 on CUDA 12.x) whenever it is not already set. That
# made the `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` block further down dead
# code, so any configure that did not pass -DCMAKE_CUDA_ARCHITECTURES silently
# built the CUDA letterbox preprocessing kernel for sm_52. It still JITs on a
# modern GPU, but returns WRONG pixel data — which silently corrupts the
# results of every task. Deciding here, before project(), is what actually
# takes effect.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64" AND
       (EXISTS "/etc/nv_tegra_release" OR EXISTS "/usr/lib/aarch64-linux-gnu/tegra"))
        set(CMAKE_CUDA_ARCHITECTURES "72;87")          # Jetson Xavier / Orin
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90") # Turing -> Hopper
    endif()
endif()

project(YOLOs-TRT VERSION 3.0.0 LANGUAGES CXX CUDA)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# CUDA Standard
# ============================================================================
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# Build Options
# ============================================================================
option(BUILD_EXAMPLES "Build example applications" OFF)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)

# ============================================================================
# Platform Detection — Jetson vs x86
# ============================================================================
set(IS_JETSON OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    if(EXISTS "/etc/nv_tegra_release" OR EXISTS "/usr/lib/aarch64-linux-gnu/tegra")
        set(IS_JETSON ON)
        add_compile_definitions(JETSON_PLATFORM)
        message(STATUS "Platform: NVIDIA Jetson (aarch64)")
    else()
        message(STATUS "Platform: aarch64 (non-Jetson)")
    endif()
else()
    message(STATUS "Platform: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# ============================================================================
# CUDA
# ============================================================================
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")

# TensorRT 10 requires Turing (sm_75) or newer, so the nvcc default of sm_52
# is never a legitimate target here — catch a regression of the above.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    message(FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES is 52 (the nvcc default). The CUDA "
        "preprocessing kernel returns incorrect data when built for sm_52 and "
        "run on a modern GPU. Pass -DCMAKE_CUDA_ARCHITECTURES=<arch> explicitly.")
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================================
# TensorRT
# ============================================================================
if(NOT DEFINED TENSORRT_DIR)
    # Try common TensorRT locations
    set(TENSORRT_SEARCH_PATHS
        "/usr/lib/x86_64-linux-gnu"          # Ubuntu x86 system install
        "/usr/lib/aarch64-linux-gnu"          # Jetson / aarch64 system
        "/usr/local/tensorrt"                 # Manual install
        "$ENV{TENSORRT_DIR}"                  # Environment variable
        "${CMAKE_SOURCE_DIR}/TensorRT"        # Local copy
    )
    foreach(path ${TENSORRT_SEARCH_PATHS})
        if(EXISTS "${path}/libnvinfer.so" OR EXISTS "${path}/lib/libnvinfer.so")
            set(TENSORRT_DIR "${path}")
            break()
        endif()
    endforeach()
endif()

# Find TensorRT headers
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR}/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include
)

# Find TensorRT libraries
find_library(NVINFER_LIB nvinfer
    HINTS ${TENSORRT_DIR}/lib ${TENSORRT_DIR} /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
)
find_library(NVONNXPARSER_LIB nvonnxparser
    HINTS ${TENSORRT_DIR}/lib ${TENSORRT_DIR} /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
)

if(NOT NVINFER_LIB)
    message(FATAL_ERROR
        "TensorRT not found!\n"
        "Please install TensorRT or set TENSORRT_DIR:\n"
        "  cmake .. -DTENSORRT_DIR=/path/to/tensorrt\n"
        "For Jetson, TensorRT is typically pre-installed via JetPack.\n"
        "For data-center GPUs, download from: https://developer.nvidia.com/tensorrt"
    )
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${NVINFER_LIB}")

# ============================================================================
# OpenCV
# ============================================================================
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    "${CMAKE_SOURCE_DIR}/include"
    "${TENSORRT_INCLUDE_DIR}"
)

# ============================================================================
# Compiler Flags
# ============================================================================
if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic)
    endif()
endif()

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(NOT MSVC)
        add_compile_options(-O3)
    endif()
endif()

# Threading support
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ============================================================================
# CUDA Preprocessing Library
# ============================================================================
add_library(cuda_preprocessing STATIC
    "${CMAKE_SOURCE_DIR}/include/yolos/core/cuda_preprocessing.cu"
)
target_include_directories(cuda_preprocessing PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(cuda_preprocessing PRIVATE
    CUDA::cudart
)
set_target_properties(cuda_preprocessing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Helper Function
# ============================================================================
function(add_yolo_executable name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE
        "${TENSORRT_INCLUDE_DIR}"
    )
    target_link_libraries(${name} PRIVATE
        ${OpenCV_LIBS}
        ${NVINFER_LIB}
        CUDA::cudart
        cuda_preprocessing
        Threads::Threads
    )
    # Optional: link nvonnxparser if available
    if(NVONNXPARSER_LIB)
        target_link_libraries(${name} PRIVATE ${NVONNXPARSER_LIB})
    endif()

    if(UNIX)
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
        )
    endif()
endfunction()

# ============================================================================
# Main Executables
# ============================================================================
add_yolo_executable(image_inference src/image_inference.cpp)
add_yolo_executable(video_inference src/video_inference.cpp)
add_yolo_executable(camera_inference src/camera_inference.cpp)
add_yolo_executable(class_image_inference src/class_image_inference.cpp)
add_yolo_executable(batch_image_inference src/batch_image_inference.cpp)
add_yolo_executable(depth_image_inference src/depth_image_inference.cpp)

# ============================================================================
# Examples (optional)
# ============================================================================
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== YOLOs-TRT Configuration ===")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Toolkit:      ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  TensorRT:          ${NVINFER_LIB}")
message(STATUS "  OpenCV:            ${OpenCV_VERSION}")
message(STATUS "  Jetson:            ${IS_JETSON}")
message(STATUS "  Examples:          ${BUILD_EXAMPLES}")
message(STATUS "")
