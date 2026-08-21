# ============================================================================
# YOLOs-TRT Test Suite
# ============================================================================
cmake_minimum_required(VERSION 3.18)
project(yolos_trt_tests VERSION 3.0.0 LANGUAGES CXX CUDA)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# Base Path Definitions
# ============================================================================
add_compile_definitions("BASE_PATH=${CMAKE_SOURCE_DIR}/")
add_compile_definitions("BASE_PATH_DETECTION=${CMAKE_SOURCE_DIR}/detection/")
add_compile_definitions("BASE_PATH_CLASSIFICATION=${CMAKE_SOURCE_DIR}/classification/")
add_compile_definitions("BASE_PATH_SEGMENTATION=${CMAKE_SOURCE_DIR}/segmentation/")
add_compile_definitions("BASE_PATH_POSE=${CMAKE_SOURCE_DIR}/pose/")
add_compile_definitions("BASE_PATH_OBB=${CMAKE_SOURCE_DIR}/obb/")
add_compile_definitions("BASE_PATH_DEPTH=${CMAKE_SOURCE_DIR}/depth/")

message(STATUS "Test Base Path: ${CMAKE_SOURCE_DIR}")

# ============================================================================
# CUDA
# ============================================================================
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

# ============================================================================
# TensorRT
# ============================================================================
if(NOT DEFINED TENSORRT_DIR)
    set(TENSORRT_SEARCH_PATHS
        "/usr/lib/x86_64-linux-gnu"
        "/usr/lib/aarch64-linux-gnu"
        "/usr/local/tensorrt"
        "$ENV{TENSORRT_DIR}"
    )
    foreach(path ${TENSORRT_SEARCH_PATHS})
        if(EXISTS "${path}/libnvinfer.so" OR EXISTS "${path}/lib/libnvinfer.so")
            set(TENSORRT_DIR "${path}")
            break()
        endif()
    endforeach()
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR}/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include
)
find_library(NVINFER_LIB nvinfer
    HINTS ${TENSORRT_DIR}/lib ${TENSORRT_DIR} /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
)

if(NOT NVINFER_LIB)
    message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_DIR=/path/to/tensorrt")
endif()

message(STATUS "TensorRT: ${NVINFER_LIB}")

# ============================================================================
# Dependencies
# ============================================================================
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# nlohmann_json
include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(json)

# GoogleTest
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Threading
find_package(Threads REQUIRED)

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    "${CMAKE_SOURCE_DIR}/../include"
    "${TENSORRT_INCLUDE_DIR}"
)

# ============================================================================
# CUDA Preprocessing Library
# ============================================================================
add_library(cuda_preprocessing_test STATIC
    "${CMAKE_SOURCE_DIR}/../include/yolos/core/cuda_preprocessing.cu"
)
target_include_directories(cuda_preprocessing_test PRIVATE
    "${CMAKE_SOURCE_DIR}/../include"
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(cuda_preprocessing_test PRIVATE CUDA::cudart)
set_target_properties(cuda_preprocessing_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Test Task Selection
# ============================================================================
option(testTask "Specify test task: 0=detection, 1=classification, 2=segmentation, 3=pose, 4=obb, 5=all, 6=depth" STRING)
message(STATUS "Selected test task: ${testTask}")

set(detection_path "${CMAKE_SOURCE_DIR}/detection")
set(classification_path "${CMAKE_SOURCE_DIR}/classification")
set(segmentation_path "${CMAKE_SOURCE_DIR}/segmentation")
set(pose_path "${CMAKE_SOURCE_DIR}/pose")
set(obb_path "${CMAKE_SOURCE_DIR}/obb")
set(depth_path "${CMAKE_SOURCE_DIR}/depth")

set(INFERENCE_EXES)
set(TEST_EXES)

# ============================================================================
# Detection Tests
# ============================================================================
if(testTask STREQUAL "0" OR testTask STREQUAL "5")
    add_executable(inference_detection_cpp
        ${detection_path}/inference_detection_cpp.cpp)
    add_executable(compare_detection_results
        ${detection_path}/compare_results.cpp)

    list(APPEND INFERENCE_EXES inference_detection_cpp)
    list(APPEND TEST_EXES compare_detection_results)
endif()

# ============================================================================
# Classification Tests
# ============================================================================
if(testTask STREQUAL "1" OR testTask STREQUAL "5")
    add_executable(inference_classification_cpp
        ${classification_path}/inference_classification_cpp.cpp)
    add_executable(compare_classification_results
        ${classification_path}/compare_results.cpp)

    list(APPEND INFERENCE_EXES inference_classification_cpp)
    list(APPEND TEST_EXES compare_classification_results)
endif()

# ============================================================================
# Segmentation Tests
# ============================================================================
if(testTask STREQUAL "2" OR testTask STREQUAL "5")
    add_executable(inference_segmentation_cpp
        ${segmentation_path}/inference_segmentation_cpp.cpp)
    add_executable(compare_segmentation_results
        ${segmentation_path}/compare_results.cpp)

    list(APPEND INFERENCE_EXES inference_segmentation_cpp)
    list(APPEND TEST_EXES compare_segmentation_results)
endif()

# ============================================================================
# Pose Tests
# ============================================================================
if(testTask STREQUAL "3" OR testTask STREQUAL "5")
    add_executable(inference_pose_cpp
        ${pose_path}/inference_pose_cpp.cpp)
    add_executable(compare_pose_results
        ${pose_path}/compare_results.cpp)

    list(APPEND INFERENCE_EXES inference_pose_cpp)
    list(APPEND TEST_EXES compare_pose_results)
endif()

# ============================================================================
# OBB Tests
# ============================================================================
if(testTask STREQUAL "4" OR testTask STREQUAL "5")
    add_executable(inference_obb_cpp
        ${obb_path}/inference_obb_cpp.cpp)
    add_executable(compare_obb_results
        ${obb_path}/compare_obb_results.cpp)

    list(APPEND INFERENCE_EXES inference_obb_cpp)
    list(APPEND TEST_EXES compare_obb_results)
endif()

# ============================================================================
# Depth Estimation Tests
# ============================================================================
if(testTask STREQUAL "6" OR testTask STREQUAL "5")
    add_executable(inference_depth_cpp
        ${depth_path}/inference_depth_cpp.cpp)
    add_executable(compare_depth_results
        ${depth_path}/compare_results.cpp)

    list(APPEND INFERENCE_EXES inference_depth_cpp)
    list(APPEND TEST_EXES compare_depth_results)
endif()

# ============================================================================
# Configure Inference Executables
# ============================================================================
foreach(target ${INFERENCE_EXES})
    message(STATUS "Configuring inference target: ${target}")

    target_include_directories(${target} PRIVATE
        "${CMAKE_SOURCE_DIR}/../include"
        "${TENSORRT_INCLUDE_DIR}"
    )

    target_link_libraries(${target} PRIVATE
        ${OpenCV_LIBS}
        ${NVINFER_LIB}
        CUDA::cudart
        cuda_preprocessing_test
        nlohmann_json::nlohmann_json
        Threads::Threads
    )

    if(UNIX)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
        )
    endif()
endforeach()

# ============================================================================
# Configure Test Executables
# ============================================================================
foreach(target ${TEST_EXES})
    message(STATUS "Adding test target: ${target}")

    target_link_libraries(${target} PRIVATE
        nlohmann_json::nlohmann_json
        GTest::gtest_main
        ${OpenCV_LIBS}
    )

    gtest_discover_tests(${target})
endforeach()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== Test Suite Configuration ===")
message(STATUS "  Inference executables: ${INFERENCE_EXES}")
message(STATUS "  Test executables: ${TEST_EXES}")
message(STATUS "")
