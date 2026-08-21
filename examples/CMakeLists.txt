cmake_minimum_required(VERSION 3.18)
project(yolo_trt_examples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories("../include/")

# ============================================================================
# CUDA preprocessing library
# ============================================================================
add_library(cuda_preprocessing_examples STATIC
    ../include/yolos/core/cuda_preprocessing.cu
)
target_include_directories(cuda_preprocessing_examples PUBLIC "../include/")
set_target_properties(cuda_preprocessing_examples PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# TensorRT discovery
# ============================================================================
if(DEFINED TENSORRT_DIR)
    set(TENSORRT_INCLUDE_DIR "${TENSORRT_DIR}/include")
    set(TENSORRT_LIB_DIR "${TENSORRT_DIR}/lib")
else()
    # System-installed TensorRT
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/include/aarch64-linux-gnu
    )
    find_library(TENSORRT_LIB nvinfer
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/lib/aarch64-linux-gnu
    )
    get_filename_component(TENSORRT_LIB_DIR "${TENSORRT_LIB}" DIRECTORY)
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib dir: ${TENSORRT_LIB_DIR}")

# ============================================================================
# IMAGE EXAMPLES
# ============================================================================
add_executable(example_image_det example_image_det.cpp)
add_executable(example_image_seg example_image_seg.cpp)
add_executable(example_image_obb example_image_obb.cpp)
add_executable(example_image_class example_image_class.cpp)
add_executable(example_image_pose example_image_pose.cpp)
add_executable(example_image_depth example_image_depth.cpp)
# ============================================================================
# VIDEO EXAMPLES
# ============================================================================
add_executable(example_video_det example_video_det.cpp)
add_executable(example_video_seg example_video_seg.cpp)
add_executable(example_video_obb example_video_obb.cpp)
add_executable(example_video_class example_video_class.cpp)
add_executable(example_video_pose example_video_pose.cpp)
# ============================================================================
# CAMERA EXAMPLES
# ============================================================================
add_executable(example_camera_det example_camera_det.cpp)
add_executable(example_camera_seg example_camera_seg.cpp)
add_executable(example_camera_obb example_camera_obb.cpp)
add_executable(example_camera_class example_camera_class.cpp)
add_executable(example_camera_pose example_camera_pose.cpp)

# ============================================================================
# Link all examples
# ============================================================================
set(ALL_EXAMPLES
    example_image_det example_image_seg example_image_obb example_image_class example_image_pose
    example_image_depth
    example_video_det example_video_seg example_video_obb example_video_class example_video_pose
    example_camera_det example_camera_seg example_camera_obb example_camera_class example_camera_pose
)

foreach(example ${ALL_EXAMPLES})
    target_include_directories(${example} PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(${example}
        ${OpenCV_LIBS}
        cuda_preprocessing_examples
        CUDA::cudart
        nvinfer
    )
    target_link_directories(${example} PRIVATE ${TENSORRT_LIB_DIR})
endforeach()
