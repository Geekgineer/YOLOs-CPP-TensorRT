#!/bin/bash
# ============================================================================
# YOLOs-TRT Build Script
# ============================================================================
# Builds the project against system-installed TensorRT and CUDA.
# Optionally accepts a custom TENSORRT_DIR for non-standard installs.
#
# Usage:
#   ./build.sh                        # Release build (auto-detect TRT)
#   ./build.sh --tensorrt /opt/TRT    # Custom TensorRT location
#   ./build.sh --debug                # Debug build
#   ./build.sh --clean                # Clean before building
#   ./build.sh --arch "86;89"         # Override CUDA architectures
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ============================================================================
# Defaults
# ============================================================================
BUILD_TYPE="Release"
TENSORRT_DIR=""
CUDA_ARCHS=""
CLEAN=0
BUILD_DIR="${SCRIPT_DIR}/build"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# Parse Arguments
# ============================================================================
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --tensorrt DIR    Path to TensorRT installation (auto-detect if omitted)"
    echo "  --debug           Build in Debug mode (default: Release)"
    echo "  --clean           Remove build directory before building"
    echo "  --arch ARCHS      CUDA architectures, semicolon-separated (e.g. \"86;89;90\")"
    echo "  --jobs N          Parallel build jobs (default: nproc)"
    echo "  -h, --help        Show this help"
    echo ""
    echo "Prerequisites:"
    echo "  - CUDA Toolkit >= 12.0"
    echo "  - TensorRT >= 10.0 (installed via apt or manually)"
    echo "  - OpenCV >= 4.5"
    echo "  - CMake >= 3.18"
    echo ""
    echo "Examples:"
    echo "  $0                              # Auto-detect everything, Release build"
    echo "  $0 --clean                      # Clean build"
    echo "  $0 --tensorrt /usr/local/TensorRT-10.4  # Custom TRT path"
    echo "  $0 --arch \"86;89\" --debug       # Debug build for specific GPUs"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tensorrt)
            TENSORRT_DIR="$2"
            shift 2
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --arch)
            CUDA_ARCHS="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# ============================================================================
# Environment Check
# ============================================================================
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    # CMake
    if ! command -v cmake &>/dev/null; then
        echo -e "${RED}ERROR: cmake not found. Install with: sudo apt install cmake${NC}"
        exit 1
    fi
    local cmake_ver
    cmake_ver=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+')
    echo -e "  CMake:    ${GREEN}$(cmake --version | head -1)${NC}"

    # CUDA
    if ! command -v nvcc &>/dev/null; then
        echo -e "${RED}ERROR: nvcc not found. Install CUDA Toolkit: sudo apt install cuda-toolkit${NC}"
        exit 1
    fi
    echo -e "  CUDA:     ${GREEN}$(nvcc --version | tail -1)${NC}"

    # TensorRT headers
    local trt_header=""
    if [[ -n "$TENSORRT_DIR" ]]; then
        trt_header="$TENSORRT_DIR/include/NvInfer.h"
    elif [[ -f /usr/include/x86_64-linux-gnu/NvInfer.h ]]; then
        trt_header="/usr/include/x86_64-linux-gnu/NvInfer.h"
    elif [[ -f /usr/include/NvInfer.h ]]; then
        trt_header="/usr/include/NvInfer.h"
    elif [[ -f /usr/include/aarch64-linux-gnu/NvInfer.h ]]; then
        trt_header="/usr/include/aarch64-linux-gnu/NvInfer.h"
    fi

    if [[ -z "$trt_header" || ! -f "$trt_header" ]]; then
        echo -e "${RED}ERROR: TensorRT headers not found.${NC}"
        echo "  Install via: sudo apt install tensorrt"
        echo "  Or specify: $0 --tensorrt /path/to/TensorRT"
        exit 1
    fi
    local trt_ver
    trt_ver=$(grep -oP 'NV_TENSORRT_MAJOR\s+\K[0-9]+' "$trt_header" 2>/dev/null || echo "?")
    echo -e "  TensorRT: ${GREEN}version ${trt_ver}.x (${trt_header})${NC}"

    # OpenCV
    if ! pkg-config --exists opencv4 2>/dev/null; then
        echo -e "${YELLOW}WARNING: pkg-config cannot find opencv4. CMake may still find it.${NC}"
    else
        echo -e "  OpenCV:   ${GREEN}$(pkg-config --modversion opencv4)${NC}"
    fi

    echo ""
}

# ============================================================================
# Build
# ============================================================================
build_project() {
    if [[ $CLEAN -eq 1 && -d "$BUILD_DIR" ]]; then
        echo -e "${YELLOW}Cleaning build directory...${NC}"
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    echo -e "${BLUE}Configuring CMake (${BUILD_TYPE})...${NC}"

    local cmake_args=(
        ".."
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    )

    if [[ -n "$TENSORRT_DIR" ]]; then
        cmake_args+=("-DTENSORRT_DIR=${TENSORRT_DIR}")
    fi

    if [[ -n "$CUDA_ARCHS" ]]; then
        cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}")
    fi

    if [[ "$BUILD_TYPE" == "Release" ]]; then
        cmake_args+=("-DCMAKE_CXX_FLAGS_RELEASE=-O3 -march=native")
    fi

    cmake "${cmake_args[@]}"

    echo ""
    echo -e "${BLUE}Building with ${JOBS} parallel jobs...${NC}"
    cmake --build . -- -j"${JOBS}"
}

# ============================================================================
# Main
# ============================================================================
echo "============================================"
echo "  YOLOs-TRT Build"
echo "============================================"
echo "  Build Type:  $BUILD_TYPE"
echo "  Build Dir:   $BUILD_DIR"
echo "  TensorRT:    ${TENSORRT_DIR:-auto-detect}"
echo "  CUDA Archs:  ${CUDA_ARCHS:-75;80;86;89;90 (default, not auto-detected)}"
echo "  Jobs:        $JOBS"
echo "============================================"
echo ""

check_prerequisites
build_project

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "Executables in ${BUILD_DIR}/:"
ls -1 "${BUILD_DIR}/"*_inference 2>/dev/null || true
echo ""
echo "Quick test:"
echo "  cd build && ./image_inference ../models/yolo11n.trt ../data/dog.jpg ../models/coco.names"
