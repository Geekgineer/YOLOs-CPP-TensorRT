#!/bin/bash
# ============================================================================
# YOLOs-TRT Test Build Script
# ============================================================================
# Builds the test suite against system-installed TensorRT and CUDA.
#
# Usage:
#   ./build_test.sh [TEST_TASK]
#
# TEST_TASK:
#   0 = Detection, 1 = Classification, 2 = Segmentation,
#   3 = Pose, 4 = OBB, 5 = All (default), 6 = Depth
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Default values
TEST_TASK="${1:-5}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================================================
# Usage
# ============================================================================
usage() {
    echo "Usage: $0 [TEST_TASK]"
    echo ""
    echo "Arguments:"
    echo "  TEST_TASK   Task to test:"
    echo "                0 = Detection"
    echo "                1 = Classification"
    echo "                2 = Segmentation"
    echo "                3 = Pose"
    echo "                4 = OBB"
    echo "                5 = All tasks (default)"
    echo "                6 = Depth estimation"
    echo ""
    echo "Prerequisites:"
    echo "  - CUDA Toolkit >= 12.0"
    echo "  - TensorRT >= 10.0"
    echo "  - OpenCV >= 4.5"
    echo "  - CMake >= 3.18"
    echo ""
    echo "Examples:"
    echo "  $0            # Build all tests"
    echo "  $0 0          # Build detection tests only"
    echo "  $0 1          # Build classification tests only"
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# ============================================================================
# Check Prerequisites
# ============================================================================
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    if ! command -v cmake &>/dev/null; then
        echo -e "${RED}ERROR: cmake not found${NC}"
        exit 1
    fi

    if ! command -v nvcc &>/dev/null; then
        echo -e "${RED}ERROR: nvcc not found. Install CUDA Toolkit.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Prerequisites OK${NC}"
}

# ============================================================================
# Build Project
# ============================================================================
build_project() {
    local build_dir="${SCRIPT_DIR}/build"

    mkdir -p "$build_dir"
    cd "$build_dir"

    echo -e "${YELLOW}Configuring CMake...${NC}"
    cmake .. \
        -DtestTask="${TEST_TASK}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3"

    echo -e "${YELLOW}Building...${NC}"
    cmake --build . -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    echo -e "${GREEN}Build completed successfully!${NC}"
}

# ============================================================================
# Main
# ============================================================================
echo "============================================"
echo "  YOLOs-TRT Test Build"
echo "============================================"
echo "  Task:    $TEST_TASK"
echo "  Backend: TensorRT + CUDA"
echo "============================================"

check_prerequisites
build_project

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo "Run tests with: cd build && ctest --output-on-failure"
