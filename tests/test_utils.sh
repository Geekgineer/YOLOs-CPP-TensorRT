#!/bin/bash
# ============================================================================
# YOLOs-TRT Test Utilities
# Shared functions for all test scripts
# ============================================================================

# Colors for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export NC='\033[0m' # No Color

# Virtual environment directory
VENV_DIR="${TEST_VENV_DIR:-$HOME/.yolos-trt-test-venv}"

# ============================================================================
# Install uv (fast Python package installer)
# ============================================================================
install_uv() {
    if command -v uv &> /dev/null; then
        echo -e "${GREEN}uv is already installed${NC}"
        return 0
    fi

    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if command -v uv &> /dev/null; then
        echo -e "${GREEN}uv installed successfully${NC}"
        return 0
    else
        echo -e "${RED}Failed to install uv${NC}"
        return 1
    fi
}

# ============================================================================
# Setup virtual environment
# ============================================================================
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${BLUE}Creating virtual environment at $VENV_DIR${NC}"
        if command -v uv &> /dev/null; then
            uv venv "$VENV_DIR"
        else
            python3 -m venv "$VENV_DIR"
        fi
    fi

    # Activate the virtual environment
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}Virtual environment activated${NC}"
}

# ============================================================================
# Install Python packages using uv (with pip fallback)
# ============================================================================
install_python_packages() {
    local packages="$@"

    echo -e "${BLUE}Installing Python packages: $packages${NC}"

    # Ensure venv is set up and activated
    setup_venv

    # Try uv first (within the venv)
    if command -v uv &> /dev/null; then
        echo "Using uv..."
        uv pip install $packages --quiet && return 0
    fi

    # Fallback to pip (venv's pip)
    echo "Using pip..."
    pip install -q $packages && return 0

    echo -e "${RED}Failed to install packages${NC}"
    return 1
}

# ============================================================================
# Export PyTorch models to ONNX
# ============================================================================
export_models_to_onnx() {
    local models_dir="$1"
    local export_script="$2"

    cd "$models_dir" || return 1

    echo -e "${BLUE}Exporting models to ONNX...${NC}"

    # Ensure venv is activated
    setup_venv

    # Try the export script first
    if [ -f "$export_script" ]; then
        python3 "$export_script" cpu 2>&1 && return 0
    fi

    # Fallback: export each .pt file individually
    echo "Using individual export fallback..."
    for pt_file in *.pt; do
        if [ -f "$pt_file" ]; then
            onnx_file="${pt_file%.pt}.onnx"
            if [ ! -f "$onnx_file" ]; then
                echo "Exporting $pt_file -> $onnx_file"
                python3 -c "
from ultralytics import YOLO
model = YOLO('$pt_file')
model.export(format='onnx', opset=12, simplify=True, imgsz=320)
" 2>&1 || echo -e "${YELLOW}Warning: Failed to export $pt_file${NC}"
            else
                echo "Skipping $pt_file (ONNX already exists)"
            fi
        fi
    done

    # Verify we have ONNX models
    local onnx_count=$(ls -1 *.onnx 2>/dev/null | wc -l)
    if [ "$onnx_count" -eq 0 ]; then
        echo -e "${RED}ERROR: No ONNX models available${NC}"
        return 1
    fi

    echo -e "${GREEN}Found $onnx_count ONNX model(s)${NC}"
    return 0
}

# ============================================================================
# Convert ONNX models to TensorRT engines
# ============================================================================
convert_onnx_to_trt() {
    local models_dir="$1"
    local precision="${2:-fp16}"  # fp16 by default for tests

    cd "$models_dir" || return 1

    # Ensure trtexec is discoverable. Ubuntu's tensorrt apt package ships it under
    # /usr/src/tensorrt/bin/, which is not on the default PATH.
    if ! command -v trtexec &>/dev/null; then
        for trt_bin in /usr/src/tensorrt/bin /usr/local/tensorrt/bin /opt/tensorrt/bin; do
            if [ -x "$trt_bin/trtexec" ]; then
                export PATH="$trt_bin:$PATH"
                break
            fi
        done
    fi

    echo -e "${BLUE}Converting ONNX models to TensorRT engines...${NC}"

    local converted=0
    local failed=0

    for onnx_file in *.onnx; do
        if [ -f "$onnx_file" ]; then
            local trt_file="${onnx_file%.onnx}.trt"
            if [ -f "$trt_file" ]; then
                echo "Skipping $onnx_file (TRT engine already exists)"
                converted=$((converted + 1))
                continue
            fi

            echo -e "${YELLOW}Converting $onnx_file -> $trt_file${NC}"

            # Try trtexec first (fastest, no Python dependency)
            if command -v trtexec &>/dev/null; then
                local trtexec_args="--onnx=$onnx_file --saveEngine=$trt_file"
                if [ "$precision" = "fp16" ]; then
                    trtexec_args="$trtexec_args --fp16"
                elif [ "$precision" = "int8" ]; then
                    trtexec_args="$trtexec_args --int8"
                fi

                if trtexec $trtexec_args 2>&1 | tail -5; then
                    converted=$((converted + 1))
                    echo -e "${GREEN}Converted: $trt_file${NC}"
                    continue
                else
                    echo -e "${YELLOW}trtexec failed, trying Python converter...${NC}"
                fi
            fi

            # Fallback to Python converter. It does `import tensorrt`, which is
            # NOT provided by the libnvinfer-dev / tensorrt-dev apt packages —
            # only the Python wheel ships it. Install it on demand, pinned to the
            # system TensorRT version: engines are version-locked, so a wheel of a
            # different version produces engines the C++ runtime cannot
            # deserialize.
            if ! python3 -c "import tensorrt" &>/dev/null; then
                local sys_trt
                sys_trt=$(dpkg-query -W -f='${Version}' libnvinfer-dev 2>/dev/null \
                          | sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+).*/\1/')
                if [ -n "$sys_trt" ]; then
                    echo -e "${YELLOW}Installing tensorrt==${sys_trt} for the Python converter...${NC}"
                    pip install -q "tensorrt==${sys_trt}" 2>&1 | tail -3
                else
                    echo -e "${YELLOW}Installing tensorrt for the Python converter...${NC}"
                    pip install -q tensorrt 2>&1 | tail -3
                fi
            fi

            # Fallback to Python converter
            local converter_script=""
            if [ -f "../../trt-files/scripts/convert_to_tensorrt.py" ]; then
                converter_script="../../trt-files/scripts/convert_to_tensorrt.py"
            elif [ -f "../../../trt-files/scripts/convert_to_tensorrt.py" ]; then
                converter_script="../../../trt-files/scripts/convert_to_tensorrt.py"
            fi

            if [ -n "$converter_script" ]; then
                local py_args="--onnx $onnx_file --output $trt_file"
                if [ "$precision" = "fp16" ]; then
                    py_args="$py_args --fp16"
                fi

                if python3 "$converter_script" $py_args 2>&1; then
                    converted=$((converted + 1))
                    echo -e "${GREEN}Converted: $trt_file${NC}"
                else
                    echo -e "${RED}Failed to convert $onnx_file${NC}"
                    failed=$((failed + 1))
                fi
            else
                echo -e "${RED}No way to build a TensorRT engine for $onnx_file.${NC}"
                echo -e "${RED}  trtexec: not on PATH and not in /usr/src/tensorrt/bin${NC}"
                echo -e "${RED}  (apt's libnvinfer-dev/tensorrt-dev do NOT ship trtexec;${NC}"
                echo -e "${RED}   it comes with the samples package or the TensorRT tarball)${NC}"
                echo -e "${RED}  python converter: trt-files/scripts/convert_to_tensorrt.py not found${NC}"
                failed=$((failed + 1))
            fi
        fi
    done

    if [ "$converted" -eq 0 ]; then
        echo -e "${RED}ERROR: No TRT engines were produced${NC}"
        return 1
    fi

    echo -e "${GREEN}Converted $converted engine(s), $failed failure(s)${NC}"
    return 0
}

# ============================================================================
# Print section header
# ============================================================================
print_header() {
    local title="$1"
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  $title${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# ============================================================================
# Print success message
# ============================================================================
print_success() {
    echo -e "${GREEN}[PASS] $1${NC}"
}

# ============================================================================
# Print error message
# ============================================================================
print_error() {
    echo -e "${RED}[FAIL] $1${NC}"
}

# ============================================================================
# Download test images if missing
# ============================================================================
download_test_images() {
    local images_dir="$1"
    local test_type="$2"
    local original_dir="$(pwd)"

    mkdir -p "$images_dir"
    cd "$images_dir" || return 1

    # Remove any .REMOVED.git-id placeholder files
    rm -f *.REMOVED.git-id 2>/dev/null || true

    # Check if we have valid images
    local valid_images=$(find . -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) ! -name "*.REMOVED*" 2>/dev/null | wc -l)

    if [ "$valid_images" -gt 0 ]; then
        echo -e "${GREEN}Found $valid_images existing test image(s)${NC}"
        cd "$original_dir"
        return 0
    fi

    echo -e "${YELLOW}Downloading test images for $test_type...${NC}"

    case "$test_type" in
        "detection"|"segmentation")
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "bus.jpg" 2>/dev/null || true
            curl -sL "https://ultralytics.com/images/zidane.jpg" -o "zidane.jpg" 2>/dev/null || true
            ;;
        "pose")
            curl -sL "https://ultralytics.com/images/zidane.jpg" -o "test1.jpg" 2>/dev/null || true
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "test2.jpg" 2>/dev/null || true
            ;;
        "obb")
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "image.png" 2>/dev/null || true
            ;;
        "classification")
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "bus.jpg" 2>/dev/null || true
            ;;
        "depth")
            # One outdoor street scene and one closer subject give the depth
            # comparison both far-field and near-field pixels to check.
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "bus.jpg" 2>/dev/null || true
            curl -sL "https://ultralytics.com/images/zidane.jpg" -o "zidane.jpg" 2>/dev/null || true
            ;;
        *)
            curl -sL "https://ultralytics.com/images/bus.jpg" -o "test.jpg" 2>/dev/null || true
            ;;
    esac

    valid_images=$(find . -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) ! -name "*.REMOVED*" 2>/dev/null | wc -l)

    cd "$original_dir"

    if [ "$valid_images" -eq 0 ]; then
        echo -e "${RED}ERROR: Failed to download test images${NC}"
        return 1
    fi

    echo -e "${GREEN}Downloaded $valid_images test image(s)${NC}"
    return 0
}
