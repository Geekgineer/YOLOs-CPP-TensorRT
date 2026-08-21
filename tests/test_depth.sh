#!/bin/bash
# ============================================================================
# YOLOs-TRT Depth Estimation Test Runner
# ============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/test_utils.sh"

print_header "YOLOs-TRT Depth Estimation Test"

# ============================================================================
# Setup
# ============================================================================
cd "$SCRIPT_DIR/depth"
echo "Working directory: $(pwd)"

print_header "Checking Test Images"
download_test_images "$(pwd)/data/images" "depth"

print_header "Installing Dependencies"
install_uv
install_python_packages ultralytics onnx tqdm

# ============================================================================
# Download and Export Models
# ============================================================================
print_header "Preparing Models"
cd models

if [ ! -f "yolo26n-depth.pt" ]; then
    echo "Downloading YOLO26 depth model from Ultralytics..."
    python3 -c "
from ultralytics import YOLO
YOLO('yolo26n-depth.pt')
"
fi

# Released YOLO26 depth weights are trained at imgsz=768; export at that size so
# the engine geometry matches the Ultralytics ground truth.
for pt_file in *.pt; do
    [ -f "$pt_file" ] || continue
    onnx_file="${pt_file%.pt}.onnx"
    if [ -f "$onnx_file" ]; then
        echo "Skipping $pt_file (ONNX already exists)"
        continue
    fi
    echo "Exporting $pt_file -> $onnx_file (imgsz=768)"
    python3 -c "
from ultralytics import YOLO
YOLO('$pt_file').export(format='onnx', opset=12, simplify=True, imgsz=768)
"
done

if [ -z "$(ls -1 *.onnx 2>/dev/null)" ]; then
    print_error "No ONNX depth models available"
    exit 1
fi

# ============================================================================
# Convert ONNX to TensorRT Engines
# ============================================================================
print_header "Converting to TensorRT"
convert_onnx_to_trt "$(pwd)" "fp16"

# ============================================================================
# Generate Python Ground Truth
# ============================================================================
print_header "Generating Python Ground Truth"
cd "$SCRIPT_DIR/depth"
echo "Running Ultralytics depth inference..."
python3 inference_depth_ultralytics.py || {
    print_error "Failed to generate Python ground truth"
    exit 1
}
print_success "Python ground truth generated"

# ============================================================================
# Build and Run Tests
# ============================================================================
print_header "Building Test Suite"
cd "$SCRIPT_DIR"
./build_test.sh 6

print_header "Running C++ Inference"
cd build
./inference_depth_cpp

print_header "Running Comparison Tests"
./compare_depth_results

print_success "Depth estimation tests completed!"
