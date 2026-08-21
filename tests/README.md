# YOLOs-TRT Test Suite

Comprehensive test suite validating C++ TensorRT implementations against Python Ultralytics reference.

## Test Status

| Task | Tests | Models | Status |
|------|-------|--------|--------|
| Detection | 8/8 | YOLOv5, v6, v8, v9, v10, v11, v12, YOLO26 | Pass |
| Classification | 6/6 | YOLOv8, v11, YOLO26 | Pass |
| Pose | 7/7 | YOLOv8, v11, YOLO26 | Pass |
| Segmentation | 8/8 | YOLOv8, v11, YOLO26 | Pass |
| OBB | 7/7 | YOLOv8, v11, YOLO26 | Pass |
| Depth | 8/8 | YOLO26-depth | New |
| **Total** | **44/44** | | |

## Requirements

- **NVIDIA GPU** with Compute Capability >= 7.5
- **CUDA Toolkit** >= 12.0
- **TensorRT** >= 10.0
- **OpenCV** >= 4.5
- **CMake** >= 3.18
- **Python 3.10+** with `uv` package manager (auto-installed)

## Quick Start

```bash
# Run all tests (requires GPU)
./test_all.sh

# Run individual task tests
./test_detection.sh
./test_classification.sh
./test_pose.sh
./test_segmentation.sh
./test_obb.sh
./test_depth.sh
```

## How Tests Work

1. **Model Download**: Downloads pretrained `.pt` files from Ultralytics
2. **ONNX Export**: Exports models to ONNX format (opset 12)
3. **TensorRT Conversion**: Converts ONNX to `.trt` engines (FP16 by default)
4. **Python Inference**: Runs Ultralytics to generate ground truth
5. **C++ Build**: Builds C++ inference executables (linked against TensorRT + CUDA)
6. **C++ Inference**: Runs C++ TensorRT implementation
7. **Comparison**: Compares results using GoogleTest

## Directory Structure

```
tests/
├── test_utils.sh           # Shared utilities (uv, venv, TRT conversion)
├── test_all.sh             # Master test runner
├── test_detection.sh       # Detection task runner
├── test_classification.sh  # Classification task runner
├── test_segmentation.sh    # Segmentation task runner
├── test_pose.sh            # Pose estimation task runner
├── test_obb.sh             # OBB detection task runner
├── test_depth.sh           # Metric depth estimation task runner
├── build_test.sh           # CMake build script (TensorRT + CUDA)
├── CMakeLists.txt          # Test suite CMake config
│
├── detection/
│   ├── models/             # .pt, .onnx, and .trt models
│   ├── data/images/        # Test images
│   ├── results/            # JSON results
│   ├── inference_detection_cpp.cpp
│   ├── inference_detection_ultralytics.py
│   └── compare_results.cpp
│
├── classification/         # Similar structure
├── segmentation/           # Similar structure
├── pose/                   # Similar structure
├── obb/                    # Similar structure
└── depth/                  # Similar structure
```

## Tolerance Settings

The comparison tests use configurable error margins (TensorRT FP16 may introduce
additional numerical differences compared to PyTorch FP32):

| Metric | Tolerance | Description |
|--------|-----------|-------------|
| Confidence | +/-0.2 | Accounts for preprocessing + FP16 differences |
| Bounding Box | +/-50px | Pixel coordinate tolerance |
| Keypoints | +/-20px | Pose keypoint position tolerance |
| Mask Pixels | 20% | Segmentation mask difference |
| OBB Center | +/-50px | Oriented box center tolerance |
| OBB Angle | +/-0.2 rad | Rotation angle tolerance |
| Depth | 10% rel or +/-0.15 m | Metric depth, whichever is looser |
| Depth samples | 20% outliers | Pixels on a depth edge may fall either side |

## CI/CD Integration

The test scripts are designed for CI/CD pipelines:

- Uses `uv` for fast, reproducible Python environment
- Auto-converts ONNX models to TensorRT engines, preferring the bundled
  `onnx2trt` tool, then `trtexec`, then the Python converter
- Builds against system-installed TensorRT and CUDA (no downloads needed)
- Returns proper exit codes (0 = pass, non-zero = fail)

```yaml
# Example GitHub Actions (requires self-hosted GPU runner)
- name: Run YOLOs-TRT Tests
  run: |
    cd tests
    ./test_all.sh
```

## Notes

1. **GPU required**: All tests require an NVIDIA GPU with TensorRT
2. **Engine portability**: TRT engines are GPU-specific. Rebuild if switching hardware.
3. **Model size**: Uses smaller input (320x320) for faster testing; the
   depth suite uses 768x768, the resolution its weights were trained at
4. **YOLO26 models**: Feature end-to-end NMS-free architecture

## Troubleshooting

**nvcc not found:**
```bash
sudo apt install cuda-toolkit
```

**TensorRT not found:**
```bash
sudo apt install tensorrt
```

**trtexec not found:**

Expected on an apt-only install — `libnvinfer-dev` and `tensorrt-dev` do not
ship `trtexec`. The harness falls back to the bundled `onnx2trt`, so just build
the main project first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

If you do have the samples package or the tarball, `trtexec` lives here:

```bash
export PATH=$PATH:/usr/src/tensorrt/bin
```

**Results differ from the Python ground truth:**

Check the CUDA architecture the tests were built for:

```bash
grep CMAKE_CUDA_ARCHITECTURES tests/build/CMakeCache.txt
```

`52` means the CUDA preprocessing kernel was built for sm_52 and is returning
incorrect pixel data on a modern GPU. Reconfigure from a clean build directory.

**Python package issues:**
```bash
source ~/.yolos-trt-test-venv/bin/activate
uv pip install ultralytics onnx tqdm
```

**Model conversion fails:**
```bash
# Convert manually with trtexec
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```
