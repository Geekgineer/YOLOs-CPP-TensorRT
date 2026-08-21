<div align="center">

<img src="data/cover.png" alt="YOLOs-TRT" width="100%">

<br/>

**The fastest way to run YOLO models in C++ on NVIDIA GPUs.**

Header-only. TensorRT-native. Zero-copy GPU pipeline. Sub-2ms inference.

<br/>

[![CI/CD](https://github.com/Geekgineer/YOLOs-CPP-TensorRT/actions/workflows/main.yml/badge.svg)](https://github.com/Geekgineer/YOLOs-CPP-TensorRT/actions)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![TensorRT](https://img.shields.io/badge/TensorRT-%E2%89%A5%2010.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-%E2%89%A5%2012.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)
[![Stars](https://img.shields.io/github/stars/Geekgineer/YOLOs-CPP-TensorRT?style=social)](https://github.com/Geekgineer/YOLOs-CPP-TensorRT/stargazers)

<br/>

[**Quick Start**](#-quick-start) · [**Benchmarks**](#-benchmarks) · [**Installation**](#-installation) · [**API**](#-api-reference) · [**Docker**](#-docker) · [**Docs**](doc/)

</div>

---

## Why YOLOs-TRT?

Most YOLO C++ wrappers treat preprocessing as an afterthought — resizing on the CPU, copying synchronously, rebuilding launch parameters every frame. **YOLOs-TRT** was built from scratch around a single principle: **the GPU should never wait for the CPU**.

| | YOLOs-TRT | Typical C++ YOLO Wrappers |
|---|:---:|:---:|
| Preprocessing | **GPU** (single CUDA kernel) | CPU (OpenCV) |
| Host-to-device transfer | **Async** (pinned memory) | Synchronous |
| Inference dispatch | **CUDA Graph replay** | Per-frame `enqueue` |
| YOLO version config | **Auto-detected** from tensor shape | Manual flag |
| API surface | **1 header include** | Multiple source files to compile |

The result: **sub-2ms end-to-end latency** and **530+ FPS** on a laptop GPU.

---

## ⚡ Benchmarks

<div align="center">

Measured on **NVIDIA RTX 2000 Ada (Laptop)** — YOLOv11n · 640×640 · 1000 iterations · 10-iter warm-up

| Precision | FPS | Avg Latency | P50 | P99 | GPU Memory |
|:---------:|:---:|:-----------:|:---:|:---:|:----------:|
| **FP32** | **466** | 2.14 ms | 2.04 ms | 3.03 ms | 530 MB |
| **FP16** | **479** | 2.09 ms | 1.98 ms | 2.91 ms | 536 MB |
| **INT8** | **530** | 1.89 ms | 1.78 ms | 2.70 ms | 444 MB |

</div>

> **Note**: These numbers include the full pipeline — preprocessing, inference, and postprocessing. Scaling is roughly linear on higher-end GPUs (RTX 4090, A100, H100).

<details>
<summary><b>What makes it fast?</b></summary>

<br/>

| Optimization | Impact |
|---|---|
| **GPU letterbox + normalize** | A single CUDA kernel performs bilinear letterbox resize, BGR→RGB conversion, and `/255.0` normalization — writing directly into the TRT input buffer. Eliminates all CPU preprocessing from the hot path. |
| **Pinned staging buffers** | Raw BGR pixels are `memcpy`'d into a CUDA pinned buffer, enabling truly asynchronous `cudaMemcpyAsync` H2D transfer that overlaps with compute. |
| **CUDA Graph capture** | For fixed-shape engines, the entire `enqueueV3` call graph is captured once and replayed via `cudaGraphLaunch`, cutting ~0.1–0.3 ms of per-frame dispatch overhead. |
| **10-iteration warm-up** | Lets TensorRT's internal autotuner converge on optimal kernel selections before timing begins. |
| **Single-stream pipeline** | Minimal synchronization points — one `cudaStream_t` drives the entire preprocess → infer → postprocess pipeline. |

</details>

---

## 🚀 Quick Start

**3 commands to your first inference:**

```bash
# 1. Clone & build
git clone https://github.com/Geekgineer/YOLOs-CPP-TensorRT.git
cd YOLOs-CPP-TensorRT && ./build.sh

# 2. Convert a model (requires Python + ultralytics)
pip install -r requirements.txt
python models/export_onnx.py --model yolo11n
./build/onnx2trt models/yolo11n.onnx models/yolo11n.trt --fp16

# 3. Run inference
./build/image_inference models/yolo11n.trt data/dog.jpg models/coco.names
```

**In your own code — 5 lines:**

```cpp
#include "yolos/tasks/detection.hpp"

int main() {
    yolos::det::YOLODetector detector("yolo11n.trt", "coco.names");
    cv::Mat image = cv::imread("dog.jpg");
    auto results = detector.detect(image);
    detector.drawDetections(image, results);
    cv::imshow("YOLOs-TRT", image);
    cv::waitKey(0);
}
```

---

## 🧩 Supported Models & Tasks

YOLOs-TRT auto-detects the YOLO version from output tensor shapes — **no manual configuration required**.

<div align="center">

| Task | API | Supported Versions |
|:-----|:----|:-------------------|
| **Detection** | `YOLODetector::detect()` | YOLOv5 · v7 · v8 · v9 · v10 · v11 · v12 · v26 · NAS |
| **Instance Segmentation** | `YOLOSegDetector::segment()` | YOLOv8-seg · v11-seg · v26-seg |
| **Pose Estimation** | `YOLOPoseDetector::detect()` | YOLOv8-pose · v11-pose · v26-pose |
| **Oriented BBox (OBB)** | `YOLOOBBDetector::detect()` | YOLOv8-obb · v11-obb · v26-obb |
| **Classification** | `YOLOClassifier::classify()` | YOLOv8-cls · v11-cls · v12-cls · v26-cls |
| **Metric Depth** | `YOLODepthEstimator::estimate()` | YOLO26-depth |

</div>

---

## 📦 Installation

### Prerequisites

| Dependency | Version | Notes |
|:-----------|:-------:|:------|
| NVIDIA GPU | CC ≥ 7.5 | Turing, Ampere, Ada, Hopper, or Jetson Xavier/Orin |
| CUDA Toolkit | ≥ 12.0 | |
| TensorRT | 10.x | Tensor-based API (`enqueueV3`). **Not** 11.x — see [installation](doc/installation.md) |
| OpenCV | ≥ 4.5 | Image I/O and visualization |
| CMake | ≥ 3.18 | CUDA language support |
| C++ compiler | C++17 | GCC 9+ / Clang 10+ |

### Build from Source

```bash
git clone https://github.com/Geekgineer/YOLOs-CPP-TensorRT.git
cd YOLOs-CPP-TensorRT
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Builds for `75;80;86;89;90` by default (`72;87` on Jetson). CUDA architectures
are **not** auto-detected — narrowing to your own GPU makes compilation much
faster:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89"
```

<details>
<summary><b>Custom TensorRT path / Jetson / Advanced options</b></summary>

```bash
# Custom TensorRT location
cmake .. -DCMAKE_BUILD_TYPE=Release -DTENSORRT_DIR=/opt/TensorRT-10.4

# Specific CUDA architectures (e.g. Jetson Orin)
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="87"

# Build example applications
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
```

**Jetson (JetPack 6.x):** TensorRT and CUDA are pre-installed. Just clone, build, and run.

</details>

### Install TensorRT

<details>
<summary><b>Ubuntu (apt)</b></summary>

```bash
sudo apt update && sudo apt install -y tensorrt
```

</details>

<details>
<summary><b>Jetson (JetPack)</b></summary>

Pre-installed. Verify with:
```bash
dpkg -l | grep tensorrt
```

</details>

<details>
<summary><b>Manual / Tarball</b></summary>

Download from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt), extract, and pass `-DTENSORRT_DIR=/path/to/tensorrt` to CMake.

</details>

---

## 🔄 Model Conversion

C++ inference requires serialized TensorRT engines. Python is only needed for this one-time conversion step.

```bash
# Install Python deps (uv recommended for speed)
pip install uv && uv pip install -r requirements.txt

# Export ONNX from Ultralytics
python models/export_onnx.py --model yolo11n

# Convert to TensorRT engine using the bundled tool
./build/onnx2trt models/yolo11n.onnx models/yolo11n.trt --fp16
```

> `trtexec` also works, but it is **not** shipped by the `libnvinfer-dev` /
> `tensorrt-dev` apt packages — only by the samples package or the tarball.
> `onnx2trt` is built from this repo and links the TensorRT libraries the
> project already requires, so it is always available.

<details>
<summary><b>INT8 quantization with calibration</b></summary>

```bash
# Generate calibration images (or use your own dataset)
python trt-files/scripts/generate_calibration_data.py

# Convert with INT8
python trt-files/scripts/convert_to_tensorrt.py \
    --onnx models/yolo11n.onnx --int8 \
    --calib-data trt-files/scripts/calibration_data/
```

See the [Quantization Guide](doc/quantization.md) for details on INT8 calibration strategies.

</details>

> **Important:** TensorRT engines are GPU-architecture and TRT-version specific. Always rebuild on the target hardware.

---

## 📖 API Reference

### Include & Use

YOLOs-TRT is header-only. Include the task header you need, link against TensorRT and CUDA, done.

```cpp
#include "yolos/tasks/detection.hpp"      // Detection
#include "yolos/tasks/segmentation.hpp"   // Instance Segmentation
#include "yolos/tasks/pose.hpp"           // Pose Estimation
#include "yolos/tasks/obb.hpp"            // Oriented Bounding Boxes
#include "yolos/tasks/classification.hpp" // Classification
#include "yolos/tasks/depth.hpp"          // Monocular metric depth
#include "yolos/yolos.hpp"                // Everything
```

### Constructor

All task classes share the same constructor pattern:

```cpp
ClassName(const std::string& enginePath,
          const std::string& labelsPath,
          YOLOVersion version = YOLOVersion::Auto,  // auto-detect from tensor shape
          int dlaCore = -1);                         // -1 = GPU, 0/1 = Jetson DLA core
```

### Task Examples

<details>
<summary><b>Detection</b></summary>

```cpp
yolos::det::YOLODetector detector("yolo11n.trt", "coco.names");
auto detections = detector.detect(image, 0.4f, 0.45f);
detector.drawDetections(image, detections);
```

</details>

<details>
<summary><b>Instance Segmentation</b></summary>

```cpp
yolos::seg::YOLOSegDetector seg("yolo11n-seg.trt", "coco.names");
auto results = seg.segment(image);
seg.drawSegmentations(image, results);
```

</details>

<details>
<summary><b>Pose Estimation</b></summary>

```cpp
yolos::pose::YOLOPoseDetector pose("yolo11n-pose.trt");
auto results = pose.detect(image);
pose.drawPoses(image, results);
```

</details>

<details>
<summary><b>Oriented Bounding Box (OBB)</b></summary>

```cpp
yolos::obb::YOLOOBBDetector obb("yolo11n-obb.trt", "Dota.names");
auto results = obb.detect(image);
obb.drawDetections(image, results);
```

</details>

<details>
<summary><b>Classification</b></summary>

```cpp
yolos::cls::YOLOClassifier cls("yolov8n-cls.trt", "ImageNet.names");
auto result = cls.classify(image);
std::cout << result.className << ": " << result.confidence * 100 << "%" << std::endl;
```

</details>

<details>
<summary><b>Monocular Metric Depth</b></summary>

```cpp
// No labels file — depth models predict a value, not a class.
yolos::depth::YOLODepthEstimator depth("yolo26n-depth.trt");

// CV_32FC1 at the original image size; each value is METERS.
cv::Mat depthMap = depth.estimate(image);

float centreMeters = yolos::depth::YOLODepthEstimator::depthAt(
    depthMap, depthMap.cols / 2, depthMap.rows / 2);

cv::imshow("depth", yolos::depth::colorizeDepth(depthMap));
cv::imshow("overlay", yolos::depth::overlayDepth(image, depthMap));
```

</details>

### Factory Functions

```cpp
// Auto-detect version from tensor shape
auto det = yolos::det::createDetector("yolo11n.trt", "coco.names");

// Explicit version override
auto det = yolos::det::createDetector("yolo11n.trt", "coco.names", yolos::YOLOVersion::V11);
```

---

## 🐳 Docker

### Data-Center GPU

```bash
docker build -f Dockerfile.tensorrt -t yolos-trt .
docker run --gpus all \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    yolos-trt \
    ./image_inference ./models/yolo11n.trt ./data/dog.jpg ./models/coco.names
```

### NVIDIA Jetson (JetPack 6.x)

```bash
docker build -f Dockerfile.tensorrt.jetson -t yolos-trt-jetson .
docker run --runtime nvidia \
    -v $(pwd)/models:/app/models \
    yolos-trt-jetson \
    ./image_inference ./models/yolo11n.trt ./data/dog.jpg ./models/coco.names
```

---

## 🏗️ Architecture

```
cv::Mat (BGR, host)
   │
   │  memcpy → pinned staging buffer
   ▼
Pinned Host ──cudaMemcpyAsync──► Device uint8 (raw BGR)
                                      │
                          ┌───────────┘
                          ▼
              letterboxNormalizeKernel()          ← single CUDA kernel
              ┌─────────────────────────┐
              │  bilinear letterbox      │
              │  BGR → RGB               │
              │  ÷ 255.0 normalize       │
              │  HWC → NCHW transpose    │
              └────────────┬────────────┘
                           ▼
              TRT input buffer (float32, device)
                           │
                  enqueueV3() / cudaGraphLaunch()
                           │
                           ▼
              TRT output buffer(s) (device)
                           │
                  cudaMemcpyAsync D→H
                           │
                           ▼
              Postprocess (CPU) → Detections / Masks / Keypoints
```

### Core Components

| Component | File | Role |
|:----------|:-----|:-----|
| **TrtSessionBase** | `core/trt_session_base.hpp` | Engine deserialization, I/O buffer management, warm-up, CUDA Graph capture, async inference |
| **CUDA Preprocessing** | `core/cuda_preprocessing.cu` | Single kernel: letterbox + BGR→RGB + normalize + HWC→NCHW |
| **Version Detection** | `core/version.hpp` | Auto-detect YOLO version from output tensor shape |
| **NMS** | `core/nms.hpp` | Batched non-maximum suppression |
| **Drawing** | `core/drawing.hpp` | Bounding box, mask, and skeleton visualization |
| **Task Heads** | `tasks/*.hpp` | Version-aware postprocessing for each task type |

---

## 📂 Project Structure

```
YOLOs-CPP-TensorRT/
├── include/yolos/            # Header-only library
│   ├── core/                 #   Engine, preprocessing, NMS, drawing, types
│   └── tasks/                #   Detection, segmentation, pose, OBB, classification, depth
├── src/                      # Ready-to-use inference binaries
│   ├── image_inference.cpp   #   Single image / folder
│   ├── video_inference.cpp   #   Video file (multi-threaded)
│   ├── camera_inference.cpp  #   Live camera feed
│   ├── batch_image_inference.cpp
│   ├── class_image_inference.cpp
│   └── depth_image_inference.cpp #   Monocular metric depth
├── tools/                    # onnx2trt — dependency-free engine builder
├── examples/                 # Per-task examples (image / video / camera)
├── benchmarks/               # Unified benchmark tool (FPS, latency, mAP)
├── tests/                    # Per-task validation suites (C++ vs Python ground truth)
├── models/                   # ONNX export script + label files
├── trt-files/scripts/        # TensorRT conversion & INT8 calibration tools
├── doc/                      # Installation, usage, quantization, contributing guides
├── Dockerfile.tensorrt       # Multi-stage Docker (data-center GPU)
├── Dockerfile.tensorrt.jetson # Multi-stage Docker (Jetson)
└── CMakeLists.txt            # Top-level build (CXX + CUDA)
```

---

## 🏃 Built-in Binaries

The build produces five ready-to-use executables:

```bash
# Object detection on a single image
./image_inference models/yolo11n.trt data/dog.jpg models/coco.names

# Process a video file
./video_inference models/yolo11n.trt data/video.mp4 output.mp4 models/coco.names

# Live camera feed (V4L2 / RTSP)
./camera_inference models/yolo11n.trt /dev/video0 models/coco.names

# Batch detection over a folder
./batch_image_inference models/yolo11n.trt data/ models/coco.names

# Image classification
./class_image_inference models/yolov8n-cls.trt data/dog.jpg models/ImageNet.names

# Monocular metric depth (third arg writes a colorized PNG instead of showing it)
./depth_image_inference models/yolo26n-depth.trt data/dog.jpg depth.png
```

---

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

---

## 🧪 Testing

```bash
cd tests
./test_all.sh          # Run all task tests
./test_detection.sh    # Detection only
./test_segmentation.sh # Segmentation only
./test_pose.sh         # Pose estimation only
./test_obb.sh          # Oriented bounding box only
./test_classification.sh # Classification only
./test_depth.sh        # Metric depth estimation only
```

Tests export models via Ultralytics, convert to TRT engines, run inference in both Python and C++, and compare outputs for correctness.

---

## 📊 Benchmarking

```bash
cd benchmarks
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Quick performance test
./yolo_unified_benchmark quick models/yolo11n.trt data/dog.jpg

# Comprehensive multi-model sweep
./yolo_unified_benchmark comprehensive
```

See the [Benchmark README](benchmarks/README.md) for all modes (image, video, camera, accuracy evaluation).

---

## 🔧 Troubleshooting

<details>
<summary><b>"TensorRT not found" during CMake</b></summary>

```bash
# Check if TensorRT is installed
dpkg -l | grep nvinfer

# Point CMake to a custom install
cmake .. -DTENSORRT_DIR=/opt/TensorRT-10.4
```

</details>

<details>
<summary><b>"CUDA graph capture failed"</b></summary>

This is expected for **dynamic-shape** models. YOLOs-TRT automatically falls back to standard `enqueueV3` dispatch. Performance impact is minimal (~0.1–0.3 ms).

</details>

<details>
<summary><b>Engine fails to load / crashes on inference</b></summary>

TensorRT engines are **GPU-specific** and **TRT-version-specific**. Rebuild on the target device:

```bash
./build/onnx2trt model.onnx model.trt --fp16
```

</details>

<details>
<summary><b>Low FPS on first few frames</b></summary>

The 10-iteration warm-up handles this automatically. If you're benchmarking manually, discard the first ~10 frames.

</details>

---

## 🤝 Contributing

Contributions are welcome! Please see the [Contributing Guide](doc/contributing.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push to the branch and open a Pull Request

---

## 📚 Documentation

| Guide | Description |
|:------|:------------|
| [Installation](doc/installation.md) | Detailed setup for Ubuntu, Jetson, and Docker |
| [Usage](doc/usage.md) | API walkthrough with examples |
| [Models](doc/models.md) | Model export, conversion, and optimization |
| [Quantization](doc/quantization.md) | INT8 calibration and precision tuning |
| [Development](doc/development.md) | Architecture deep-dive and extending the library |
| [Acknowledgments](doc/acknowledgments.md) | Credits and references |

---

## ⭐ Star History

If YOLOs-TRT helps your project, consider giving it a star — it helps others discover it!

[![Star History Chart](https://api.star-history.com/svg?repos=Geekgineer/YOLOs-CPP-TensorRT&type=Date)](https://star-history.com/#Geekgineer/YOLOs-CPP-TensorRT&Date)

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) and [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**

<sub>Made with dedication to pushing the limits of real-time inference.</sub>

</div>
