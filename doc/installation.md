# Installation Guide

This guide covers system requirements, build options, and troubleshooting for YOLOs-TRT.

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 20.04+) |
| **GPU** | NVIDIA GPU, Compute Capability >= 7.5 (Turing, Ampere, Ada, Hopper) |
| **CUDA Toolkit** | >= 12.0 |
| **TensorRT** | 10.x (**not** 11.x — see note) |
| **Compiler** | GCC 9+, Clang 10+ (C++17 support) |
| **CMake** | >= 3.18 (CUDA language support) |
| **OpenCV** | >= 4.5 |

### Jetson Requirements

| Component | Requirement |
|-----------|-------------|
| **Platform** | Jetson Xavier or Orin |
| **JetPack** | >= 6.0 (includes CUDA + TensorRT) |
| **L4T** | >= 36.x |

## Quick Install (Linux)

```bash
# Clone the repository
git clone https://github.com/Geekgineer/YOLOs-CPP-TensorRT.git
cd YOLOs-CPP

# Build (auto-detects system TensorRT and CUDA)
./build.sh
```

## Step-by-Step Install

### Step 1: Install CUDA Toolkit

**Ubuntu 22.04 / 24.04:**

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install -y cuda-toolkit-12-4
```

**Verify:**

```bash
nvcc --version
# Expected: Cuda compilation tools, release 12.4, ...
```

### Step 2: Install TensorRT

**Ubuntu (apt):**

```bash
sudo apt-get install -y tensorrt
```

This installs TensorRT libraries, headers, and `trtexec` CLI.

**Jetson (JetPack):**

TensorRT is pre-installed with JetPack. Verify:

```bash
dpkg -l | grep tensorrt
```

**Manual install:**

Download from <https://developer.nvidia.com/tensorrt> and extract. Set the `TENSORRT_DIR` CMake variable when building.

**Verify:**

```bash
# Check trtexec is available
trtexec --help | head -5

# Check headers
ls /usr/include/x86_64-linux-gnu/NvInfer.h 2>/dev/null || \
ls /usr/include/NvInfer.h 2>/dev/null || \
echo "TensorRT headers not found"
```

### Step 3: Install OpenCV

```bash
sudo apt-get install -y libopencv-dev
```

### Step 4: Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Custom TensorRT path:**

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DTENSORRT_DIR=/opt/TensorRT-10.4
```

**Specific CUDA architectures:**

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;89"
```

### Step 5: Verify

```bash
# You need a .trt engine file first (see Model Conversion below)
./image_inference ../models/yolo11n.trt ../data/dog.jpg ../models/coco.names
```

## Model Conversion

C++ inference requires serialized TensorRT engine files (`.trt`). You need Python for conversion only.

### Install Python Dependencies

```bash
# Using uv (recommended, fast)
pip install uv
uv pip install -r requirements.txt

# Or standard pip
pip install -r requirements.txt
```

### Export from Ultralytics to ONNX

```bash
python models/export_onnx.py --model yolo11n
```

### Convert ONNX to TensorRT Engine

```bash
# FP16 (recommended for most GPUs)
python trt-files/scripts/convert_to_tensorrt.py --onnx models/yolo11n.onnx --fp16

# INT8 with calibration data
python trt-files/scripts/convert_to_tensorrt.py \
    --onnx models/yolo11n.onnx --int8 \
    --calibration-data trt-files/scripts/calibration_data/

# FP32 (baseline)
python trt-files/scripts/convert_to_tensorrt.py --onnx models/yolo11n.onnx

# Or use trtexec directly
trtexec --onnx=models/yolo11n.onnx --saveEngine=models/yolo11n.trt --fp16
```

**Note:** TensorRT engines are GPU-specific. An engine built on one GPU model will not work on a different GPU architecture. Always rebuild engines on the target hardware.

## Build Options

| CMake Variable | Default | Description |
|----------------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug/Release/RelWithDebInfo) |
| `TENSORRT_DIR` | auto-detect | Path to custom TensorRT installation |
| `CMAKE_CUDA_ARCHITECTURES` | `75;80;86;89;90` (Jetson: `72;87`) | Target GPU compute capabilities. **Not** auto-detected — see the note below. |
| `BUILD_EXAMPLES` | OFF | Build task-specific examples in `examples/` |

> **Pin TensorRT to 10.x.** `apt-get install libnvinfer-dev` now resolves to
> TensorRT 11.x built against CUDA 13. This project targets the TensorRT 10
> tensor API (`enqueueV3`, `setTensorAddress`), so 11.x will not build. To hold
> the whole family back:
>
> ```bash
> sudo tee /etc/apt/preferences.d/tensorrt-pin > /dev/null <<'PIN'
> Package: libnvinfer* libnvonnxparsers* tensorrt*
> Pin: version 10.*
> Pin-Priority: 1001
> PIN
> sudo apt-get update
> ```

## Docker

### Data-center GPU

```bash
docker build -f Dockerfile.tensorrt -t yolos-trt .
docker run --gpus all -v /path/to/models:/app/models yolos-trt
```

### Jetson

```bash
docker build -f Dockerfile.tensorrt.jetson -t yolos-trt-jetson .
docker run --runtime nvidia -v /path/to/models:/app/models yolos-trt-jetson
```

## Troubleshooting

### "nvcc not found"

```bash
sudo apt install cuda-toolkit-12-4
export PATH=/usr/local/cuda/bin:$PATH
```

### "TensorRT not found" during CMake

```bash
# Check if TensorRT is installed
dpkg -l | grep nvinfer

# If using a manual install, specify the path
cmake .. -DTENSORRT_DIR=/opt/TensorRT-10.4
```

### "OpenCV not found"

```bash
sudo apt install libopencv-dev
pkg-config --modversion opencv4
```

### CUDA architectures are not auto-detected

The build targets `75;80;86;89;90` by default (`72;87` on Jetson). Narrowing it
to your own GPU cuts compile time substantially:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap \
             --format=csv,noheader | head -1 | tr -d '.')"
```

> **Do not** let this end up as `52`. CMake's `project(... LANGUAGES CUDA)`
> seeds `CMAKE_CUDA_ARCHITECTURES` from the nvcc default (sm_52) when nothing
> else sets it. An sm_52 build of the CUDA preprocessing kernel still *runs* on
> a modern GPU via PTX JIT, but returns incorrect pixel data — inference then
> produces plausible-looking but wrong results, with no error anywhere. The
> build now fails outright if the value is `52`; if you see that error, pass an
> architecture explicitly rather than working around it.

### "CUDA architecture mismatch"

Check your GPU's compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

Then specify it:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"
```

### Engine fails to load

TensorRT engines are GPU-specific and TRT-version-specific. Rebuild:

```bash
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

## Next Steps

- [Usage Guide](usage.md) -- Learn the API
- [Model Guide](models.md) -- Export and optimize models
