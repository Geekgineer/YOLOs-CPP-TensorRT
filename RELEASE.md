# Release Guide for YOLOs-TRT

This guide explains how to prepare and publish a release for YOLOs-TRT.

## Pre-Release Checklist

### 1. Verify All Tests Pass

```bash
cd tests
./test_all.sh
```

Expected output: All 44 tests should pass (100%).

### 2. Run Benchmarks

```bash
cd benchmarks
./auto_bench.sh
```

### 3. Update Version Numbers

Update version in:
- `benchmarks/yolo_unified_benchmark.cpp` (line ~54: `BENCHMARK_VERSION`)
- `README.md` (badges and documentation)

### 4. Prepare Model Assets

```bash
./scripts/prepare_release.sh
```

This creates zip files in `release_assets/`.

## Creating the Release

### Step 1: Create Model Assets Release

1. Go to GitHub -> Releases -> "Create new release"
2. **Tag:** `v2.0.0-models`
3. **Title:** "Model Assets v2.0.0"
4. **Description:**
   ```
   Pre-trained YOLO models for YOLOs-TRT tests.
   
   Included models (ONNX format — convert to TRT on target GPU):
   - Detection: YOLOv5, v6, v8, v9, v10, v11, v12, YOLO26
   - Segmentation: YOLOv8, v11, YOLO26
   - Pose: YOLOv8, v11, YOLO26
   - OBB: YOLOv8, v11, YOLO26
   - Classification: YOLOv8, v11, YOLO26
   
   Note: TensorRT engines are GPU-specific. Download ONNX models
   and convert on your target hardware using trtexec or the
   included convert_to_tensorrt.py script.
   ```
5. Upload all `.zip` files from `release_assets/`
6. Publish release

### Step 2: Create Main Release

1. Go to GitHub -> Releases -> "Create new release"
2. **Tag:** `v2.0.0`
3. **Title:** "YOLOs-TRT v2.0.0"
4. Use the release notes template below
5. Publish release

## Release Notes Template

```markdown
## What's New in v2.0.0

### TensorRT Backend (Breaking Change)

This release replaces the ONNX Runtime backend with NVIDIA TensorRT for
maximum inference performance on NVIDIA GPUs. Key changes:

- **Zero CPU preprocessing**: Single CUDA kernel for letterbox + BGR-to-RGB + normalize
- **CUDA graph capture**: Eliminates per-frame kernel launch overhead
- **Pinned staging buffers**: Truly async host-to-device transfers
- **GPU-only execution**: TensorRT is always GPU

### Performance (RTX 2000 Ada Laptop GPU, YOLOv11n 640x640)

| Precision | FPS | P50 Latency | Peak GPU Memory |
|:---------:|:---:|:-----------:|:---------------:|
| FP32      | 466 | 2.040 ms    | 529.5 MB        |
| FP16      | 479 | 1.976 ms    | 535.7 MB        |
| INT8      | 530 | 1.775 ms    | 443.7 MB        |

### Supported Models
| Version | Detection | Segmentation | Pose | OBB | Classification |
|---------|-----------|--------------|------|-----|----------------|
| YOLOv5  | Y | - | - | - | - |
| YOLOv6  | Y | - | - | - | - |
| YOLOv8  | Y | Y | Y | Y | Y |
| YOLOv9  | Y | - | - | - | - |
| YOLOv10 | Y | - | - | - | - |
| YOLOv11 | Y | Y | Y | Y | Y |
| YOLOv12 | Y | - | - | - | - |
| YOLO26  | Y | Y | Y | Y | Y |

### Requirements
- NVIDIA GPU (Compute Capability >= 7.5)
- CUDA Toolkit >= 12.0
- TensorRT >= 10.0
- OpenCV >= 4.5
- CMake >= 3.18
- C++17 compiler

### Breaking Changes
- ONNX Runtime backend removed; TensorRT is now the sole backend
- Constructor no longer accepts `isGPU` parameter (always GPU)
- Model files must be `.trt` (serialized TensorRT engines), not `.onnx`
- `build.sh` no longer downloads ONNX Runtime; uses system TensorRT
```

## Post-Release

### Verify CI/CD

After release, verify that:
1. GitHub Actions workflow passes (compilation check)
2. Tests can download models from releases
3. Docker images build successfully
4. Documentation links work

## File Locations

| Purpose | Location |
|---------|----------|
| CI/CD workflow | `.github/workflows/main.yml` |
| Release prep script | `scripts/prepare_release.sh` |
| Dockerfiles | `Dockerfile.tensorrt`, `Dockerfile.tensorrt.jetson` |
| Model converter | `trt-files/scripts/convert_to_tensorrt.py` |
