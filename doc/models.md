# Model Guide

Supported models, ONNX export, TensorRT conversion, and optimization for YOLOs-TRT.

## Supported Models

### Detection

| Model | Params | mAP | TRT FP16 Latency |
|-------|-------:|----:|------------------:|
| YOLOv5n | 1.9M | 28.0 | ~1.5ms |
| YOLOv8n | 3.2M | 37.3 | ~1.8ms |
| YOLOv11n | 2.6M | 39.5 | ~2.0ms |
| YOLO26n | 2.5M | 40.2 | ~2.1ms |

### Segmentation

| Model | Params | mAP | TRT FP16 Latency |
|-------|-------:|----:|------------------:|
| YOLOv8n-seg | 3.4M | 36.7 | ~2.5ms |
| YOLOv11n-seg | 2.9M | 38.9 | ~2.3ms |
| YOLO26n-seg | 2.8M | 39.4 | ~2.4ms |

### Pose Estimation

| Model | Params | mAP | TRT FP16 Latency |
|-------|-------:|----:|------------------:|
| YOLOv8n-pose | 3.3M | 50.4 | ~1.7ms |
| YOLOv11n-pose | 2.9M | 52.1 | ~1.6ms |
| YOLO26n-pose | 2.8M | 53.0 | ~1.8ms |

### OBB (Oriented Bounding Boxes)

| Model | Params | Dataset |
|-------|-------:|---------|
| YOLOv8n-obb | 3.1M | DOTA |
| YOLOv11n-obb | 2.7M | DOTA |
| YOLO26n-obb | 2.6M | DOTA |

### Classification

| Model | Params | Top-1 Acc |
|-------|-------:|----------:|
| YOLOv8n-cls | 2.7M | 66.6% |
| YOLOv11n-cls | 1.6M | 70.0% |
| YOLO26n-cls | 1.5M | 71.2% |

### Metric Depth

Monocular depth models predicting per-pixel distance in meters. Trained on a
~2.19M image indoor/outdoor mix; delta-1 accuracy is on NYU Depth V2.

| Model | Params | delta-1 Acc |
|-------|-------:|------------:|
| YOLO26n-depth | 2.4M | 0.882 |
| YOLO26s-depth | - | - |
| YOLO26m-depth | - | - |
| YOLO26l-depth | - | - |
| YOLO26x-depth | - | 0.933 |

Export these at `imgsz=768` (their training resolution).

## Model Pipeline: PyTorch -> ONNX -> TensorRT

```
.pt (PyTorch)  -->  .onnx (ONNX)  -->  .trt (TensorRT Engine)
   Ultralytics       export_onnx.py       onnx2trt / convert_to_tensorrt.py
```

### Step 1: Export to ONNX

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=False,
    half=False,
    dynamic=False,
    nms=False        # NMS is done in C++ postprocessing
)
```

Or use the batch export script:

```bash
python models/export_onnx.py --model yolo11n
```

### Step 2: Convert to TensorRT

**Option A: Using the bundled `onnx2trt` (recommended — no extra dependencies)**

```bash
# Built alongside the inference binaries whenever the ONNX parser is present
./build/onnx2trt models/yolo11n.onnx models/yolo11n.trt --fp16
./build/onnx2trt models/yolo11n.onnx models/yolo11n_fp32.trt
```

It links the same TensorRT libraries the project already needs, so there is no
version-skew risk. For calibrated INT8, use the Python converter below.

**Option B: Using trtexec**

> `trtexec` is **not** installed by the `libnvinfer-dev` or `tensorrt-dev` apt
> packages — it ships with the samples package or the TensorRT tarball. If you
> installed TensorRT from apt, you probably do not have it; use Option A.

```bash
# FP16 (best throughput on Tensor Core GPUs)
trtexec --onnx=models/yolo11n.onnx --saveEngine=models/yolo11n.trt --fp16

# INT8 (requires calibration for best accuracy)
trtexec --onnx=models/yolo11n.onnx --saveEngine=models/yolo11n_int8.trt --int8 \
    --calib=calibration_cache.bin

# FP32 (baseline, largest engine)
trtexec --onnx=models/yolo11n.onnx --saveEngine=models/yolo11n_fp32.trt
```

**Option C: Using the Python converter**

```bash
# FP16
python trt-files/scripts/convert_to_tensorrt.py --onnx models/yolo11n.onnx --fp16

# INT8 with calibration images
python trt-files/scripts/convert_to_tensorrt.py --onnx models/yolo11n.onnx --int8 \
    --calibration-data trt-files/scripts/calibration_data/

# Batch convert all ONNX models in a directory
python trt-files/scripts/convert_to_tensorrt.py --convert-all --models-dir models/ --fp16
```

### Export Options

| Option | Value | Notes |
|--------|-------|-------|
| `opset` | 12-17 | Use 12 for max compatibility |
| `imgsz` | 640 | Match your inference resolution |
| `half` | False | Keep FP32 ONNX; TRT handles precision |
| `dynamic` | False | Static shapes for best TRT performance |
| `nms` | False | C++ handles NMS (or use YOLO26/v10 for built-in NMS) |

## TensorRT Engine Precision Comparison

| Precision | Accuracy | Throughput | Engine Size | Use Case |
|-----------|----------|------------|-------------|----------|
| FP32 | Baseline | 1x | Large | Accuracy-critical applications |
| FP16 | ~Same | 1.5-2x | Medium | General deployment (recommended) |
| INT8 | -0.5-2% mAP | 2-3x | Small | Edge/Jetson, latency-critical |

## Important Notes

1. **GPU-specific**: TRT engines are built for a specific GPU architecture. An engine from an RTX 3090 will NOT work on an RTX 4090.
2. **TRT-version-specific**: Engines are also tied to the TensorRT version. Rebuild when upgrading.
3. **Batch size**: Engines are built for a specific batch size (default: 1). Use `--batch-size` for larger batches.
4. **Warm-up**: The first few inferences are slower as TRT auto-tunes. The library runs 10 warm-up iterations automatically.

## Label Files

| File | Classes | Use Case |
|------|--------:|----------|
| `coco.names` | 80 | General detection/segmentation/pose |
| `Dota.names` | 15 | Aerial/satellite OBB |
| `ImageNet.names` | 1000 | Classification |
| _(none)_ | - | Depth — predicts meters, not classes |

## Model Paths in C++

```cpp
// Detection
YOLODetector det("models/yolo11n.trt", "models/coco.names");

// Segmentation
YOLOSegDetector seg("models/yolo11n-seg.trt", "models/coco.names");

// Pose
YOLOPoseDetector pose("models/yolo11n-pose.trt", "");

// OBB
YOLOOBBDetector obb("models/yolo11n-obb.trt", "models/Dota.names");

// Metric depth (no labels file)
YOLODepthEstimator depth("models/yolo26n-depth.trt");

// Classification
YOLOClassifier cls("models/yolo11n-cls.trt", "models/ImageNet.names");
```

## INT8 Calibration

For INT8 quantization with best accuracy:

```bash
# Generate calibration images (from COCO validation set)
python trt-files/scripts/generate_calibration_data.py --count 500 --output calibration_data/

# Convert with calibration
python trt-files/scripts/convert_to_tensorrt.py \
    --onnx models/yolo11n.onnx \
    --int8 \
    --calibration-data calibration_data/ \
    --calibration-batch-size 8
```

## Custom Models

To use custom-trained models:

1. Train with Ultralytics
2. Export to ONNX with compatible settings
3. Convert to TensorRT engine on target GPU
4. Create matching label file
5. Load in C++

```cpp
yolos::det::YOLODetector detector(
    "custom_model.trt",
    "custom_labels.txt"
);
```

## Next Steps

- [Usage Guide](usage.md) -- API reference
- [Development](development.md) -- Extend the library
