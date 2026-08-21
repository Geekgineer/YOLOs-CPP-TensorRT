# Usage Guide

Complete API reference and code examples for YOLOs-TRT.

## Quick Start

```cpp
#include "yolos/yolos.hpp"

// Initialize detector with TensorRT engine — version auto-detected
yolos::det::YOLODetector detector("model.trt", "labels.txt");

// Run inference (GPU preprocessing + TensorRT inference)
auto detections = detector.detect(frame, /*conf=*/0.25f, /*iou=*/0.45f);

// Visualize
detector.drawDetections(frame, detections);
```

## Namespace Structure

| Namespace | Purpose |
|-----------|---------|
| `yolos::det::` | Object detection |
| `yolos::seg::` | Instance segmentation |
| `yolos::pose::` | Pose estimation |
| `yolos::obb::` | Oriented bounding boxes |
| `yolos::cls::` | Image classification |
| `yolos::depth::` | Monocular metric depth estimation |

## Object Detection

```cpp
#include "yolos/tasks/detection.hpp"

yolos::det::YOLODetector detector(
    "models/yolo11n.trt",
    "models/coco.names"
    // YOLOVersion::Auto by default — auto-detected from output tensors
);

cv::Mat image = cv::imread("image.jpg");
auto detections = detector.detect(image, 0.25f, 0.45f);

for (const auto& det : detections) {
    std::cout << det.className << ": " << det.confidence << std::endl;
}

detector.drawDetections(image, detections);
```

## Instance Segmentation

```cpp
#include "yolos/tasks/segmentation.hpp"

yolos::seg::YOLOSegDetector detector(
    "models/yolo11n-seg.trt",
    "models/coco.names"
);

auto segments = detector.segment(image, 0.25f, 0.45f);
detector.drawSegmentations(image, segments, 0.5f);  // 50% opacity
```

## Pose Estimation

```cpp
#include "yolos/tasks/pose.hpp"

yolos::pose::YOLOPoseDetector detector(
    "models/yolo11n-pose.trt",
    ""  // No labels needed for pose
);

auto poses = detector.detect(image, 0.25f, 0.45f);
detector.drawPoses(image, poses);
```

## Oriented Bounding Boxes

```cpp
#include "yolos/tasks/obb.hpp"

yolos::obb::YOLOOBBDetector detector(
    "models/yolo11n-obb.trt",
    "models/Dota.names"
);

auto boxes = detector.detect(image, 0.25f, 0.45f);
detector.drawOBBs(image, boxes);
```

## Image Classification

```cpp
#include "yolos/tasks/classification.hpp"

yolos::cls::YOLOClassifier classifier(
    "models/yolo11n-cls.trt",
    "models/ImageNet.names"
);

auto result = classifier.classify(image);
std::cout << result.className << ": " << result.confidence * 100 << "%" << std::endl;
```

## Video Processing

```cpp
cv::VideoCapture cap("video.mp4");
cv::Mat frame;

while (cap.read(frame)) {
    auto detections = detector.detect(frame);
    detector.drawDetections(frame, detections);
    cv::imshow("Detection", frame);
    if (cv::waitKey(1) == 27) break;
}
```

## Camera Stream

```cpp
cv::VideoCapture cap(0);
cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

cv::Mat frame;
while (cap.read(frame)) {
    auto detections = detector.detect(frame);
    detector.drawDetections(frame, detections);
    cv::imshow("Live", frame);
    if (cv::waitKey(1) == 27) break;
}
```

## DLA on Jetson

To run inference on a DLA core (available on Xavier and Orin):

```cpp
// Use DLA core 0
yolos::det::YOLODetector detector(
    "model.trt", "coco.names",
    yolos::YOLOVersion::Auto,
    /*dlaCore=*/0
);
```

Build the TensorRT engine with DLA support for best compatibility.

## Performance Tips

1. **Reuse detector instances** -- Create once, infer many times. The first inference includes warm-up.
2. **Use FP16 engines** -- Near-identical accuracy, ~2x throughput on Tensor Cores.
3. **Adjust thresholds** -- Higher confidence = fewer detections, faster NMS.
4. **Match input resolution** -- Use the model's expected size (typically 640x640).
5. **Build engines on target hardware** -- TRT engines are GPU-specific.
6. **Warm-up is automatic** -- `TrtSessionBase` runs 10 warm-up iterations at construction.
7. **CUDA graphs** -- Enabled automatically for fixed-shape models, eliminating launch overhead.

## Error Handling

```cpp
try {
    yolos::det::YOLODetector detector("model.trt", "labels.txt");
    auto dets = detector.detect(image);
} catch (const std::runtime_error& e) {
    std::cerr << "TensorRT error: " << e.what() << std::endl;
}
```

Common errors:
- `Failed to deserialize engine` -- Engine built for a different GPU or TRT version. Rebuild it.
- `CUDA out of memory` -- Use a smaller model or reduce batch size.
- `Invalid engine file` -- File is corrupted or is an ONNX file (needs conversion first).

## Factory Functions

For convenience, factory functions handle auto-detection:

```cpp
// Detection — auto-detect version
auto det = yolos::det::createDetector("yolo11n.trt", "coco.names");

// Classification
auto cls = yolos::cls::createClassifier("yolov8n-cls.trt", "ImageNet.names");
```

## Next Steps

- [Model Guide](models.md) -- Export and optimize models
- [Development](development.md) -- Extend the library

## Monocular Metric Depth

```cpp
#include "yolos/tasks/depth.hpp"

// Depth models predict a value per pixel, so there is no labels file.
yolos::depth::YOLODepthEstimator estimator("models/yolo26n-depth.trt");

// CV_32FC1 at the ORIGINAL image size. Every value is a distance in METERS —
// the exported graph already applies the log-affine metric calibration, so no
// scaling or normalization is needed on your side.
cv::Mat depthMap = estimator.estimate(image);

// Distance to whatever is at a given pixel.
float meters = yolos::depth::YOLODepthEstimator::depthAt(depthMap, 320, 240);

// Visualization helpers. Passing min/max clamps the colour range, which keeps
// the colours stable across frames of a video (auto-ranging makes them flicker).
cv::Mat colored = yolos::depth::colorizeDepth(depthMap, 0.5f, 20.0f);
cv::Mat blended = yolos::depth::overlayDepth(image, depthMap, 0.6f);
```

Supported weights: `yolo26{n,s,m,l,x}-depth`. Export at `imgsz=768` — that is the
resolution the released weights were trained at, and the engine geometry must
match it for the depth scale to be correct.
