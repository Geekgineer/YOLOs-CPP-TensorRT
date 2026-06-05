# YOLOs-CPP-TensorRT

**High-Performance YOLO Inference Engine in C++ with TensorRT**

A fast, clean, and maintainable implementation for running modern YOLO models (YOLOv8, YOLOv10, YOLOv11) using NVIDIA TensorRT.

---

### Features

- TensorRT 8.6 acceleration for maximum speed
- Custom CUDA preprocessing (letterbox + normalization)
- Clean C++17 modular architecture
- Batch image inference support
- Easy to extend for custom models
- Successfully builds and runs on WSL2

### Tech Stack

- C++17 + CUDA
- TensorRT 8.6
- OpenCV 4
- CMake

### Project Structure

| Directory/File                    | Purpose |
|----------------------------------|--------|
| `src/core/trt_session_base.cpp`  | Core TensorRT engine loading & context management |
| `src/tasks/detection.cpp`        | YOLO object detection logic + post-processing |
| `src/tasks/classification.cpp`   | Image classification logic |
| `src/batch_image_inference.cpp`  | Main executable for batch inference |
| `include/yolos/`                 | All header files |
| `CMakeLists.txt`                 | Build configuration |
| `models/`                        | Place your `.trt` models and `coco.names` here |

### How to Build

```bash
cd build
cmake .. -DTENSORRT_DIR=/path/to/TensorRT-8.6.1.6
make -j4
How to Run
Bash./batch_image_inference ../models/yolo11n.trt ../models/coco.names ../data/dog.jpg
Current Status

✅ Compiles and runs successfully on WSL2
✅ Fixed missing source files, CMake issues, and linker errors
⚠️ Real inference is currently in placeholder mode (add a valid .trt model to enable full detection)


Made with ❤️ for learning high-performance Computer Vision
