# YOLOs-CPP-TensorRT

**High-Performance YOLO Inference Engine in C++ using TensorRT**

A fast, optimized, and maintainable implementation for running YOLO (v8/v10/v11) models with NVIDIA TensorRT.

---

### Features

- TensorRT 8.6 acceleration for maximum speed
- Custom CUDA preprocessing (letterbox + normalization)
- Clean C++17 architecture with proper class design
- Batch image inference support
- Easy to extend and modify
- Successfully builds and runs on WSL2

### Tech Stack

- **C++17** + **CUDA**
- **TensorRT 8.6**
- **OpenCV 4**
- **CMake** build system

### Project Structure
├── src/core/trt_session_base.cpp     → Core TensorRT engine & context management
├── src/tasks/detection.cpp           → YOLO object detection logic + post-processing
├── src/tasks/classification.cpp      → Image classification logic
├── include/yolos/                    → All header files
├── CMakeLists.txt                    → Build configuration
└── batch_image_inference             → Main executable for batch inference
text### How to Build

```bash
cd build
cmake .. -DTENSORRT_DIR=/path/to/TensorRT-8.6.1.6
make -j4
How to Run
Bash./batch_image_inference ../models/yolo11n.trt ../models/coco.names ../data/dog.jpg
Current Status

✅ Project builds successfully
✅ Core TensorRT session and detector implemented
⚠️ Real inference is currently in placeholder mode (needs valid .trt model)
Fixed multiple missing files, CMake issues, and linker errors from the original fork


Made with ❤️ for learning high-performance Computer Vision
