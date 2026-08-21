# Changelog

All notable changes to YOLOs-TRT are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026-08-21

### Added

- **Monocular metric depth estimation** (`yolos::depth::YOLODepthEstimator`) for
  `yolo26{n,s,m,l,x}-depth` models, bringing the TensorRT port up to YOLOs-CPP
  v1.1.0. `estimate()` returns a `CV_32FC1` map at the original image size where
  every value is a distance in **meters** — the exported graph already contains
  the clamp, exp and log-affine calibration, so no scaling is needed by callers.
  Includes `normalizeDepth()`, `colorizeDepth()` and `overlayDepth()` helpers.
- `depth_image_inference` binary and `examples/example_image_depth.cpp`.
- **`tools/onnx2trt`** — a dependency-free ONNX to TensorRT engine builder. It
  links the TensorRT libraries the project already requires, so engine
  conversion no longer depends on `trtexec` (which the apt packages do not
  ship) or on the multi-gigabyte `tensorrt` Python wheel.
- Depth test suite (8 cases) comparing against Ultralytics ground truth via
  per-image statistics plus a deterministic 5x5 sample lattice.

### Fixed

- **CUDA architectures were never applied, silently corrupting GPU inference.**
  `project(... LANGUAGES CUDA)` seeds `CMAKE_CUDA_ARCHITECTURES` from the nvcc
  default, so the `if(NOT DEFINED ...)` guard placed after `project()` was dead
  code and builds targeted **sm_52**. That still JITs on modern GPUs but makes
  the shared CUDA letterbox kernel return wrong pixel data — detection,
  segmentation, pose, OBB and depth all produced plausible but incorrect
  results, with no error reported. Architectures are now chosen before
  `project()`, and a value of `52` is a hard build error.
- Engine conversion had no working path on an apt-only TensorRT install:
  `trtexec` is absent and the Python fallback needs a wheel no test script
  installed. Conversion now prefers `onnx2trt`, then `trtexec`, then the Python
  converter, and reports precisely what is missing when none is available.
- TensorRT is pinned to 10.x in CI. `libnvinfer-dev` now resolves to TensorRT
  11.x built against CUDA 13, which does not build against the TRT 10 tensor
  API this project targets.
- Classification and OBB result comparison matched entries by list position
  while both sides iterated unordered containers, so a passing run depended on
  hash ordering. Comparison is now keyed on `image_path`.
- Missing `#include <set>` in the detection and OBB comparison tests, which
  compiled only via GoogleTest's transitive includes.
- Depth statistics are computed over finite values on both the C++ and Python
  sides; previously a single non-finite pixel made the two disagree.
- `tests/depth/{models,results,data/images}` are tracked, so a fresh clone can
  run the depth suite.
- Docker images copy `tools/` into the build context and ship the
  `depth_image_inference` and `onnx2trt` binaries.

### Changed

- Test result JSON is deterministic: ordered containers and sorted model and
  image discovery, so `results_cpp.json` is stable and diffable.
- CI: GitHub Actions updated off Node 20 (removed from runners 2026-09-16), a
  concurrency group cancels superseded runs, and `gpu-tests` has a 45-minute
  job timeout — previously a queued run could occupy the pipeline for a day.
- Documentation corrected where it was wrong: `CMAKE_CUDA_ARCHITECTURES` was
  documented as "auto-detect" (it is not, and that belief is what produced
  sm_52 builds), and the TensorRT requirement is now stated as 10.x rather than
  ">= 10.0", with apt-pinning instructions.

[3.1.0]: https://github.com/Geekgineer/YOLOs-CPP-TensorRT/releases/tag/v3.1.0
