#pragma once

// ============================================================================
// YOLO Monocular Metric Depth Estimation (TensorRT Backend)
// ============================================================================
// Dense per-pixel metric depth estimation using YOLO26-depth models
// (yolo26{n,s,m,l,x}-depth).
//
// The exported ONNX/TensorRT graph already contains the clamp, exp and
// log-affine calibration plus the 4x upsample, so the network output is
// metric depth in METERS at the model input resolution. Postprocessing is
// therefore purely geometric: strip the letterbox padding and resize back
// to the original image size.
//
// Author: YOLOs-TRT Team, https://github.com/Geekgineer/YOLOs-CPP-TensorRT
// ============================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolos/core/types.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/trt_session_base.hpp"

namespace yolos {
namespace depth {

// ============================================================================
// Depth Colorization Helpers
// ============================================================================

/// @brief Normalize a metric depth map into an 8-bit image for visualization.
/// @param depthMeters  CV_32FC1 depth map in meters
/// @param minDepth     Lower clamp in meters; <= 0 means "use map minimum"
/// @param maxDepth     Upper clamp in meters; <= 0 means "use map maximum"
/// @return CV_8UC1 image where 0 = nearest and 255 = farthest
inline cv::Mat normalizeDepth(const cv::Mat& depthMeters,
                              float minDepth = 0.0f,
                              float maxDepth = 0.0f) {
    if (depthMeters.empty()) return {};

    double lo = static_cast<double>(minDepth);
    double hi = static_cast<double>(maxDepth);

    if (lo <= 0.0 || hi <= 0.0) {
        double dataMin = 0.0;
        double dataMax = 0.0;
        cv::minMaxLoc(depthMeters, &dataMin, &dataMax);
        if (lo <= 0.0) lo = dataMin;
        if (hi <= 0.0) hi = dataMax;
    }

    // Degenerate range — emit a flat mid-gray rather than dividing by zero.
    if (!(hi > lo)) {
        return cv::Mat(depthMeters.size(), CV_8UC1, cv::Scalar(128));
    }

    cv::Mat clamped;
    cv::max(depthMeters, lo, clamped);
    cv::min(clamped, hi, clamped);

    cv::Mat out;
    clamped.convertTo(out, CV_8UC1, 255.0 / (hi - lo), -255.0 * lo / (hi - lo));
    return out;
}

/// @brief Apply a perceptual colormap to a metric depth map.
/// @param depthMeters  CV_32FC1 depth map in meters
/// @param minDepth     Lower clamp in meters; <= 0 means "use map minimum"
/// @param maxDepth     Upper clamp in meters; <= 0 means "use map maximum"
/// @param colormap     OpenCV colormap id (INFERNO reads well for depth)
/// @return CV_8UC3 BGR visualization
inline cv::Mat colorizeDepth(const cv::Mat& depthMeters,
                             float minDepth = 0.0f,
                             float maxDepth = 0.0f,
                             int colormap = cv::COLORMAP_INFERNO) {
    cv::Mat gray = normalizeDepth(depthMeters, minDepth, maxDepth);
    if (gray.empty()) return {};

    cv::Mat colored;
    cv::applyColorMap(gray, colored, colormap);
    return colored;
}

/// @brief Blend a colorized depth map over the original image.
/// @param image        CV_8UC3 BGR source image
/// @param depthMeters  CV_32FC1 depth map in meters, same size as image
/// @param alpha        Depth opacity in [0, 1]
/// @param minDepth     Lower clamp in meters; <= 0 means "use map minimum"
/// @param maxDepth     Upper clamp in meters; <= 0 means "use map maximum"
/// @param colormap     OpenCV colormap id
/// @return CV_8UC3 blended image (a copy of `image` if depth is unusable)
inline cv::Mat overlayDepth(const cv::Mat& image,
                            const cv::Mat& depthMeters,
                            float alpha = 0.6f,
                            float minDepth = 0.0f,
                            float maxDepth = 0.0f,
                            int colormap = cv::COLORMAP_INFERNO) {
    if (image.empty()) return {};
    if (depthMeters.empty()) return image.clone();

    cv::Mat colored = colorizeDepth(depthMeters, minDepth, maxDepth, colormap);
    if (colored.empty()) return image.clone();

    if (colored.size() != image.size()) {
        cv::resize(colored, colored, image.size(), 0, 0, cv::INTER_LINEAR);
    }

    const double a = static_cast<double>(utils::clamp(alpha, 0.0f, 1.0f));
    cv::Mat blended;
    cv::addWeighted(colored, a, image, 1.0 - a, 0.0, blended);
    return blended;
}

// ============================================================================
// YOLODepthEstimator
// ============================================================================

/// @brief Monocular metric depth estimator for YOLO26-depth TensorRT engines.
///
/// Usage:
/// @code
///   yolos::depth::YOLODepthEstimator estimator("yolo26n-depth.engine");
///   cv::Mat depth = estimator.estimate(image);   // CV_32FC1, meters
///   float d = estimator.depthAt(depth, x, y);    // meters at a pixel
/// @endcode
class YOLODepthEstimator : public TrtSessionBase {
public:
    /// @brief Constructor
    /// @param enginePath Path to the TensorRT engine file
    /// @param dlaCore    DLA core index (-1 = GPU, 0/1 = DLA on Jetson)
    explicit YOLODepthEstimator(const std::string& enginePath, int dlaCore = -1)
        : TrtSessionBase(enginePath, dlaCore) {

        if (numOutputs() != 1) {
            std::ostringstream ss;
            ss << "Expected 1 output node for a depth model, got " << numOutputs();
            throw std::runtime_error(ss.str());
        }

        const auto& outShape = getOutputShape(0);
        if (outShape.size() != 4 || outShape[1] != 1) {
            std::ostringstream ss;
            ss << "Expected depth output shape [1, 1, H, W], got [";
            for (size_t i = 0; i < outShape.size(); ++i) {
                ss << outShape[i] << (i + 1 < outShape.size() ? ", " : "");
            }
            ss << "]";
            throw std::runtime_error(ss.str());
        }

        depthHeight_ = static_cast<int>(outShape[2]);
        depthWidth_  = static_cast<int>(outShape[3]);

        if (depthHeight_ <= 0 || depthWidth_ <= 0) {
            throw std::runtime_error("Depth model reported a non-positive output resolution");
        }

        std::cout << "[INFO] Depth model loaded" << std::endl;
        std::cout << "[INFO] Input shape: " << inputShape_.width << "x" << inputShape_.height << std::endl;
        std::cout << "[INFO] Depth output: " << depthWidth_ << "x" << depthHeight_
                  << " (metric, meters)" << std::endl;
    }

    ~YOLODepthEstimator() override = default;

    /// @brief Run depth estimation on an image.
    /// @param image Input BGR image (any size)
    /// @return CV_32FC1 metric depth in meters at the original image size,
    ///         or an empty Mat if the input is empty.
    cv::Mat estimate(const cv::Mat& image) {
        if (image.empty()) return {};

        // Full GPU pipeline: H2D → CUDA letterbox → enqueueV3 → D2H
        inferGpu(image);

        return postprocess(image.size());
    }

    /// @brief Depth in meters at a pixel, or NaN if out of bounds.
    /// @param depthMeters CV_32FC1 depth map returned by estimate()
    static float depthAt(const cv::Mat& depthMeters, int x, int y) {
        if (depthMeters.empty() ||
            x < 0 || y < 0 || x >= depthMeters.cols || y >= depthMeters.rows) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return depthMeters.at<float>(y, x);
    }

    /// @brief Native depth output resolution of the engine.
    [[nodiscard]] cv::Size getDepthShape() const noexcept {
        return {depthWidth_, depthHeight_};
    }

private:
    int depthWidth_{0};
    int depthHeight_{0};

    // ------------------------------------------------------------------
    // Postprocess: strip letterbox padding, resize back to original size
    // ------------------------------------------------------------------
    cv::Mat postprocess(const cv::Size& originalSize) const {
        // Wrap the pinned host output as a cv::Mat without copying.
        const cv::Mat raw(depthHeight_, depthWidth_, CV_32FC1,
                          const_cast<float*>(getOutputData(0)));

        // The depth map lives at `depthWidth_ x depthHeight_` while the
        // letterbox padding was computed for the model input resolution,
        // so scale the padding into depth-map coordinates.
        const float scaleX = static_cast<float>(depthWidth_) /
                             static_cast<float>(inputShape_.width);
        const float scaleY = static_cast<float>(depthHeight_) /
                             static_cast<float>(inputShape_.height);

        const float padX = getCachedPadX();
        const float padY = getCachedPadY();

        int x1 = static_cast<int>(std::round((padX - 0.1f) * scaleX));
        int y1 = static_cast<int>(std::round((padY - 0.1f) * scaleY));
        int x2 = static_cast<int>(std::round((inputShape_.width  - padX + 0.1f) * scaleX));
        int y2 = static_cast<int>(std::round((inputShape_.height - padY + 0.1f) * scaleY));

        x1 = std::max(0, std::min(x1, depthWidth_  - 1));
        y1 = std::max(0, std::min(y1, depthHeight_ - 1));
        x2 = std::max(x1 + 1, std::min(x2, depthWidth_));
        y2 = std::max(y1 + 1, std::min(y2, depthHeight_));

        const cv::Mat cropped = raw(cv::Rect(x1, y1, x2 - x1, y2 - y1));

        cv::Mat depthMeters;
        cv::resize(cropped, depthMeters, originalSize, 0, 0, cv::INTER_LINEAR);
        return depthMeters;
    }
};

} // namespace depth
} // namespace yolos
