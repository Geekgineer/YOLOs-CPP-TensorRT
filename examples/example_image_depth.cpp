/**
 * @file example_image_depth.cpp
 * @brief Monocular metric depth estimation on images using YOLO26-depth models
 * @details Produces a dense per-pixel distance map in METERS and saves both a
 *          colorized depth map and a blended overlay.
 */

#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include "yolos/tasks/depth.hpp"
#include "utils.hpp"

using namespace yolos::depth;

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    // Default configuration. Depth models predict a distance, not a class, so
    // there is no labels file.
    std::string modelPath = "../../models/yolo26n-depth.trt";
    std::string inputPath = "../../data/dog.jpg";
    std::string outputDir = "../../outputs/depth/";

    if (argc > 1) modelPath = argv[1];
    if (argc > 2) inputPath = argv[2];
    if (argc > 3) outputDir = argv[3];

    utils::printUsage(argv[0], "Metric Depth", modelPath, inputPath, "(none — depth needs no labels)");

    // Collect image files
    std::vector<std::string> imageFiles;
    if (fs::is_directory(inputPath)) {
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            if (entry.is_regular_file() && utils::isImageFile(entry.path().string())) {
                imageFiles.push_back(fs::absolute(entry.path()).string());
            }
        }
        if (imageFiles.empty()) {
            std::cerr << "❌ No image files found in: " << inputPath << std::endl;
            return -1;
        }
    } else if (fs::is_regular_file(inputPath)) {
        imageFiles.push_back(inputPath);
    } else {
        std::cerr << "❌ Invalid path: " << inputPath << std::endl;
        return -1;
    }

    std::cout << "🔄 Loading depth model: " << modelPath << std::endl;

    try {
        YOLODepthEstimator estimator(modelPath);
        std::cout << "✅ Model loaded successfully!" << std::endl;
        std::cout << "📐 Input shape: " << estimator.getInputShape() << std::endl;
        std::cout << "📐 Depth shape: " << estimator.getDepthShape() << std::endl;

        for (const auto& imgPath : imageFiles) {
            std::cout << "\n📷 Processing: " << imgPath << std::endl;

            cv::Mat image = cv::imread(imgPath);
            if (image.empty()) {
                std::cerr << "❌ Could not load image: " << imgPath << std::endl;
                continue;
            }

            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat depthMap = estimator.estimate(image);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

            if (depthMap.empty()) {
                std::cerr << "❌ Depth estimation returned an empty map" << std::endl;
                continue;
            }

            double minDepth = 0.0;
            double maxDepth = 0.0;
            cv::minMaxLoc(depthMap, &minDepth, &maxDepth);

            std::cout << "✅ Depth estimation completed!" << std::endl;
            std::cout << "  Range:  " << minDepth << " m .. " << maxDepth << " m" << std::endl;
            std::cout << "  Centre: "
                      << YOLODepthEstimator::depthAt(depthMap,
                                                     depthMap.cols / 2,
                                                     depthMap.rows / 2)
                      << " m" << std::endl;

            utils::printMetrics("Metric Depth", duration.count(),
                                duration.count() > 0 ? 1000.0 / duration.count() : -1);

            // Fixing the colour range keeps successive frames comparable;
            // auto-ranging per image makes the palette jump around.
            cv::Mat colored = colorizeDepth(depthMap,
                                            static_cast<float>(minDepth),
                                            static_cast<float>(maxDepth));
            cv::Mat overlay = overlayDepth(image, depthMap, 0.6f,
                                           static_cast<float>(minDepth),
                                           static_cast<float>(maxDepth));

            std::string depthOut   = utils::saveImage(colored, imgPath, outputDir);
            std::string overlayOut = utils::saveImage(overlay, imgPath, outputDir + "overlay/");
            std::cout << "💾 Saved depth map: " << depthOut << std::endl;
            std::cout << "💾 Saved overlay:   " << overlayOut << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
