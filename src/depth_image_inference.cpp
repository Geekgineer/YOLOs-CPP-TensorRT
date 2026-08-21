/**
 * @file depth_image_inference.cpp
 * @brief Monocular metric depth estimation using YOLO26-depth TensorRT engines.
 *
 * Usage:
 *   ./depth_image_inference [engine_path] [image_path] [output_path]
 *
 * Author: YOLOs-TRT Team
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <iostream>
#include <string>

#include "yolos/tasks/depth.hpp"

using namespace yolos::depth;

int main(int argc, char** argv) {
    std::string modelPath  = "../models/yolo26n-depth.engine";
    std::string imagePath  = "../data/dog.jpg";
    std::string outputPath;

    if (argc > 1) modelPath  = argv[1];
    if (argc > 2) imagePath  = argv[2];
    if (argc > 3) outputPath = argv[3];

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: could not open image: " << imagePath << std::endl;
        return -1;
    }

    YOLODepthEstimator estimator(modelPath);

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat depth = estimator.estimate(image);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);

    if (depth.empty()) {
        std::cerr << "Error: depth estimation returned an empty map" << std::endl;
        return -1;
    }

    double minDepth = 0.0;
    double maxDepth = 0.0;
    cv::minMaxLoc(depth, &minDepth, &maxDepth);

    std::cout << "Depth estimation took: " << elapsed.count() << " ms\n";
    std::cout << "Depth range: " << minDepth << " m .. " << maxDepth << " m\n";
    std::cout << "Center depth: "
              << YOLODepthEstimator::depthAt(depth, depth.cols / 2, depth.rows / 2)
              << " m" << std::endl;

    cv::Mat colored = colorizeDepth(depth);

    if (!outputPath.empty()) {
        if (!cv::imwrite(outputPath, colored)) {
            std::cerr << "Error: could not write output: " << outputPath << std::endl;
            return -1;
        }
        std::cout << "Wrote depth visualization to: " << outputPath << std::endl;
        return 0;
    }

    cv::imshow("Depth (metric)", colored);
    cv::imshow("Depth overlay", overlayDepth(image, depth));
    cv::waitKey(0);
    return 0;
}
