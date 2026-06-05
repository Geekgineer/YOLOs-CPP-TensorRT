/**
 * @file class_image_inference.cpp
 * @brief Image classification using YOLO TensorRT engines.
 *
 * Usage:
 *   ./class_image_inference [engine_path] [image_path] [labels_path]
 *
 * Author: YOLOs-TRT Team
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <chrono>

#include "yolos/tasks/classification.hpp"

using namespace yolos::cls;

int main(int argc, char** argv) {
    std::string labelsPath = "../models/ImageNet.names";
    std::string imagePath  = "../data/dog.jpg";
    std::string modelPath  = "../models/yolov8n-cls.trt";

    if (argc > 1) modelPath  = argv[1];
    if (argc > 2) imagePath  = argv[2];
    if (argc > 3) labelsPath = argv[3];

    // Initialize classifier with TensorRT engine
    YOLOClassifier classifier(modelPath, labelsPath);

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: could not open image: " << imagePath << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    ClassificationResult result = classifier.classify(image);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << "Classification took: " << elapsed.count() << " ms\n";
    std::cout << "Result: " << result.className << " (" << result.confidence * 100.0f << "%)" << std::endl;

    classifier.drawResult(image, result, cv::Point(18, 28));
    cv::imshow("Classification", image);
    cv::waitKey(0);
    return 0;
}
