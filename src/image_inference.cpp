/**
 * @file image_inference.cpp
 * @brief Object detection in a static image using YOLO TensorRT engines.
 *
 * This application loads a TensorRT engine, runs object detection on one or
 * more images, and displays the results with bounding boxes.
 *
 * Usage:
 *   ./image_inference <engine_path> [image_path_or_folder] [labels_path]
 *
 * Author: YOLOs-TRT Team
 */

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>

#include "yolos/tasks/detection.hpp"

using namespace yolos::det;

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    // Paths to the model, labels, and test image
    std::string labelsPath = "../models/coco.names";
    std::string imagePath  = "../data/dog.jpg";
    std::string modelPath  = "../models/yolo11n.trt";
    std::vector<std::string> imageFiles;

    if (argc > 1) {
        modelPath = argv[1];
    }
    if (argc > 2) {
        imagePath = argv[2];
        if (fs::is_directory(imagePath)) {
            for (const auto& entry : fs::directory_iterator(imagePath)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff" || ext == ".tif") {
                        imageFiles.push_back(fs::absolute(entry.path()).string());
                    }
                }
            }
            if (imageFiles.empty()) {
                std::cerr << "No image files found in directory: " << imagePath << std::endl;
                return -1;
            }
        } else if (fs::is_regular_file(imagePath)) {
            imageFiles.push_back(imagePath);
        } else {
            std::cerr << "Provided path is not a valid file or directory: " << imagePath << std::endl;
            return -1;
        }
    } else {
        std::cout << "Usage: " << argv[0] << " <engine_path> [image_path_or_folder] [labels_path]\n";
        std::cout << "No image path provided. Using default: " << imagePath << std::endl;
        imageFiles.push_back(imagePath);
    }
    if (argc > 3) {
        labelsPath = argv[3];
    }

    // Initialize the YOLO detector with TensorRT engine
    YOLODetector detector(modelPath, labelsPath);

    for (const auto& imgPath : imageFiles) {
        std::cout << "\nProcessing: " << imgPath << std::endl;

        cv::Mat image = cv::imread(imgPath);
        if (image.empty()) {
            std::cerr << "Error: Could not open or find the image!\n";
            continue;
        }

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> results = detector.detect(image);
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - start);

        std::cout << "Detection completed in: " << duration.count() << " ms" << std::endl;
        std::cout << "Number of detections found: " << results.size() << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "Detection " << i << ": Class=" << results[i].classId
                      << ", Confidence=" << results[i].conf
                      << ", Box=(" << results[i].box.x << "," << results[i].box.y
                      << "," << results[i].box.width << "," << results[i].box.height << ")" << std::endl;
        }

        detector.drawDetections(image, results);
        cv::imshow("Detections", image);
        cv::waitKey(0);
    }

    return 0;
}
