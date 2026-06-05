#include "yolos/tasks/detection.hpp"
#include <fstream>
#include <iostream>

namespace yolos {
namespace det {

YOLODetector::YOLODetector(const std::string& enginePath, 
                           const std::string& labelsPath, 
                           YOLOVersion version, int dlaCore)
    : TrtSessionBase(enginePath, dlaCore), version_(version), inputShape_(cv::Size(640, 640)) {
    loadLabels(labelsPath);
    if (version_ == YOLOVersion::UNKNOWN) {
        version_ = detectVersion();
    }
}

void YOLODetector::loadLabels(const std::string& labelsPath) {
    std::ifstream file(labelsPath);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) labels_.push_back(line);
    }
}

YOLOVersion YOLODetector::detectVersion() {
    return YOLOVersion::UNKNOWN;
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image, float confThreshold, float iouThreshold) {
    // Placeholder - replace with real inference later
    std::vector<Detection> detections;
    return detections;
}

void YOLODetector::drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = (det.classId < (int)labels_.size()) ? labels_[det.classId] : "object";
        cv::putText(image, label, cv::Point(det.box.x, det.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}

void YOLODetector::drawDetectionsWithMask(cv::Mat& image, const std::vector<Detection>& detections) {
    drawDetections(image, detections);
}

} // namespace det
} // namespace yolos
