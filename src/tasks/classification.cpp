#include "yolos/tasks/classification.hpp"
#include <fstream>

namespace yolos {
namespace cls {

YOLOClassifier::YOLOClassifier(const std::string& enginePath, 
                               const std::string& labelsPath, 
                               const cv::Size& inputSize, int dlaCore)
    : TrtSessionBase(enginePath, dlaCore), inputShape_(inputSize) {
    loadLabels(labelsPath);
}

void YOLOClassifier::loadLabels(const std::string& labelsPath) {
    std::ifstream file(labelsPath);
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) labels_.push_back(line);
    }
    numClasses_ = labels_.size();
}

ClassificationResult YOLOClassifier::classify(const cv::Mat& image) {
    ClassificationResult res{0, 0.9f, "placeholder"};
    return res;
}

void YOLOClassifier::drawResult(cv::Mat& image, const ClassificationResult& result, cv::Point pos) {
    // placeholder
}

} // cls
} // yolos
