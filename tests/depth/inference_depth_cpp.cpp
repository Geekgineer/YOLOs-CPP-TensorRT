/**
 * @file inference_depth_cpp.cpp
 * @brief Monocular metric depth estimation inference test for YOLOs-TRT.
 *
 * Emits results/results_cpp.json in the same schema as
 * inference_depth_ultralytics.py so compare_results.cpp can diff the two.
 */

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "yolos/tasks/depth.hpp"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace yolos::depth;

// Must match GRID_N in inference_depth_ultralytics.py
constexpr int GRID_N = 5;

struct SamplePoint {
    int x{0};
    int y{0};
    float depth{0.0f};
};

struct DepthImageResult {
    int width{0};
    int height{0};
    double minDepth{0.0};
    double maxDepth{0.0};
    double meanDepth{0.0};
    double medianDepth{0.0};
    std::vector<SamplePoint> samples;
};

struct ResultsDepth {
    std::string weightsPath;
    std::string task;
    std::map<std::string, DepthImageResult> inferenceResults;
};

bool validatePaths(const std::unordered_map<std::string, std::string>& paths) {
    for (const auto& [key, path] : paths) {
        if (!fs::exists(path)) {
            std::cerr << "Error: " << key << " path does not exist: " << path << std::endl;
            return false;
        }
    }
    std::cout << "All paths are valid." << std::endl;
    return true;
}

bool loadImages(const std::string& imagesPath, std::vector<std::string>& imageFiles) {
    if (!fs::exists(imagesPath) || !fs::is_directory(imagesPath)) {
        std::cerr << "Error: Images path does not exist: " << imagesPath << std::endl;
        return false;
    }
    const std::vector<std::string> validExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    for (const auto& entry : fs::directory_iterator(imagesPath)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(validExtensions.begin(), validExtensions.end(), ext) != validExtensions.end()) {
            imageFiles.push_back(entry.path().string());
        }
    }
    // Deterministic order so results_cpp.json is stable and diffable
    std::sort(imageFiles.begin(), imageFiles.end());
    return !imageFiles.empty();
}

void findModels(const std::string& modelsDir, std::vector<std::string>& modelFiles) {
    if (!fs::exists(modelsDir) || !fs::is_directory(modelsDir)) return;
    for (const auto& entry : fs::directory_iterator(modelsDir)) {
        if (entry.is_regular_file() &&
            (entry.path().extension() == ".trt" || entry.path().extension() == ".engine")) {
            modelFiles.push_back(entry.path().string());
        }
    }
    std::sort(modelFiles.begin(), modelFiles.end());
}

/// @brief Deterministic GRID_N x GRID_N pixel lattice — must match the Python side.
std::vector<cv::Point> samplePoints(int width, int height) {
    std::vector<cv::Point> points;
    points.reserve(static_cast<size_t>(GRID_N) * GRID_N);
    for (int gy = 0; gy < GRID_N; ++gy) {
        for (int gx = 0; gx < GRID_N; ++gx) {
            int x = static_cast<int>((gx + 0.5) / GRID_N * width);
            int y = static_cast<int>((gy + 0.5) / GRID_N * height);
            x = std::min(std::max(x, 0), width - 1);
            y = std::min(std::max(y, 0), height - 1);
            points.emplace_back(x, y);
        }
    }
    return points;
}

/// @brief min/max/mean/median over the FINITE depth values only.
///
/// The Python ground truth summarizes `depth[np.isfinite(depth)]`, so the C++
/// side must filter identically. cv::minMaxLoc and cv::mean do not skip NaN /
/// Inf, so a single bad pixel would turn the C++ mean into NaN while Python
/// silently ignored it — a mismatch that looks like a depth error but isn't.
struct DepthStats {
    double minDepth{0.0};
    double maxDepth{0.0};
    double meanDepth{0.0};
    double medianDepth{0.0};
    size_t finiteCount{0};
};

DepthStats computeStats(const cv::Mat& depth) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(depth.total()));
    for (int r = 0; r < depth.rows; ++r) {
        const float* row = depth.ptr<float>(r);
        for (int c = 0; c < depth.cols; ++c) {
            if (std::isfinite(row[c])) values.push_back(row[c]);
        }
    }

    DepthStats stats;
    stats.finiteCount = values.size();
    if (values.empty()) return stats;

    double sum = 0.0;
    stats.minDepth = values[0];
    stats.maxDepth = values[0];
    for (const float v : values) {
        sum += v;
        stats.minDepth = std::min(stats.minDepth, static_cast<double>(v));
        stats.maxDepth = std::max(stats.maxDepth, static_cast<double>(v));
    }
    stats.meanDepth = sum / static_cast<double>(values.size());

    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const double hi = values[mid];
    if (values.size() % 2 == 1) {
        stats.medianDepth = hi;
    } else {
        // Even count — average the two central order statistics, as NumPy does.
        std::nth_element(values.begin(), values.begin() + mid - 1, values.begin() + mid);
        stats.medianDepth = 0.5 * (hi + values[mid - 1]);
    }
    return stats;
}

void runInference(const std::string& modelPath,
                  const std::vector<std::string>& imageFiles,
                  std::map<std::string, DepthImageResult>& inferenceResults) {
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Device: GPU (TensorRT)" << std::endl;

    YOLODepthEstimator estimator(modelPath);

    for (const auto& imagePath : imageFiles) {
        std::cout << "\nProcessing: " << imagePath << std::endl;

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Warning: could not read " << imagePath << ", skipping" << std::endl;
            continue;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        cv::Mat depth = estimator.estimate(image);
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        if (depth.empty()) {
            std::cerr << "Warning: empty depth map for " << imagePath << ", skipping" << std::endl;
            continue;
        }

        const DepthStats stats = computeStats(depth);
        if (stats.finiteCount == 0) {
            std::cerr << "Warning: depth map for " << imagePath
                      << " has no finite values, skipping" << std::endl;
            continue;
        }

        DepthImageResult res;
        res.width       = image.cols;
        res.height      = image.rows;
        res.minDepth    = stats.minDepth;
        res.maxDepth    = stats.maxDepth;
        res.meanDepth   = stats.meanDepth;
        res.medianDepth = stats.medianDepth;

        for (const auto& pt : samplePoints(image.cols, image.rows)) {
            res.samples.push_back({pt.x, pt.y, depth.at<float>(pt.y, pt.x)});
        }

        std::cout << "Depth time: " << duration.count() << " ms"
                  << ", range: " << stats.minDepth << " m .. " << stats.maxDepth << " m"
                  << ", mean: " << stats.meanDepth << " m" << std::endl;

        inferenceResults[imagePath] = std::move(res);
    }
}

void toJson(const std::map<std::string, ResultsDepth>& results,
            const std::string& basePath, json& outputJson) {
    for (const auto& [modelName, result] : results) {
        outputJson[modelName] = json();
        outputJson[modelName]["weights_path"] = result.weightsPath.substr(basePath.length());
        outputJson[modelName]["task"] = result.task;
        outputJson[modelName]["results"] = json::array();

        for (const auto& [imagePath, res] : result.inferenceResults) {
            json entry;
            entry["image_path"] = imagePath.substr(basePath.length());
            entry["width"]      = res.width;
            entry["height"]     = res.height;
            entry["min"]        = res.minDepth;
            entry["max"]        = res.maxDepth;
            entry["mean"]       = res.meanDepth;
            entry["median"]     = res.medianDepth;
            entry["samples"]    = json::array();

            for (const auto& s : res.samples) {
                json sample;
                sample["x"] = s.x;
                sample["y"] = s.y;
                sample["depth"] = s.depth;
                entry["samples"].push_back(sample);
            }
            outputJson[modelName]["results"].push_back(entry);
        }
    }
}

int main() {
    std::cout << "=== YOLOs-TRT Depth Estimation Test ===" << std::endl;

    const std::string basePath   = XSTRING(BASE_PATH_DEPTH);
    const std::string imagesPath = basePath + "data/images/";
    const std::string modelsPath = basePath + "models/";
    const std::string resultsPath = basePath + "results/";

    const std::unordered_map<std::string, std::string> paths_map = {
        {"images", imagesPath}, {"models", modelsPath}
    };
    if (!validatePaths(paths_map)) return -1;

    std::vector<std::string> imageFiles;
    if (!loadImages(imagesPath, imageFiles)) return -1;

    std::vector<std::string> modelFiles;
    findModels(modelsPath, modelFiles);
    if (modelFiles.empty()) {
        std::cerr << "No depth engines found in: " << modelsPath << std::endl;
        return -1;
    }

    if (!fs::exists(resultsPath)) fs::create_directories(resultsPath);
    const std::string resultsFilePath = resultsPath + "results_cpp.json";
    if (fs::exists(resultsFilePath)) fs::remove(resultsFilePath);

    std::map<std::string, ResultsDepth> allResults;

    for (const auto& modelPath : modelFiles) {
        const std::string modelName = fs::path(modelPath).stem().string();
        allResults[modelName] = ResultsDepth{modelPath, "depth", {}};

        std::cout << "\n======== Running: " << modelName << " ========" << std::endl;
        runInference(modelPath, imageFiles, allResults[modelName].inferenceResults);
    }

    json outputJson;
    toJson(allResults, basePath, outputJson);

    std::ofstream file(resultsFilePath);
    file << std::setw(2) << outputJson << std::endl;

    std::cout << "Results saved to: " << resultsFilePath << std::endl;
    return 0;
}
