#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

using json = nlohmann::json;

// ============================================================================
// Tolerances
// ============================================================================
// The C++ path runs an fp16 TensorRT engine while the Ultralytics ground truth
// runs fp32, and the two resize/crop paths are independent implementations of
// the same letterbox geometry. Depth is metric (meters) on an open-ended log
// scale, so a relative tolerance is the meaningful check, with a small absolute
// floor so near-camera pixels (a few centimeters) do not trip on rounding.
constexpr double REL_TOLERANCE = 0.10;  // 10% relative
constexpr double ABS_TOLERANCE = 0.15;  // meters — absolute floor

/// @brief True when two depths agree within either tolerance.
static bool depthClose(double a, double b) {
    const double diff = std::abs(a - b);
    if (diff <= ABS_TOLERANCE) return true;
    const double scale = std::max(std::abs(a), std::abs(b));
    return scale > 0.0 && (diff / scale) <= REL_TOLERANCE;
}

static json read_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("File not found: " + path);
    }
    json j;
    f >> j;
    return j;
}

// Locate the C++ result entry for a given image path. The C++ driver stores
// per-image results keyed by path, so positional indices are NOT comparable
// between the two JSON files — always match on image_path.
static const json* findByImagePath(const json& cpp_results, const std::string& image_path) {
    for (const auto& entry : cpp_results) {
        if (entry.value("image_path", "") == image_path) return &entry;
    }
    return nullptr;
}

class DepthResultsFixture : public ::testing::Test {
protected:
    json results_ultralytics;
    json results_cpp;
    std::string basePath = XSTRING(BASE_PATH_DEPTH);

    void SetUp() override {
        ASSERT_NO_THROW(results_ultralytics = read_json(basePath + "results/results_ultralytics.json"));
        ASSERT_NO_THROW(results_cpp = read_json(basePath + "results/results_cpp.json"));
    }
};

TEST_F(DepthResultsFixture, ResultsNotEmpty) {
    ASSERT_FALSE(results_ultralytics.empty()) << "results_ultralytics is empty";
    ASSERT_FALSE(results_cpp.empty()) << "results_cpp is empty";
}

TEST_F(DepthResultsFixture, CompareModelsNames) {
    std::set<std::string> models_ultra, models_cpp;
    for (auto& el : results_ultralytics.items()) models_ultra.insert(el.key());
    for (auto& el : results_cpp.items()) models_cpp.insert(el.key());
    for (const auto& name : models_ultra) {
        ASSERT_TRUE(models_cpp.count(name)) << "Model " << name << " is missing in results_cpp";
    }
}

TEST_F(DepthResultsFixture, CompareImagesCounts) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();
        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];
        ASSERT_EQ(ultra_results.size(), cpp_results.size())
            << "Number of results mismatch for model " << model_name;
    }
}

TEST_F(DepthResultsFixture, CompareImagesPaths) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();
        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];
        for (size_t i = 0; i < ultra_results.size(); ++i) {
            const std::string path_ultra = ultra_results[i].value("image_path", "");
            ASSERT_NE(findByImagePath(cpp_results, path_ultra), nullptr)
                << "Image " << path_ultra << " is missing from results_cpp for model " << model_name;
        }
    }
}

// The depth map must come back at the ORIGINAL image resolution — this is the
// test that catches a broken un-letterbox / resize step.
TEST_F(DepthResultsFixture, CompareDepthMapDimensions) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();
        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {
            const std::string image_path = ultra_results[i].value("image_path", "");
            const json* cpp_entry = findByImagePath(cpp_results, image_path);
            ASSERT_NE(cpp_entry, nullptr)
                << "Image " << image_path << " is missing from results_cpp for model " << model_name;

            ASSERT_EQ(ultra_results[i].value("width", -1), cpp_entry->value("width", -2))
                << "Depth map width mismatch for model " << model_name << ", image: " << image_path;
            ASSERT_EQ(ultra_results[i].value("height", -1), cpp_entry->value("height", -2))
                << "Depth map height mismatch for model " << model_name << ", image: " << image_path;
        }
    }
}

// Depth must be positive and physically plausible. This catches a model whose
// log-affine calibration was lost in export (which shows up as near-zero or
// absurdly large values) independently of the ground-truth comparison.
TEST_F(DepthResultsFixture, DepthValuesArePlausible) {
    for (auto& el : results_cpp.items()) {
        const std::string& model_name = el.key();
        auto& cpp_results = el.value()["results"];

        for (const auto& entry : cpp_results) {
            const std::string image_path = entry.value("image_path", "");
            const double minDepth = entry.value("min", -1.0);
            const double maxDepth = entry.value("max", -1.0);

            ASSERT_GT(minDepth, 0.0)
                << "Non-positive minimum depth for model " << model_name << ", image: " << image_path;
            ASSERT_LT(maxDepth, 1000.0)
                << "Implausibly large maximum depth (" << maxDepth << " m) for model "
                << model_name << ", image: " << image_path;
            ASSERT_GE(maxDepth, minDepth)
                << "Inverted depth range for model " << model_name << ", image: " << image_path;
        }
    }
}

TEST_F(DepthResultsFixture, CompareDepthStatistics) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();
        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {
            const std::string image_path = ultra_results[i].value("image_path", "");
            const json* cpp_entry = findByImagePath(cpp_results, image_path);
            ASSERT_NE(cpp_entry, nullptr)
                << "Image " << image_path << " is missing from results_cpp for model " << model_name;

            for (const char* stat : {"mean", "median", "min", "max"}) {
                const double ultra = ultra_results[i].value(stat, 0.0);
                const double cpp = cpp_entry->value(stat, 0.0);
                ASSERT_TRUE(depthClose(ultra, cpp))
                    << "Depth " << stat << " mismatch for model " << model_name
                    << ", image: " << image_path
                    << ": ultralytics: " << ultra << " m != cpp: " << cpp << " m";
            }
        }
    }
}

// Per-pixel agreement on a deterministic lattice. This is the test that catches
// a spatially misaligned depth map (flipped axis, wrong crop offset) which
// whole-image statistics alone would happily pass.
TEST_F(DepthResultsFixture, CompareDepthSamplePoints) {
    for (auto& el : results_ultralytics.items()) {
        const std::string& model_name = el.key();
        auto& ultra_results = el.value()["results"];
        auto& cpp_results = results_cpp[model_name]["results"];

        for (size_t i = 0; i < ultra_results.size(); ++i) {
            const std::string image_path = ultra_results[i].value("image_path", "");
            const json* cpp_entry = findByImagePath(cpp_results, image_path);
            ASSERT_NE(cpp_entry, nullptr)
                << "Image " << image_path << " is missing from results_cpp for model " << model_name;

            const auto ultra_samples = ultra_results[i].value("samples", json::array());
            const auto cpp_samples = cpp_entry->value("samples", json::array());

            ASSERT_FALSE(ultra_samples.empty())
                << "No ground-truth samples for model " << model_name << ", image: " << image_path;
            ASSERT_EQ(ultra_samples.size(), cpp_samples.size())
                << "Sample count mismatch for model " << model_name << ", image: " << image_path;

            // Both sides generate the lattice from the same formula, so the
            // sample order and coordinates must line up exactly.
            size_t mismatches = 0;
            for (size_t s = 0; s < ultra_samples.size(); ++s) {
                ASSERT_EQ(ultra_samples[s].value("x", -1), cpp_samples[s].value("x", -2))
                    << "Sample x mismatch at index " << s << " for image: " << image_path;
                ASSERT_EQ(ultra_samples[s].value("y", -1), cpp_samples[s].value("y", -2))
                    << "Sample y mismatch at index " << s << " for image: " << image_path;

                const double d_ultra = ultra_samples[s].value("depth", 0.0);
                const double d_cpp = cpp_samples[s].value("depth", 0.0);
                if (!depthClose(d_ultra, d_cpp)) {
                    ++mismatches;
                    std::cout << "  [warn] sample " << s
                              << " at (" << ultra_samples[s].value("x", -1) << ", "
                              << ultra_samples[s].value("y", -1) << "): "
                              << d_ultra << " m vs " << d_cpp << " m" << std::endl;
                }
            }

            // Allow a small number of outliers: individual pixels on a depth
            // discontinuity can legitimately land on either side of an edge
            // after two independent resize paths.
            const size_t maxMismatches = ultra_samples.size() / 5;  // 20%
            ASSERT_LE(mismatches, maxMismatches)
                << mismatches << " of " << ultra_samples.size()
                << " sample points disagree beyond tolerance for model "
                << model_name << ", image: " << image_path;
        }
    }
}
