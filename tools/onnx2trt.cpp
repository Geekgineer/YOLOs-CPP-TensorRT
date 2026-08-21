/**
 * @file onnx2trt.cpp
 * @brief Standalone ONNX -> TensorRT engine builder.
 *
 * Exists because neither of the usual conversion routes is dependable:
 *   - `trtexec` is NOT shipped by the libnvinfer-dev / tensorrt-dev apt
 *     packages. It comes with the samples package or the TensorRT tarball, so
 *     on a plain apt install there is no trtexec anywhere on the system.
 *   - The Python converter needs the `tensorrt` wheel (~4 GB with its CUDA
 *     dependencies), pinned to the exact system TensorRT version, because
 *     engines are version-locked.
 *
 * This tool links the TensorRT libraries the project already requires, so it
 * builds engines with zero extra dependencies and no version-skew risk.
 *
 * Usage:
 *   ./onnx2trt <model.onnx> <out.engine> [--fp16] [--int8] [--workspace MiB]
 *
 * Author: YOLOs-TRT Team
 */

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

Logger gLogger;

void printDims(const char* label, const char* name, const nvinfer1::Dims& d) {
    std::cout << "  " << label << " " << name << " [";
    for (int i = 0; i < d.nbDims; ++i) {
        std::cout << d.d[i] << (i + 1 < d.nbDims ? ", " : "");
    }
    std::cout << "]" << std::endl;
}

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " <model.onnx> <out.engine> [--fp16] [--int8] [--workspace MiB]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    const std::string onnxPath   = argv[1];
    const std::string enginePath = argv[2];

    bool fp16 = false;
    bool int8 = false;
    size_t workspaceMiB = 4096;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--fp16") {
            fp16 = true;
        } else if (arg == "--int8") {
            int8 = true;
        } else if (arg == "--workspace" && i + 1 < argc) {
            workspaceMiB = static_cast<size_t>(std::strtoul(argv[++i], nullptr, 10));
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        std::cerr << "failed to create TensorRT builder" << std::endl;
        return 1;
    }

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto parser  = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, gLogger));

    if (!parser->parseFromFile(onnxPath.c_str(),
                              static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "failed to parse " << onnxPath << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "  " << parser->getError(i)->desc() << std::endl;
        }
        return 1;
    }

    std::cout << "--- network IO ---" << std::endl;
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto* t = network->getInput(i);
        printDims("in ", t->getName(), t->getDimensions());
    }
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        auto* t = network->getOutput(i);
        printDims("out", t->getName(), t->getDimensions());
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceMiB * 1024ULL * 1024ULL);

    if (fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "precision: fp16" << std::endl;
    }
    if (int8) {
        // No calibrator here: use trt-files/scripts/convert_to_tensorrt.py for
        // calibrated INT8. This flag alone gives INT8 without calibration data,
        // which is only useful for throughput experiments.
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::cout << "precision: int8 (uncalibrated)" << std::endl;
    }
    if (!fp16 && !int8) {
        std::cout << "precision: fp32" << std::endl;
    }

    std::cout << "building engine (this can take a few minutes)..." << std::endl;
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        std::cerr << "engine build failed" << std::endl;
        return 1;
    }

    std::ofstream out(enginePath, std::ios::binary);
    if (!out) {
        std::cerr << "cannot write " << enginePath << std::endl;
        return 1;
    }
    out.write(static_cast<const char*>(plan->data()), static_cast<std::streamsize>(plan->size()));
    out.close();

    std::cout << "wrote " << enginePath << " (" << plan->size() / (1024 * 1024) << " MB)" << std::endl;
    return 0;
}
