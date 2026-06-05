#include "yolos/core/trt_session_base.hpp"
#include <fstream>
#include <vector>
#include <iostream>

namespace yolos {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

TrtSessionBase::TrtSessionBase(const std::string& enginePath, int dlaCore) {
    initEngine(enginePath, dlaCore);
}

TrtSessionBase::~TrtSessionBase() {
    freeMemoryResources();
}

void TrtSessionBase::initEngine(const std::string& enginePath, int dlaCore) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open engine: " << enginePath << std::endl;
        return;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    m_runtime.reset(nvinfer1::createInferRuntime(gLogger));
    if (m_runtime) {
        m_engine.reset(m_runtime->deserializeCudaEngine(buffer.data(), size));
        if (m_engine) {
            m_context.reset(m_engine->createExecutionContext());
        }
    }
}

void TrtSessionBase::freeMemoryResources() {
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
}

// Corrected placeholders to match header
cudaError_t TrtSessionBase::asyncPreprocess(const cv::Mat& src, float* d_dst, int dstW, int dstH) {
    // TODO: Real CUDA preprocessing later
    return cudaSuccess;
}

void TrtSessionBase::inferGpu(const cv::Mat& image) {
    // TODO: Real inference later
}

void TrtSessionBase::warmUp(int runs) {
    // TODO: Warmup later
}

} // namespace yolos
