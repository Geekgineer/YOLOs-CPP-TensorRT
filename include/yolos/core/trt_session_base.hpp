#pragma once

// ============================================================================
// TensorRT Session Base — High-Performance Edition
// ============================================================================

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// === Memory Management ===
#include "MemoryArena.hpp"

#include "yolos/core/trt_utils.hpp"
#include "yolos/core/cuda_preprocessing.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"

namespace yolos {

// ============================================================================
// TrtSessionBase - With Advanced Memory Management
// ============================================================================

class TrtSessionBase {
public:
    TrtSessionBase(const std::string& enginePath,
                   int dlaCore = -1,
                   int warmupRuns = 10);

    virtual ~TrtSessionBase();

    // Advanced async preprocess using pinned memory + arena
    cudaError_t asyncPreprocess(const cv::Mat& src, float* d_dst, int dstW, int dstH);

protected:
    nvinfer1::ICudaEngine* m_engine = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;
    cudaStream_t m_mainStream = nullptr;

    // === Advanced Memory Management (Your Contribution) ===
    core::MemoryArena m_pinnedArena{512ULL * 1024 * 1024};   // 512 MB Pinned Arena

    // Pinned staging buffer for fast H2D transfers
    uint8_t* m_pinnedStagingBuffer = nullptr;
    size_t   m_stagingBufferSize = 0;

    // Zero-copy support
    void* m_zeroCopyPtr = nullptr;

private:
    void initEngine(const std::string& enginePath, int dlaCore);
    void allocGpuPreprocessBuffers();
    void warmUp(int runs);
    void captureInferenceGraph();

    // Clean up
    void freeMemoryResources();
};

// ============================================================================
// Implementation
// ============================================================================

TrtSessionBase::TrtSessionBase(const std::string& enginePath, int dlaCore, int warmupRuns)
{
    initEngine(enginePath, dlaCore);

    // === Advanced Memory Setup ===
    cudaStreamCreate(&m_mainStream);

    // Allocate pinned memory for async H2D (this is the real shit)
    cudaMallocHost(&m_pinnedStagingBuffer, 1920 * 1080 * 3);  // Full HD buffer
    m_stagingBufferSize = 1920 * 1080 * 3;

    m_pinnedArena.reset();

    allocGpuPreprocessBuffers();
    warmUp(warmupRuns);
    captureInferenceGraph();
}

TrtSessionBase::~TrtSessionBase()
{
    freeMemoryResources();
}

cudaError_t TrtSessionBase::asyncPreprocess(const cv::Mat& src, float* d_dst, int dstW, int dstH)
{
    if (!m_pinnedStagingBuffer) {
        return cudaErrorMemoryAllocation;
    }

    // Async H2D using pinned memory
    cudaMemcpyAsync(m_pinnedStagingBuffer, src.data, src.total() * 3,
                    cudaMemcpyHostToDevice, m_mainStream);

    // Call optimized CUDA kernel
    yolos::cuda::letterboxPreprocess(
        m_pinnedStagingBuffer,
        src.cols, src.rows,
        d_dst, dstW, dstH,
        m_mainStream
    );

    return cudaGetLastError();
}

void TrtSessionBase::freeMemoryResources()
{
    if (m_pinnedStagingBuffer) {
        cudaFreeHost(m_pinnedStagingBuffer);
        m_pinnedStagingBuffer = nullptr;
    }
    if (m_mainStream) {
        cudaStreamDestroy(m_mainStream);
        m_mainStream = nullptr;
    }
}

} // namespace yolos