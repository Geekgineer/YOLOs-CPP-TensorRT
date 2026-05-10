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

// === NEW: Our Memory Arena ===
#include "MemoryArena.hpp"

#include "yolos/core/trt_utils.hpp"
#include "yolos/core/cuda_preprocessing.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"

namespace yolos {

// ============================================================================
// TrtSessionBase
// ============================================================================

class TrtSessionBase {
public:
    TrtSessionBase(const std::string& enginePath,
                   int dlaCore = -1,
                   int warmupRuns = 10)
    {
        initEngine(enginePath, dlaCore);
        
        // === NEW: Initialize Memory Arena ===
        m_memoryArena.reset();   // Ready for use

        allocGpuPreprocessBuffers();
        warmUp(warmupRuns);
        captureInferenceGraph();
    }

    // ... (rest of the class remains same)

protected:
    // ... existing members ...

    // ── NEW: Custom Memory Arena for optimization ───────────────────────
    core::MemoryArena m_memoryArena{256ULL * 1024 * 1024};  // 256 MB arena

private:
    // ... rest of private members ...