#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace yolos::core {

class PinnedMemory {
public:
    // Allocate pinned (page-locked) memory - faster for H2D transfers
    static void* allocate(size_t size_bytes) {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, size_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMallocHost failed: " + std::string(cudaGetErrorString(err)));
        }
        return ptr;
    }

    static void deallocate(void* ptr) {
        if (ptr != nullptr) {
            cudaFreeHost(ptr);
        }
    }

    // Async Host to Device copy using stream
    static void copyH2DAsync(void* d_dst, const void* h_src, size_t size_bytes, cudaStream_t stream) {
        cudaMemcpyAsync(d_dst, h_src, size_bytes, cudaMemcpyHostToDevice, stream);
    }

    // Async Device to Host copy
    static void copyD2HAsync(void* h_dst, const void* d_src, size_t size_bytes, cudaStream_t stream) {
        cudaMemcpyAsync(h_dst, d_src, size_bytes, cudaMemcpyDeviceToHost, stream);
    }
};

} // namespace yolos::core